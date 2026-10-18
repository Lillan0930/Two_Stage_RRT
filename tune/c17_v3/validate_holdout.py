#!/usr/bin/env python
"""C17 + ABMIL + v3 —— **GPU7 hold-out validation worker**。

职责
------
某个组合的 12 条 Optuna trial 跑满、``<组合>/best_params.json`` 落盘之后，
本 worker 领取该组合，用 best params 在 **hold-out seeds = 3611 / 6430 / 39998**
上各跑一次正式训练，然后与这三个 seed 上**已有的原始正式 baseline** 做
paired comparison。

    GPU7 -> 对已完成搜索的组合立即验证
            读完一个组合 -> 继续领取下一个
            暂时没有可验证的组合 -> 等待，**不参与 Optuna study**

为什么 hold-out 必须是搜索没用过的 seed
----------------------------------------
搜索阶段 12 条 trial 都跑在 search seed=42 上，best params 是**按那个 seed 的
val AUC** 选出来的。如果验证还用 42，"新参数更好"里就混进了"这一轮恰好偏好
新参数"的选择偏差 —— 同一份数据既调参又验证。3611 / 6430 / 39998 在搜索期
从未出现（``optuna_v3.py`` 里有硬断言挡着），所以这三个 seed 上的
delta 才是对"调参是否真的有用"的**诚实**估计。

口径与搜索、与历史正式 run 三者同源
--------------------------------------
    epochs=30  patience=15  min_epochs=10  T_max=30  eta_min=lr*0.01
用的还是 ``ours_main_c17_abmil.build_config()``、同一份 ``C17_features``
原路径、同一个 patient-level 划分（patient_000..099 / patient_100..199）。
**不重新分折、不复制特征、不改 v3 结构。**

配对比较的 baseline 是**现成的**（``C17+abmil/<组合>/seed<seed>/metrics.json``），
不重跑 —— 只读。

运行方式
----------
    # 单次：把所有已就绪的组合验一遍后退出
    python tune/c17_v3/validate_holdout.py --gpu 7 --once

    # 常驻 worker：轮询等待搜索完成（run_optuna.sh 用的就是这个）
    python tune/c17_v3/validate_holdout.py --gpu 7

    # 只验一个组合
    python tune/c17_v3/validate_holdout.py --combo HE+Ki67 --gpu 7 --once

磁盘
------
每个 hold-out run 的 checkpoint 在指标取出后删除（§7：每个组合只保留
``best_model.pt`` 这一份）。要留就加 ``--keep-checkpoints``。
所有产物都写在 ``tune/c17_v3/`` 子树内。
"""
import argparse
import csv
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                            # noqa: E402
# ⚠ **optuna 必须在 torch 之前导入**（与 optuna_v3.py 的顺序一致）。
# optuna 会拉起 sqlite3 -> _sqlite3 -> conda 的 libicui18n.so.78；若先导入
# torch，torch 自带的 libstdc++ 会先占住符号空间，conda 的 libicui18n 就会
# 去撞系统 /lib/x86_64-linux-gnu/libstdc++.so.6，报
# `CXXABI_1.3.15 not found`。踩过一次。run_optuna.sh 里另外还设了
# LD_LIBRARY_PATH 兜底，两条都留着。
import optuna                                                 # noqa: E402

import torch                                                  # noqa: E402

import train as T                                             # noqa: E402
import ours_main_c17_abmil as O                               # noqa: E402
from run_one_c17_seed import seed_everything                  # noqa: E402
from run_one_c17_seed import HistoryTrainer                   # noqa: E402
from optuna_v3 import (COMBOS, HOLDOUT_SEEDS, SEARCH_EPOCHS,  # noqa: E402
                       SEARCH_PATIENCE, SEARCH_MIN_EPOCHS, ETA_MIN_RATIO,
                       APPLY, TUNE_ROOT, modalities_of, set_nested,
                       derive_eta_min, verify_reached, close_logging)

HOLDOUT_ROOT_NAME = "holdout"


def now():
    return datetime.now().isoformat(timespec="seconds")


def original_metrics(combo, seed):
    """原始正式 run 在这条 seed 上的 metrics.json（**只读，不重跑**）。"""
    p = O.seed_dir(combo, seed) / "metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def run_one_seed(combo, seed, params, log, keep_ckpt=False):
    """用 best params 在一个 hold-out seed 上跑一次**正式协议**训练。"""
    slug = COMBOS[combo][0]
    out = TUNE_ROOT / slug / HOLDOUT_ROOT_NAME / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    cfg = O.build_config(combo, modalities_of(combo), seed)
    for name, path in APPLY.items():
        set_nested(cfg, path, params[name])
    cfg["training"]["scheduler"]["eta_min"] = derive_eta_min(params["learning_rate"])
    cfg["training"]["num_epochs"] = SEARCH_EPOCHS
    cfg["training"]["early_stopping"]["patience"] = SEARCH_PATIENCE
    cfg["training"]["early_stopping"]["min_epochs"] = SEARCH_MIN_EPOCHS
    cfg["output"] = {"save_dir": str(out), "log_dir": str(out / "logs"),
                     "img_dir": str(out / "img")}
    cfg["experiment"]["role"] = ("Optuna hold-out validation（best params；"
                                 "该 seed 搜索期未使用）")
    cfg["experiment"]["holdout_validation"] = {
        "combination": combo, "seed": seed, "params": params,
        "protocol": (f"epochs={SEARCH_EPOCHS} patience={SEARCH_PATIENCE} "
                     f"min_epochs={SEARCH_MIN_EPOCHS} T_max={SEARCH_EPOCHS} "
                     f"eta_min=lr*{ETA_MIN_RATIO}"),
    }
    O.dump_config_yaml(cfg, out / "config.yaml")

    seed_everything(seed)
    logger, timestamp = T.setup_logging(str(out / "logs"),
                                        f"holdout_{slug}_s{seed}")
    logger.info(f"[holdout] combo={combo} seed={seed} "
                f"physical_gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '?')} "
                f"params={json.dumps(params)}")
    logger.info(f"[holdout] feature_base_dir={cfg['data']['feature_base_dir']} "
                f"val_start={cfg['data']['val_start']} "
                f"sample_seed={cfg['data']['sample_seed']} "
                f"env_seed={cfg['environment']['seed']}")

    trainer = HistoryTrainer(cfg, logger, timestamp)
    model, _ = trainer.train()

    reached = verify_reached(model, trainer.optimizer, params)
    eta_actual = float(getattr(trainer, "eta_min_actual", float("nan")))
    eta_want = derive_eta_min(params["learning_rate"])
    assert abs(eta_actual - eta_want) < 1e-18, \
        f"eta_min 未随 lr 生效: {eta_actual} vs {eta_want}"
    reached["eta_min"] = eta_actual
    logger.info(f"[holdout] 参数落点核对通过: {json.dumps(reached)}")

    metrics = O.primary_metrics(trainer.final_metrics)
    auc = float(trainer.best_val_auc)
    rec = {
        "combination": combo, "seed": seed,
        "auc": auc,
        "best_epoch": int(trainer.best_epoch) + 1,
        "epochs_run": len(trainer.epoch_rows),
        "metrics": metrics,
        "params": params,
        "params_reached": reached,
    }
    (out / "holdout_metrics.json").write_text(json.dumps(rec, indent=2) + "\n")

    del model, trainer
    close_logging()
    gc.collect()
    torch.cuda.empty_cache()

    if not keep_ckpt:
        for junk in ("best_model.pt", "last_model.pt"):
            p = out / junk
            if p.is_file():
                p.unlink()
    return rec


def paired_compare(combo, runs, log):
    """新 params vs 原始正式 baseline，在**同一批 hold-out seed** 上配对。"""
    pairs = []
    for r in runs:
        om = original_metrics(combo, r["seed"])
        if om is None:
            log(f"[warn ] {combo} seed={r['seed']} 缺原始 baseline，跳过该配对")
            continue
        pairs.append({
            "seed": r["seed"],
            "orig_auc": float(om["AUC"]),
            "new_auc": float(r["auc"]),
            "delta": float(r["auc"]) - float(om["AUC"]),
            "orig_best_epoch": int(om.get("best_epoch", 0)),
            "new_best_epoch": int(r["best_epoch"]),
        })
    if not pairs:
        return {"pairs": [], "n_pairs": 0}

    d = np.array([p["delta"] for p in pairs], dtype=float)
    out = {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "orig_auc_mean": float(np.mean([p["orig_auc"] for p in pairs])),
        "new_auc_mean": float(np.mean([p["new_auc"] for p in pairs])),
        "delta_mean": float(d.mean()),
        "delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "n_seeds_improved": int((d > 0).sum()),
        "n_seeds_worse": int((d < 0).sum()),
    }
    # n=3 的配对检验功效极低，只作参考，不拿来下结论
    if len(d) > 1:
        try:
            from scipy import stats
            t, p = stats.ttest_rel([x["new_auc"] for x in pairs],
                                   [x["orig_auc"] for x in pairs])
            out["paired_ttest"] = {"t": float(t), "p": float(p),
                                   "note": "n=3，功效极低，仅作参考"}
        except Exception:                                      # noqa: BLE001
            pass
    return out


def validate_one(combo, gpu, log, keep_ckpt=False):
    """验证一个组合，写 holdout_result.json。返回结果 dict。"""
    slug = COMBOS[combo][0]
    combo_dir = TUNE_ROOT / slug
    bp_path = combo_dir / "best_params.json"
    res_path = combo_dir / "holdout_result.json"

    bp = json.loads(bp_path.read_text())
    params = bp["params"]
    seed_used = bp.get("search_seed", bp.get("seed"))
    assert seed_used not in HOLDOUT_SEEDS, \
        f"{combo} 的搜索 seed {seed_used} 落在 hold-out 集里，结果不可信"

    log(f"[v-begin] combo={combo} search_seed={seed_used} gpu={gpu} "
        f"best_trial={bp.get('best_trial')} best_value={bp.get('best_value')}")
    log(f"[v-begin] best_params={json.dumps(params)}")
    log(f"[v-begin] holdout_seeds={list(HOLDOUT_SEEDS)} "
        f"protocol=epochs{SEARCH_EPOCHS}/patience{SEARCH_PATIENCE}/"
        f"min_epochs{SEARCH_MIN_EPOCHS}")

    runs = []
    for s in HOLDOUT_SEEDS:
        log(f"[v-run  ] combo={combo} gpu={gpu} holdout_seed={s} 开始")
        try:
            rec = run_one_seed(combo, s, params, log, keep_ckpt)
            runs.append(rec)
            log(f"[v-run  ] combo={combo} gpu={gpu} holdout_seed={s} "
                f"DONE auc={rec['auc']:.6f} best_epoch={rec['best_epoch']} "
                f"epochs={rec['epochs_run']}")
        except BaseException:                                  # noqa: BLE001
            log(f"[v-run  ] combo={combo} holdout_seed={s} FAILED\n"
                f"{traceback.format_exc()}")
            gc.collect()
            torch.cuda.empty_cache()

    cmp_ = paired_compare(combo, runs, log)
    result = {
        "combination": combo,
        "search_seed": seed_used,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "protocol": (f"epochs={SEARCH_EPOCHS} patience={SEARCH_PATIENCE} "
                     f"min_epochs={SEARCH_MIN_EPOCHS} T_max={SEARCH_EPOCHS} "
                     f"eta_min=lr*{ETA_MIN_RATIO}"),
        "best_params": params,
        "best_trial": bp.get("best_trial"),
        "search_best_value": bp.get("best_value"),
        "runs": runs,
        "paired_vs_original": cmp_,
        "verdict": ("improved" if cmp_.get("delta_mean", 0) > 0
                    else "not_improved"),
        "validated_at": now(),
        "validated_on_gpu": gpu,
    }
    res_path.write_text(json.dumps(result, indent=2) + "\n")

    # 把验证状态回填到 best_params.json，方便搜索侧/人肉查看
    try:
        bp["holdout_validation"] = {
            "status": "done",
            "delta_mean": cmp_.get("delta_mean"),
            "n_seeds_improved": cmp_.get("n_seeds_improved"),
            "verdict": result["verdict"],
            "result_file": str(res_path),
            "validated_at": result["validated_at"],
        }
        bp_path.write_text(json.dumps(bp, indent=2) + "\n")
    except Exception as exc:                                   # noqa: BLE001
        log(f"[warn ] 回填 best_params.json 失败: {exc}")

    log(f"[v-done ] combo={combo} gpu={gpu} "
        f"orig_mean={cmp_.get('orig_auc_mean'):.6f} "
        f"new_mean={cmp_.get('new_auc_mean'):.6f} "
        f"delta={cmp_.get('delta_mean'):+.6f}±{cmp_.get('delta_std'):.6f} "
        f"改善 {cmp_.get('n_seeds_improved')}/{cmp_.get('n_pairs')} "
        f"verdict={result['verdict']}")
    log(f"[v-done ] combo={combo} holdout_validation=done "
        f"-> {res_path}")
    return result


def write_summary_holdout():
    cols = ["combination", "search_seed", "best_trial", "search_best_value",
            "n_pairs", "orig_auc_mean", "new_auc_mean", "delta_mean",
            "delta_std", "n_seeds_improved", "n_seeds_worse", "verdict",
            "holdout_3611_new", "holdout_6430_new", "holdout_39998_new",
            "holdout_3611_delta", "holdout_6430_delta", "holdout_39998_delta",
            *APPLY.keys()]
    rows = []
    for combo, (slug, _) in COMBOS.items():
        p = TUNE_ROOT / slug / "holdout_result.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        c = r.get("paired_vs_original", {})
        by_seed = {x["seed"]: x for x in c.get("pairs", [])}
        row = {
            "combination": combo,
            "search_seed": r.get("search_seed"),
            "best_trial": r.get("best_trial"),
            "search_best_value": r.get("search_best_value"),
            "n_pairs": c.get("n_pairs"),
            "orig_auc_mean": c.get("orig_auc_mean"),
            "new_auc_mean": c.get("new_auc_mean"),
            "delta_mean": c.get("delta_mean"),
            "delta_std": c.get("delta_std"),
            "n_seeds_improved": c.get("n_seeds_improved"),
            "n_seeds_worse": c.get("n_seeds_worse"),
            "verdict": r.get("verdict"),
            **{k: r["best_params"].get(k) for k in APPLY},
        }
        for s in HOLDOUT_SEEDS:
            row[f"holdout_{s}_new"] = by_seed.get(s, {}).get("new_auc")
            row[f"holdout_{s}_delta"] = by_seed.get(s, {}).get("delta")
        rows.append(row)
    if not rows:
        return None
    out = TUNE_ROOT / "summary_holdout.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default=None, choices=list(COMBOS),
                    help="只验证一个组合（默认：所有已就绪的组合）")
    ap.add_argument("--gpu", type=int, default=None, help="物理卡号（仅日志用）")
    ap.add_argument("--once", action="store_true",
                    help="只扫一轮就退出（默认常驻轮询）")
    ap.add_argument("--poll", type=int, default=60, help="轮询间隔秒（默认 60）")
    ap.add_argument("--keep-checkpoints", action="store_true",
                    help="保留 hold-out 的 best_model.pt（默认删，§7 磁盘口径）")
    ap.add_argument("--force", action="store_true",
                    help="已有 holdout_result.json 也重跑")
    args = ap.parse_args()

    gpu = args.gpu
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu))
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = TUNE_ROOT / "tmp" / "holdout_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a")

    def log(msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"[worker] hold-out validation worker 启动 gpu={phys} "
        f"holdout_seeds={list(HOLDOUT_SEEDS)} once={args.once} "
        f"keep_ckpt={args.keep_checkpoints}")

    todo = [args.combo] if args.combo else list(COMBOS)
    done = set()

    while True:
        progressed = False
        for combo in list(todo):
            if combo in done:
                continue
            slug = COMBOS[combo][0]
            bp = TUNE_ROOT / slug / "best_params.json"
            res = TUNE_ROOT / slug / "holdout_result.json"
            if res.is_file() and not args.force:
                log(f"[skip  ] combo={combo} hold-out 已完成，跳过")
                done.add(combo)
                progressed = True
                continue
            if not bp.is_file():
                continue
            try:
                validate_one(combo, phys, log, args.keep_checkpoints)
            except BaseException:                              # noqa: BLE001
                log(f"[err   ] combo={combo} 验证失败\n{traceback.format_exc()}")
                done.add(combo)      # 不让一个组合的失败卡死整个 worker
                progressed = True
                continue
            done.add(combo)
            progressed = True
            p = write_summary_holdout()
            if p:
                log(f"[worker] wrote {p}")

        remaining = [c for c in todo if c not in done]
        if not remaining:
            log("[worker] 所有组合的 hold-out 验证已完成，退出")
            break
        if args.once:
            log(f"[worker] --once：本轮结束，未就绪的组合 {remaining}")
            break
        if not progressed:
            log(f"[wait  ] 等待搜索完成… 未就绪 {remaining}")
        time.sleep(args.poll)

    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
