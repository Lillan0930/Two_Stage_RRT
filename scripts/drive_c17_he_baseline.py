#!/usr/bin/env python
"""C17 **HE-only** baseline（RRT + ABMIL）—— 补齐主实验缺失的单模态对照臂。

主实验的 14 个组合全部是 HE+X，**没有 HE-only**，因此无法回答"辅助染色相对 HE
单独使用时是否有增益"。本脚本用**逐字段同源**的配置补上这一臂：

    配置来源 : 同一个 ``O.build_config("HE", ["HE"], seed)``
    seeds    : 同一个 ``<C17+abmil>/seeds.json``（不重新生成）
    唯一差异 : ``data.modalities = ["HE"]``

模型走 ``HEAuxUnifiedModel`` 的 ``if not self.aux_stains: return H, H, {}``
分支（``models/he_aux_unified.py:520``），**不经过任何 cross branch**，是干净的
HE-only 通路。

产物落在 ``<C17+abmil>/he/seed<seed>/``，与 14 个组合目录并列；schema 与主实验
完全一致（config.yaml / train.log / best_model.pt / metrics.json / history.csv /
status.json）。

    python scripts/drive_c17_he_baseline.py --gpus 0,4,5,6,7 --big-card-concurrency 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ours_main_c17_abmil as O                               # noqa: E402
from drive_ours_main_c17_abmil import (                       # noqa: E402
    RUNNER, _read_status, fmt, is_oom, load_metrics, probe_gpu_vram,
)

#: HE-only baseline 的结果目录（与 14 个组合目录并列）。
#: 用 O.he_combo_dir() 而不是自己拼 OUT_ROOT/"he"，保证与汇总侧同一路径。
HE_ROOT = O.he_combo_dir()

#: HE 单模态，组合名固定
COMBO = "HE"
MODS = ["HE"]

# 路径与完成判据都复用 ours_main_c17_abmil 里的定义 —— 汇总脚本读的是同一份
# 判据，避免"执行器认为完成、汇总认为没完成"这类分歧（best_model.pt 落在
# 组合目录那次的教训）。
he_seed_dir = O.he_seed_dir
he_is_completed = O.he_is_completed


def he_summary_text(seeds):
    """§9 格式的 summary.txt，与 combo_summary_text 同版式。"""
    rows, missing = [], []
    for s in seeds:
        sd = he_seed_dir(s)
        m = load_metrics(sd)
        if m is None or not he_is_completed(s):
            reason = "no metrics.json"
            st = sd / "status.json"
            if st.is_file():
                try:
                    reason = json.loads(st.read_text()).get("status", "unknown")
                except Exception:                              # noqa: BLE001
                    reason = "unreadable status.json"
            missing.append((s, reason))
        else:
            rows.append(m)

    lines = [
        f"Combination: {COMBO}  (HE-only baseline, 单模态对照臂)",
        "Dataset: C17",
        "MIL: ABMIL",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Completed: {len(rows)}/{len(seeds)}",
        "",
    ]
    if missing:
        lines.append("未完成 / 失败的 seed:")
        lines += [f"  seed{s}: {why}" for s, why in missing]
        lines.append("")

    if not rows:
        lines.append("INCOMPLETE — 没有任何完成的 run，无法给出 mean ± std")
        return "\n".join(lines) + "\n", None

    keys = O.PRIMARY_METRICS
    hdr = f"{'Seed':<10}" + "".join(f"{k:>13}" for k in keys)
    lines += [hdr, "-" * len(hdr)]
    for m in rows:
        lines.append(f"{m['seed']:<10}" + "".join(f"{fmt(m[k]):>13}" for k in keys))
    lines.append("-" * len(hdr))

    lines += ["", "Mean ± Std"]
    stats = {}
    for k in keys:
        vals = np.array([m[k] for m in rows], dtype=float)
        mu = float(vals.mean())
        sd_ = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        stats[k] = (mu, sd_)
        lines.append(f"{k:<12}: {mu:.4f} ± {sd_:.4f}")

    best = max(rows, key=lambda m: m["AUC"])
    worst = min(rows, key=lambda m: m["AUC"])
    lines += [
        "",
        f"Best seed: {best['seed']}",
        f"Best AUC: {best['AUC']:.4f}",
        f"Worst seed: {worst['seed']}",
        f"Worst AUC: {worst['AUC']:.4f}",
        f"Mean best epoch: {np.mean([m['best_epoch'] for m in rows]):.1f}",
        "",
    ]
    if len(rows) < len(seeds):
        lines += [f"INCOMPLETE — 只有 {len(rows)}/{len(seeds)} 完成，"
                  f"上面的 mean ± std 不是正式结果", ""]
    lines += [
        "统计口径: 标准差用 np.std(values, ddof=1)；主结果一律 mean ± std，"
        "不以 best seed 作为主结果。",
        "指标口径: Recall=sensitivity_class_1(肿瘤召回), "
        "Precision=precision_class_1(肿瘤精确率), F1=肿瘤类 F1, "
        "Specificity=sensitivity_class_0(正常类召回), AUC=roc_auc(P(tumor)), "
        "Accuracy=accuracy_score",
        "配置口径: 与 14 组合主实验逐字段同源，唯一差异是 data.modalities=[\"HE\"]。",
    ]
    return "\n".join(lines) + "\n", stats


def write_params(seeds):
    cfg = O.build_config(COMBO, MODS, seeds[0])
    # 写出**真实**落盘路径（绝对）。此前这里是占位字符串，而 --out-dir 又只
    # 重定向了执行器自己的产物、没重定向 trainer 的 checkpoint，导致
    # best_model.pt 落到组合目录 HE/ 下 —— 路径写实，避免同类误判。
    cfg["output"] = {"save_dir": str(HE_ROOT / "seed<seed>"),
                     "log_dir": str(HE_ROOT / "seed<seed>" / "logs"),
                     "img_dir": str(HE_ROOT / "seed<seed>" / "img")}
    cfg["experiment"].pop("seed", None)
    cfg["experiment"]["seeds"] = list(seeds)
    cfg["experiment"]["seeds_file"] = str(O.OUT_ROOT / "seeds.json")
    cfg["experiment"]["seeds_per_combination"] = len(seeds)
    cfg["experiment"]["role"] = (
        "HE-only baseline / 单模态对照臂 —— 主实验 14 个组合全部是 HE+X，"
        "缺此臂则无法判断辅助染色相对 HE 单独使用是否有增益")
    cfg["experiment"]["note_encoder_cfg"] = (
        "与 14 组合使用同一份 Stage-1 RRT 配置（region 4 / epeg_k 15 / crmsa_k 3 / "
        "n_heads 4 / drop_path 0.25）；本 baseline 只启用 HE 一个 encoder，"
        "HEAuxUnifiedModel 走 aux_stains 为空的分支，不经过任何 cross branch。")
    cfg["experiment"]["gpu_pool"] = O.GPU_POOL
    HE_ROOT.mkdir(parents=True, exist_ok=True)
    (HE_ROOT / "params.json").write_text(json.dumps(cfg, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=",".join(str(g) for g in O.GPU_POOL))
    ap.add_argument("--big-card-concurrency", type=int, default=1, choices=[1, 2])
    ap.add_argument("--max-oom-retry", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    HE_ROOT.mkdir(parents=True, exist_ok=True)

    seeds, seed_meta, created = O.load_or_create_seeds()
    print(f"[seeds] {'新建' if created else '复用'} {O.OUT_ROOT / 'seeds.json'}: {seeds}")

    write_params(seeds)

    tasks = [(COMBO, MODS, s) for s in seeds]
    pending = [t for t in tasks if not he_is_completed(t[2])]
    done = len(tasks) - len(pending)
    print(f"[queue] HE-only baseline 总任务 {len(tasks)}，已完成 {done}，待跑 {len(pending)}")
    if args.dry_run:
        for _, _, s in pending:
            print(f"  PENDING HE seed{s}  -> {he_seed_dir(s)}")
        return

    log_path = HE_ROOT / f"driver_{datetime.now().strftime('%m%d%H%M')}.log"
    log_f = open(log_path, "a")

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log("=== C17 HE-only baseline driver start ===")
    log(f"out_root={HE_ROOT} gpus={gpus} pending={len(pending)} already_done={done}")

    vram = probe_gpu_vram(gpus)
    is_big = {g: vram.get(g, 0) > 20000 for g in gpus}
    capacity = {g: (args.big_card_concurrency if is_big[g] else 1) for g in gpus}
    downgraded, oom_events, oom_counts = set(), [], {}
    log(f"GPU 显存总容量 (MiB): {vram}")
    log(f"初始并发容量: {capacity}  最大并发={sum(capacity.values())}")

    queue = list(pending)
    failures, slots = [], {g: [] for g in gpus}
    t_start = time.time()

    def launch(task, gpu):
        name, mods, s = task
        sd = he_seed_dir(s)
        sd.mkdir(parents=True, exist_ok=True)
        logf = open(sd / "train.log", "a")
        logf.write(f"\n===== run start {datetime.now().isoformat(timespec='seconds')} "
                   f"combo={name}(HE-only baseline) seed={s} physical_gpu={gpu} =====\n")
        logf.flush()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "4"
        cmd = [sys.executable, str(RUNNER), "--combo", name, "--seed", str(s),
               "--gpu", str(gpu), "--out-dir", str(sd)]
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT))
        return {"task": task, "proc": proc, "t0": time.time(), "logf": logf, "gpu": gpu}

    def fill():
        for gpu in gpus:
            while len(slots[gpu]) < capacity[gpu] and queue:
                task = queue.pop(0)
                slot = launch(task, gpu)
                slots[gpu].append(slot)
                log(f"[launch] gpu{gpu} <- HE seed{task[2]}  "
                    f"(并发 {len(slots[gpu])}/{capacity[gpu]}, 队列剩 {len(queue)}, "
                    f"pid={slot['proc'].pid})")

    fill()
    while any(slots[g] for g in gpus):
        time.sleep(10)
        for gpu in gpus:
            for slot in list(slots[gpu]):
                rc = slot["proc"].poll()
                if rc is None:
                    continue
                name, mods, s = slot["task"]
                dt = (time.time() - slot["t0"]) / 60
                slot["logf"].close()
                slots[gpu].remove(slot)
                sd = he_seed_dir(s)

                if rc == 0 and he_is_completed(s):
                    m = load_metrics(sd)
                    log(f"[done  ] gpu{gpu} HE seed{s} {dt:.1f}min "
                        f"AUC={m['AUC']:.4f} best_epoch={m['best_epoch']} "
                        f"epochs={m['epochs_run']} peak={m.get('peak_cuda_memory_mb')}MB")
                    continue

                if is_oom(sd):
                    oom_events.append({"seed": s, "gpu": gpu})
                    oom_counts[s] = oom_counts.get(s, 0) + 1
                    if capacity[gpu] > 1:
                        capacity[gpu] = 1
                        downgraded.add(gpu)
                        log(f"[OOM   ] gpu{gpu} 降为单任务（并发容量 → 1）")
                    if oom_counts[s] > args.max_oom_retry:
                        failures.append({"seed": s, "gpu": gpu, "rc": rc,
                                         "status": f"OOM x{oom_counts[s]} 超重试上限"})
                        log(f"[FAILED] gpu{gpu} HE seed{s} OOM 超限，放弃")
                    else:
                        queue.append((name, mods, s))
                        log(f"[OOM   ] gpu{gpu} HE seed{s} → 重新入队 "
                            f"(第 {oom_counts[s]} 次, 队列剩 {len(queue)})")
                else:
                    failures.append({"seed": s, "gpu": gpu, "rc": rc,
                                     "status": _read_status(sd)})
                    log(f"[FAILED] gpu{gpu} HE seed{s} rc={rc} {dt:.1f}min "
                        f"status={_read_status(sd)}")
        fill()

    log(f"队列清空；降级过的卡: {sorted(downgraded) or '无'}；OOM 次数: {len(oom_events)}")

    txt, stats = he_summary_text(seeds)
    (HE_ROOT / "summary.txt").write_text(txt)
    log(f"[summary] wrote {HE_ROOT / 'summary.txt'}")

    elapsed = (time.time() - t_start) / 60
    n_ok = sum(1 for _, _, s in tasks if he_is_completed(s))
    log(f"=== queue drained: {n_ok}/{len(tasks)} completed, "
        f"{len(failures)} failed, {elapsed:.1f} min wall ===")
    for f_ in failures:
        log(f"    FAILED HE seed{f_['seed']} rc={f_['rc']} status={f_['status']}")

    print(f"\nHE-only baseline 完成 {n_ok}/{len(tasks)}，失败 {len(failures)}，"
          f"耗时 {elapsed:.1f} 分钟")


if __name__ == "__main__":
    main()
