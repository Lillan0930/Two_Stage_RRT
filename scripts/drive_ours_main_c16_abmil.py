#!/usr/bin/env python
"""Ours 主实验（C16 + ABMIL）140 runs 的动态队列驱动。

规格 §10：只用 GPU 1/2/4/6/7，最多 5 个训练进程，每张卡同时只跑 1 个任务；
任务不绑定显卡，哪张卡空出来就领下一个未完成任务。

规格 §11：断点续跑 —— ``best_model.pt`` + ``metrics.json`` + ``status ==
'completed'`` 三者齐备才跳过；否则重跑该 seed。单个 run 失败只记进它自己的
``train.log`` / ``status.json``，队列继续。

用法::

    python scripts/drive_ours_main_c16_abmil.py                 # 正式 140 runs
    python scripts/drive_ours_main_c16_abmil.py --dry-run       # 只打印队列
    python scripts/drive_ours_main_c16_abmil.py --only-combo HE+PR --only-seed 42 \\
        --epochs 2 --out-root .../_smoke --tag smoke            # 冒烟
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np                                            # noqa: E402

import ours_main_c16_abmil as O                               # noqa: E402

RUNNER = REPO_ROOT / "scripts" / "run_one_ours_seed.py"


# ── 汇总（规格 §8 / §9） ────────────────────────────────────────────────

def load_metrics(sd):
    p = sd / "metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def fmt(v):
    """至少 4 位小数（规格 §8）。"""
    return f"{v:.4f}"


def combo_summary_text(name, seeds):
    """一个组合的 summary.txt 正文（规格 §8）。"""
    rows, missing = [], []
    for s in seeds:
        m = load_metrics(O.seed_dir(name, s))
        st = O.seed_dir(name, s) / "status.json"
        if m is None or not O.is_completed(name, s):
            reason = "no metrics.json"
            if st.is_file():
                try:
                    reason = json.loads(st.read_text()).get("status", "unknown")
                except Exception:                              # noqa: BLE001
                    reason = "unreadable status.json"
            missing.append((s, reason))
        else:
            rows.append(m)

    lines = []
    lines.append(f"Combination: {name}")
    lines.append("Dataset: C16")
    lines.append("MIL: ABMIL")
    if O.is_baseline(name):
        lines.append("Role: HE-only 对照基线（不属于规格 §3 的 14 个组合）。"
                     "与 14 个组合同 harness / 协议 / encoder 配置 / 同一套 10 seeds，"
                     "唯一区别是 modalities=['HE']，无 aux 分支时 H_fused = H。")
    lines.append(f"Seeds: {', '.join(str(s) for s in seeds)}")
    lines.append(f"Completed: {len(rows)}/{len(seeds)}")
    lines.append("")
    if missing:
        lines.append("未完成 / 失败的 seed:")
        for s, why in missing:
            lines.append(f"  seed{s}: {why}")
        lines.append("")

    if not rows:
        lines.append("INCOMPLETE — 没有任何完成的 run，无法给出 mean ± std")
        lines.append("")
        lines.append("指标口径: Recall=sensitivity_class_1(肿瘤召回), "
                     "Precision=precision_class_1(肿瘤精确率), "
                     "F1=肿瘤类 F1, Specificity=sensitivity_class_0(正常类召回), "
                     "AUC=roc_auc(P(tumor))")
        return "\n".join(lines), None

    keys = O.PRIMARY_METRICS
    hdr = f"{'Seed':<10}" + "".join(f"{k:>13}" for k in keys)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in rows:
        lines.append(f"{m['seed']:<10}" + "".join(f"{fmt(m[k]):>13}" for k in keys))
    lines.append("-" * len(hdr))

    lines.append("")
    lines.append("Mean ± Std")
    stats = {}
    for k in keys:
        vals = np.array([m[k] for m in rows], dtype=float)
        mu = float(vals.mean())
        sd_ = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        stats[k] = (mu, sd_)
        lines.append(f"{k:<12}: {mu:.4f} ± {sd_:.4f}")

    best = max(rows, key=lambda m: m["AUC"])
    worst = min(rows, key=lambda m: m["AUC"])
    lines.append("")
    lines.append(f"Best seed: {best['seed']}")
    lines.append(f"Best AUC: {best['AUC']:.4f}")
    lines.append(f"Worst seed: {worst['seed']}")
    lines.append(f"Worst AUC: {worst['AUC']:.4f}")
    lines.append(f"Mean best epoch: {np.mean([m['best_epoch'] for m in rows]):.1f}")
    lines.append("")

    if len(rows) < len(seeds):
        lines.append(f"INCOMPLETE — 只有 {len(rows)}/{len(seeds)} 完成，"
                     f"上面的 mean ± std 不是正式结果")
        lines.append("")

    lines.append("统计口径: 标准差用 np.std(values, ddof=1)；主结果一律 mean ± std，"
                 "不以 best seed 作为主结果。")
    lines.append("指标口径: Recall=sensitivity_class_1(肿瘤召回), "
                 "Precision=precision_class_1(肿瘤精确率), "
                 "F1=肿瘤类 F1, Specificity=sensitivity_class_0(正常类召回), "
                 "AUC=roc_auc(P(tumor)), Accuracy=accuracy_score")
    return "\n".join(lines), (stats if len(rows) == len(seeds) else None)


def collect(name, seeds):
    """某个组合下所有**完成**的 run 的 metrics.json。"""
    return [m for m, s in ((load_metrics(O.seed_dir(name, s)), s) for s in seeds)
            if m is not None and O.is_completed(name, s)]


def stat_cells(rows):
    cells = []
    for k in O.PRIMARY_METRICS:
        if not rows:
            cells.append("—".rjust(26))
            continue
        vals = np.array([r[k] for r in rows], dtype=float)
        mu = float(vals.mean())
        # n=1 时 ddof=1 的 std 是 nan —— 单条记录没有 std 可言，写 0
        sd_ = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cells.append(f"{mu:.4f}±{sd_:.4f}".rjust(26))
    return cells


def write_all_summaries(seeds):
    O.OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, _ in O.ALL_COMBINATIONS:
        text, _stats = combo_summary_text(name, seeds)
        d = O.combo_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.txt").write_text(text + "\n")

    # ── 全局汇总（规格 §9 + HE-only 基线） ──
    lines, csv_rows = [], []
    n_combos = len(O.COMBINATIONS)
    lines.append("=" * 100)
    lines.append(f"Ours 主实验 — C16 + ABMIL — {n_combos} 个组合 × 10 seeds"
                 f" + HE-only 基线（同 10 seeds）")
    lines.append("=" * 100)
    lines.append(f"结果根目录 : {O.OUT_ROOT}")
    lines.append(f"方法       : HEAuxUnifiedModel (stage2=he_aux_unified, v3-style fusion) + ABMIL, CE only")
    lines.append(f"协议       : Train / Val-as-Test (270 / 129)")
    lines.append(f"Seeds      : {', '.join(str(s) for s in seeds)}")
    lines.append(f"生成时间   : {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    hdr = (f"{'Combination':<24}{'Done':>6}" +
           "".join(f"{k + ' (mean±std)':>26}" for k in O.PRIMARY_METRICS))
    lines.append(hdr)
    lines.append("-" * len(hdr))

    per_combo_rows = {}
    for name, _ in O.ALL_COMBINATIONS:
        rows = collect(name, seeds)
        per_combo_rows[name] = rows
        if O.is_baseline(name):
            lines.append("-" * len(hdr))           # 基线与 14 个组合分栏
        label = f"[baseline] {name}" if O.is_baseline(name) else name
        lines.append(f"{label:<24}{f'{len(rows)}/10':>6}" + "".join(stat_cells(rows)))

        csv_row = {"combination": name,
                   "role": "baseline" if O.is_baseline(name) else "combination",
                   "completed": len(rows), "seeds_total": len(seeds)}
        for k in O.PRIMARY_METRICS:
            if rows:
                vals = np.array([r[k] for r in rows], dtype=float)
                csv_row[f"{k}_mean"] = float(vals.mean())
                csv_row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                csv_row[f"{k}_mean"] = ""
                csv_row[f"{k}_std"] = ""
        csv_rows.append(csv_row)

    n_complete = sum(1 for r in csv_rows
                     if r["role"] == "combination" and r["completed"] == len(seeds))
    base_ok = all(r["completed"] == len(seeds) for r in csv_rows if r["role"] == "baseline")
    lines.append("-" * len(hdr))
    lines.append(f"完整组合 (10/10): {n_complete}/{n_combos}"
                 f"     HE-only 基线 (10/10): {'yes' if base_ok else 'NO'}")
    if n_complete < n_combos or not base_ok:
        lines.append("INCOMPLETE — 存在未满 10 seeds 的行，其 mean ± std 不是正式结果")

    # ── 配对 ΔAUC：同一 seed 相减（组合 − HE-only） ──
    # 10 个 seed 是共用的，配对差比"均值减均值"灵敏得多：seed 间的方差被消掉，
    # 剩下的才是"加辅助染色"本身的效应。
    he_by_seed = {m["seed"]: m["AUC"] for m in per_combo_rows.get("HE", [])}
    if he_by_seed:
        lines.append("")
        lines.append("── 配对 ΔAUC（同一 seed：组合 − HE-only；ddof=1） ──")
        lines.append(f"{'Combination':<24}{'n':>4}{'ΔAUC (mean±std)':>22}"
                     f"{'组合 mean':>12}{'HE mean':>12}   逐 seed Δ")
        lines.append("-" * 108)
        for name, _ in O.COMBINATIONS:
            # (seed, combo_auc, he_auc) —— 只有两边都完成的 seed 才配对
            pair = [(m["seed"], m["AUC"], he_by_seed[m["seed"]])
                    for m in per_combo_rows[name] if m["seed"] in he_by_seed]
            if not pair:
                lines.append(f"{name:<24}{0:>4}{'—':>22}")
                continue
            d = np.array([c - h for _, c, h in pair], dtype=float)
            sd_ = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            per = " ".join(f"{v:+.4f}" for _, v in zip(pair, d))
            lines.append(f"{name:<24}{len(d):>4}"
                         f"{f'{d.mean():+.4f}±{sd_:.4f}':>22}"
                         f"{np.mean([c for _, c, _ in pair]):>12.4f}"
                         f"{np.mean([h for _, _, h in pair]):>12.4f}   {per}")
        lines.append("")
        lines.append("判读：ΔAUC 的 mean 与 std 同一量级 ⇒ 该组合相对 HE-only 的差别在 seed 噪声内。")

    lines.append("")
    lines.append("标准差 ddof=1；主结果一律 mean ± std，不以 best seed 作为主结果。")
    lines.append("HE-only 是后加的对照基线（不属于规格 §3 的 14 个组合），"
                 "用同一 harness / 协议 / encoder 配置 / 同一套 10 seeds 重跑。")
    lines.append("Recall=肿瘤召回(sensitivity_class_1), Precision=肿瘤精确率, "
                 "F1=肿瘤类 F1, Specificity=正常类召回(sensitivity_class_0)")
    lines.append("=" * 100)

    (O.OUT_ROOT / "all_results_summary.txt").write_text("\n".join(lines) + "\n")

    cols = ["combination", "role", "completed", "seeds_total"]
    for k in O.PRIMARY_METRICS:
        cols += [f"{k}_mean", f"{k}_std"]
    with open(O.OUT_ROOT / "all_results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)


def write_params(name, modalities, seeds):
    """规格 §6：每个组合一份完整 resolved config（同组合只有 seed 不同）。"""
    cfg = O.build_config(name, modalities, seeds[0])
    # 去掉 seed 专属的三个字段，替换为"本组合的 seed 列表"
    cfg["output"] = {"save_dir": "<组合目录>/seed<seed>/", "log_dir": ".../logs/",
                     "img_dir": ".../img/"}
    cfg["experiment"].pop("seed", None)
    cfg["experiment"]["seeds"] = list(seeds)
    cfg["experiment"]["seeds_file"] = str(O.OUT_ROOT / "seeds.json")
    cfg["experiment"]["seeds_per_combination"] = len(seeds)
    cfg["experiment"]["epochs_source"] = "training.num_epochs (early stopping patience 10 on val_auc)"
    cfg["experiment"]["train_samples"] = 270
    cfg["experiment"]["val_samples"] = 129
    cfg["experiment"]["gpu_pool"] = O.GPU_POOL
    cfg["experiment"]["note_encoder_cfg"] = (
        "五个染色统一使用同一份 Stage-1 RRT 配置（region 4 / epeg_k 9 / crmsa_k 3 / "
        "n_heads 4 / drop_path 0.0），以保证 14 个组合横向可比；历史 v3 里 PR 曾单独"
        "调过 region 8 / epeg_k 15 / crmsa_k 5 / n_heads 8 / drop_path 0.1155，本轮弃用。")
    d = O.combo_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(json.dumps(cfg, indent=2) + "\n")


# ── 队列 ────────────────────────────────────────────────────────────────

def build_tasks(seeds, only_combo=None, only_seed=None):
    """按 seed 外层、组合内层排序 —— 前 14 个任务就让每个组合都有 1 个 seed，
    这样任何时刻中断都能看到一张完整的单 seed 表。"""
    tasks = []
    for s in seeds:
        if only_seed is not None and s != only_seed:
            continue
        for name, mods in O.ALL_COMBINATIONS:
            if only_combo and name != only_combo:
                continue
            tasks.append((name, mods, s))
    return tasks


# ── GPU 占用门（规格 §10：每张卡同时只跑 1 个任务） ──────────────────────
# 这台机器是共用的：别的 session / 用户会在同一批卡上跑任务。启动前查一次
# 该卡上有没有 **compute 进程**，有就跳过等下一轮。
#
# 不能用"显存占用 > 阈值"来判断：实测别人的训练进程可能只占 ~600 MiB，
# 阈值法会把它当成空卡，结果两个训练挤在同一张卡上。
def busy_gpus():
    """当前有 compute 进程占用的物理 GPU index 集合。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        uuid2idx = {}
        for line in out.stdout.strip().splitlines():
            idx, uuid = [x.strip() for x in line.split(",")]
            uuid2idx[uuid] = int(idx)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return {uuid2idx[u] for u in (l.strip() for l in out.stdout.splitlines())
                if u in uuid2idx}
    except Exception:                                          # noqa: BLE001
        return set()      # 查不到就不阻塞，交给 CUDA 自己报错


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default=",".join(str(g) for g in O.GPU_POOL))
    ap.add_argument("--only-combo", default=None)
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="覆写 num_epochs（仅冒烟用）")
    ap.add_argument("--out-root", default=None, help="覆写结果根目录（仅冒烟用）")
    ap.add_argument("--tag", default="run", help="驱动日志前缀")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="把 status=failed 的 run 也重新排队（默认 failed 同样会重跑，"
                         "因为 is_completed 不成立；此开关只影响日志措辞）")
    args = ap.parse_args()

    if args.out_root:
        O.OUT_ROOT = Path(args.out_root)
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    O.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seeds, seed_meta, created = O.load_or_create_seeds()
    print(f"[seeds] {'新建' if created else '复用'} {O.OUT_ROOT / 'seeds.json'}: {seeds}")

    tasks = build_tasks(seeds, args.only_combo, args.only_seed)
    pending = [t for t in tasks if not O.is_completed(t[0], t[2])]
    done = len(tasks) - len(pending)
    print(f"[queue] 总任务 {len(tasks)}，已完成 {done}，待跑 {len(pending)}")

    if args.dry_run:
        for name, mods, s in pending:
            print(f"  PENDING {name:<24} seed{s}  modalities={mods}")
        return 0

    log_path = O.OUT_ROOT / f"driver_{args.tag}_{datetime.now().strftime('%m%d%H%M')}.log"
    log_f = open(log_path, "a")

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== Ours main C16+ABMIL driver start ===")
    log(f"out_root={O.OUT_ROOT} gpus={gpus} pending={len(pending)} already_done={done}")
    _busy = busy_gpus()
    log(f"busy_gpus={sorted(_busy)}  -> 本 driver 只会在空闲卡上启动，每卡 1 个任务")

    for name, mods in O.ALL_COMBINATIONS:
        write_params(name, mods, seeds)
    if created:
        (O.OUT_ROOT / "data_audit_note.txt").write_text(
            "seeds.json 于本轮首次生成；14 个组合共用同一套 10 个 seed。\n")

    failures, slots = [], {}          # gpu -> dict(task, proc, t0)
    t_start = time.time()
    idx = 0

    def launch(task, gpu):
        name, mods, s = task
        sd = O.seed_dir(name, s)
        sd.mkdir(parents=True, exist_ok=True)
        logf = open(sd / "train.log", "a")
        logf.write(f"\n===== run start {datetime.now().isoformat(timespec='seconds')} "
                   f"combo={name} seed={s} physical_gpu={gpu} =====\n")
        logf.flush()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)      # 物理卡 → 进程内 cuda:0
        env["OMP_NUM_THREADS"] = "4"
        cmd = [sys.executable, str(RUNNER), "--combo", name, "--seed", str(s),
               "--gpu", str(gpu), "--out-root", str(O.OUT_ROOT)]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT))
        return {"task": task, "proc": proc, "t0": time.time(), "logf": logf, "gpu": gpu}

    def fill():
        nonlocal idx
        busy = busy_gpus()
        for gpu in gpus:
            if gpu in slots or idx >= len(pending):
                continue
            if gpu in busy:
                continue          # 别的 session 正占着这张卡，下一轮再看
            task = pending[idx]
            idx += 1
            slots[gpu] = launch(task, gpu)
            name, _, s = task
            log(f"[launch] gpu{gpu} <- {name} seed{s}  "
                f"({idx}/{len(pending)}, pid={slots[gpu]['proc'].pid})")

    fill()
    while slots:
        time.sleep(10)
        for gpu in list(slots):
            slot = slots[gpu]
            rc = slot["proc"].poll()
            if rc is None:
                continue
            name, mods, s = slot["task"]
            dt = time.time() - slot["t0"]
            slot["logf"].close()
            ok = (rc == 0) and O.is_completed(name, s)
            if ok:
                m = load_metrics(O.seed_dir(name, s))
                log(f"[done  ] gpu{gpu} {name} seed{s} rc={rc} {dt / 60:.1f}min "
                    f"AUC={m['AUC']:.4f} best_epoch={m['best_epoch']} "
                    f"epochs={m['epochs_run']}")
            else:
                failures.append({"combo": name, "seed": s, "gpu": gpu, "rc": rc,
                                 "status": _read_status(O.seed_dir(name, s)),
                                 "minutes": round(dt / 60, 1)})
                log(f"[FAILED] gpu{gpu} {name} seed{s} rc={rc} {dt / 60:.1f}min "
                    f"status={_read_status(O.seed_dir(name, s))}")
            del slots[gpu]
        fill()

    elapsed = (time.time() - t_start) / 60
    n_ok = sum(1 for name, _, s in tasks if O.is_completed(name, s))
    log(f"=== queue drained: {n_ok}/{len(tasks)} completed, "
        f"{len(failures)} failed, {elapsed:.1f} min wall ===")
    for f_ in failures:
        log(f"    FAILED {f_['combo']} seed{f_['seed']} rc={f_['rc']} status={f_['status']}")

    write_all_summaries(seeds)
    log(f"[summary] wrote {O.OUT_ROOT / 'all_results_summary.txt'} and .csv")
    log_f.close()

    print(f"\n完成 {n_ok}/{len(tasks)}，失败 {len(failures)}，耗时 {elapsed:.1f} 分钟")
    if failures:
        print("失败任务：")
        for f_ in failures:
            print(f"  {f_['combo']} seed{f_['seed']} gpu{f_['gpu']} rc={f_['rc']} status={f_['status']}")
    return 0 if not failures else 2


def _read_status(sd):
    p = sd / "status.json"
    if not p.is_file():
        return "no status.json"
    try:
        d = json.loads(p.read_text())
        return d.get("status", "?") + (f" ({d['error']})" if d.get("error") else "")
    except Exception:                                          # noqa: BLE001
        return "unreadable"


if __name__ == "__main__":
    sys.exit(main())
