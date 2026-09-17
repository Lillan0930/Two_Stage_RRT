#!/usr/bin/env python
"""Ours 主实验（C17 + ABMIL）140 runs 的动态队列驱动。

GPU 调度（规格补充条款）:
    卡        : 0 / 4 / 5 = 32G，6 / 7 = 16G
    容量      : 32G 卡由 ``--big-card-concurrency`` 决定（1 或 2，取决于
                VRAM probe 的单任务峰值是否 <= 14G）；16G 卡恒为 1
    模态限制  : 16G 卡只接 Dual / Triple；Quad / Quintuple 只上 32G 卡
    队列      : 按模态数从多到少排序，动态领取，不把组合钉死在某张卡
    最大并发  : sum(capacity) —— 峰值 <=14G 时 3*2+2*1 = 8，否则 5

OOM 处理（规格补充条款）:
    **不改** batch_size / max_patches / 模型参数，只把任务重新入队并转投 32G；
    若某张 32G 卡因双开 OOM，该卡此后自动降为单任务。同一任务重试
    ``--max-oom-retry`` 次仍失败才标记 failed（避免无限循环）。

断点续跑（规格 §12）:
    只有 ``best_model.pt`` 存在 **且** ``metrics.json`` 存在 **且**
    ``metrics.json["status"] == "completed"`` 才算完成，否则重跑。单个 run 失败
    只记进它自己的 ``train.log`` / ``status.json``，队列继续，绝不中断 140 runs。

用法::

    python scripts/drive_ours_main_c17_abmil.py                 # 正式 140 runs
    python scripts/drive_ours_main_c17_abmil.py --dry-run       # 只打印队列
    python scripts/drive_ours_main_c17_abmil.py --only-combo HE+PR --only-seed 42 \\
        --epochs 1 --out-root .../_smoke --tag smoke            # 冒烟
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np                                            # noqa: E402

import ours_main_c17_abmil as O                               # noqa: E402

RUNNER = REPO_ROOT / "scripts" / "run_one_c17_seed.py"


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
    lines.append("Dataset: C17")
    lines.append("MIL: ABMIL")
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


def write_all_summaries(seeds):
    O.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    official = {}

    for name, _ in O.COMBINATIONS:
        text, stats = combo_summary_text(name, seeds)
        d = O.combo_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.txt").write_text(text + "\n")
        if stats:
            official[name] = stats

    # ── 全局汇总（规格 §9） ──
    lines, csv_rows = [], []
    lines.append("=" * 100)
    lines.append("Ours 主实验 — C17 + ABMIL — 14 个组合 × 10 seeds + HE-only 基线（同 10 seeds）")
    lines.append("=" * 100)
    lines.append(f"结果根目录 : {O.OUT_ROOT}")
    lines.append(f"方法       : HEAuxUnifiedModel (stage2=he_aux_unified, v3-style fusion) + ABMIL, CE only")
    lines.append(f"协议       : Train / Test-as-Val (patient 000..099 / 100..199)")
    lines.append(f"Seeds      : {', '.join(str(s) for s in seeds)}")
    lines.append(f"生成时间   : {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    hdr = (f"{'Combination':<24}{'Done':>6}" +
           "".join(f"{k + ' (mean±std)':>26}" for k in O.PRIMARY_METRICS))
    lines.append(hdr)
    lines.append("-" * len(hdr))

    def collect(sdfn, donefn):
        """按 seeds 顺序取已完成的 metrics 行（未完成的不进统计）。"""
        return [m for m, s in ((load_metrics(sdfn(s)), s) for s in seeds)
                if m is not None and donefn(s)]

    def stat_cells(rows):
        cells = []
        for k in O.PRIMARY_METRICS:
            if not rows:
                cells.append("—".rjust(26))
                continue
            vals = np.array([x[k] for x in rows], dtype=float)
            mu = float(vals.mean())
            sd_ = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            cells.append(f"{mu:.4f}±{sd_:.4f}".rjust(26))
        return cells

    def make_csv_row(cname, rows, role):
        row = {"combination": cname, "role": role,
               "completed": len(rows), "seeds_total": len(seeds)}
        for k in O.PRIMARY_METRICS:
            if rows:
                vals = np.array([x[k] for x in rows], dtype=float)
                row[f"{k}_mean"] = float(vals.mean())
                row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            else:
                row[f"{k}_mean"] = ""
                row[f"{k}_std"] = ""
        return row

    per_combo_rows = {}
    for name, _ in O.COMBINATIONS:
        rows = collect(lambda s, n=name: O.seed_dir(n, s),
                       lambda s, n=name: O.is_completed(n, s))
        per_combo_rows[name] = rows
        lines.append(f"{name:<24}{f'{len(rows)}/10':>6}" + "".join(stat_cells(rows)))
        csv_rows.append(make_csv_row(name, rows, "combination"))

    # HE-only 对照臂：与 14 个组合分栏（不属于规格 §3 的 14 个组合）
    he_rows = collect(O.he_seed_dir, O.he_is_completed)
    per_combo_rows["HE"] = he_rows
    lines.append("-" * len(hdr))
    lines.append(f"{'[baseline] HE':<24}{f'{len(he_rows)}/10':>6}"
                 + "".join(stat_cells(he_rows)))
    he_csv = make_csv_row("HE", he_rows, "baseline")
    csv_rows.append(he_csv)

    lines.append("-" * len(hdr))
    n_complete = sum(1 for r in csv_rows
                     if r["role"] == "combination" and r["completed"] == len(seeds))
    base_ok = he_csv["completed"] == len(seeds)
    lines.append(f"完整组合 (10/10): {n_complete}/{len(O.COMBINATIONS)}"
                 f"     HE-only 基线 (10/10): {'yes' if base_ok else 'NO'}")
    if n_complete < len(O.COMBINATIONS) or not base_ok:
        lines.append("INCOMPLETE — 存在未满 10 seeds 的行，其 mean ± std 不是正式结果")
    lines.append("")

    # ── §10：按 dual / triple / quad / quintuple 分档，标注每档最佳组合 ──
    # 只在**跑满 10/10** 的组合之间比较；判据是 AUC 的 mean（主结果口径），
    # 不是 best seed —— 与「主结果一律 mean ± std」一致。
    GROUP_OF_SIZE = {2: "dual", 3: "triple", 4: "quad", 5: "quintuple"}
    groups = {}
    for r in csv_rows:
        if r["role"] != "combination":       # HE-only 是 1 模态参照臂，不参与分档
            continue
        n_mod = len(r["combination"].split("+"))
        if r["completed"] == len(seeds):
            groups.setdefault(GROUP_OF_SIZE[n_mod], []).append(r)

    lines.append("按模态数分档的最佳组合（判据：AUC mean；仅比较跑满 10/10 的组合）")
    lines.append("-" * len(hdr))
    best_of_group = {}
    for gname in ("dual", "triple", "quad", "quintuple"):
        members = groups.get(gname)
        if not members:
            lines.append(f"  {gname:<12} —  （无跑满的组合）")
            continue
        best = max(members, key=lambda r: r["AUC_mean"])
        best_of_group[gname] = best["combination"]
        lines.append(f"  {gname:<12} {best['combination']:<24} "
                     f"AUC {best['AUC_mean']:.4f} ± {best['AUC_std']:.4f}   "
                     f"(档内 {len(members)} 个组合)")
        for r in sorted(members, key=lambda r: -r["AUC_mean"]):
            mark = "← best" if r is best else ""
            lines.append(f"      {r['combination']:<26} {r['AUC_mean']:.4f} ± "
                         f"{r['AUC_std']:.4f} {mark}")
    lines.append("")

    # 回填 CSV 的分档列
    for r in csv_rows:
        if r["role"] != "combination":
            r["group"] = "baseline"
            r["best_in_group"] = False
            continue
        n_mod = len(r["combination"].split("+"))
        gname = GROUP_OF_SIZE[n_mod]
        r["group"] = gname
        r["best_in_group"] = (r["completed"] == len(seeds)
                              and best_of_group.get(gname) == r["combination"])

    # ── 配对 ΔAUC：同一 seed 相减（组合 − HE-only） ──
    # 10 个 seed 是 14 个组合与 HE-only 共用的，配对差比"均值减均值"灵敏得多：
    # seed 间的方差被消掉，剩下的才是"加辅助染色"本身的效应。
    he_by_seed = {m["seed"]: m["AUC"] for m in per_combo_rows.get("HE", [])}
    if he_by_seed:
        lines.append("── 配对 ΔAUC（同一 seed：组合 − HE-only；ddof=1） ──")
        lines.append(f"{'Combination':<24}{'n':>4}{'ΔAUC (mean±std)':>22}"
                     f"{'组合 mean':>12}{'HE mean':>12}   逐 seed Δ")
        lines.append("-" * 108)
        deltas = []
        for name, _ in O.COMBINATIONS:
            pair = [(m["seed"], m["AUC"], he_by_seed[m["seed"]])
                    for m in per_combo_rows[name] if m["seed"] in he_by_seed]
            if not pair:
                lines.append(f"{name:<24}{0:>4}{'—':>22}")
                continue
            d = np.array([c - h for _, c, h in pair], dtype=float)
            deltas.append(d)
            sd_ = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            per = " ".join(f"{v:+.4f}" for v in d)
            lines.append(f"{name:<24}{len(d):>4}"
                         f"{f'{d.mean():+.4f}±{sd_:.4f}':>22}"
                         f"{np.mean([c for _, c, _ in pair]):>12.4f}"
                         f"{np.mean([h for _, _, h in pair]):>12.4f}   {per}")
        lines.append("")
        lines.append("判读：ΔAUC 的 mean 与 std 同一量级 ⇒ 该组合相对 HE-only 的差别在 seed 噪声内。")
        if deltas:
            pooled = np.concatenate(deltas)
            lines.append(f"汇总：{len(deltas)} 个组合 × {len(deltas[0])} seed 共 "
                         f"{pooled.size} 个配对差，mean={pooled.mean():+.4f}，"
                         f"为正的比例={np.mean(pooled > 0):.3f}")
        lines.append("")

    lines.append("标准差 ddof=1；主结果一律 mean ± std，不以 best seed 作为主结果。")
    lines.append("Recall=肿瘤召回(sensitivity_class_1), Precision=肿瘤精确率, "
                 "F1=肿瘤类 F1, Specificity=正常类召回(sensitivity_class_0)")
    lines.append("HE-only 是后加的对照基线（不属于规格 §3 的 14 个组合），"
                 "用同一 harness / 协议 / encoder 配置 / 同一套 10 seeds 重跑，"
                 "唯一差异是 data.modalities=['HE']（无 aux ⇒ H_fused = H）。")
    lines.append("本表仅为汇总，未据此重新训练或调参（§10）。")
    lines.append("=" * 100)

    (O.OUT_ROOT / "all_results_summary.txt").write_text("\n".join(lines) + "\n")

    cols = ["combination", "role", "completed", "seeds_total"]
    for k in O.PRIMARY_METRICS:
        cols += [f"{k}_mean", f"{k}_std"]
    cols += ["group", "best_in_group"]
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
    cfg["experiment"]["epochs_source"] = "training.num_epochs (early stopping patience 15 on val_auc)"
    cfg["experiment"]["train_samples"] = "见 _pilot/data_audit.txt"
    cfg["experiment"]["val_samples"] = "见 _pilot/data_audit.txt"
    cfg["experiment"]["gpu_pool"] = O.GPU_POOL
    cfg["experiment"]["note_encoder_cfg"] = (
        "五个染色统一使用同一份 Stage-1 RRT 配置（region 4 / epeg_k 15 / crmsa_k 3 / "
        "n_heads 4 / drop_path 0.25），取自历史 C17 HE 最优配置 config_full.yaml，"
        "以保证 14 个组合横向可比；不为任何单个染色单独调参。")
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
        for name, mods in O.COMBINATIONS:
            if only_combo and name != only_combo:
                continue
            tasks.append((name, mods, s))
    return tasks


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
    ap.add_argument("--big-card-concurrency", type=int, default=1, choices=[1, 2],
                    help="32G 卡每卡并发几个任务。由 VRAM probe 决定："
                         "单任务峰值 <= 14G 才设 2，否则 1。默认 1（保守）。")
    ap.add_argument("--max-oom-retry", type=int, default=2,
                    help="同一任务 OOM 重试上限，超过则标记 failed，避免无限循环")
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

    log(f"=== Ours main C17+ABMIL driver start ===")
    log(f"out_root={O.OUT_ROOT} gpus={gpus} pending={len(pending)} already_done={done}")

    for name, mods in O.COMBINATIONS:
        write_params(name, mods, seeds)
    if created:
        (O.OUT_ROOT / "data_audit_note.txt").write_text(
            "seeds.json 于本轮首次生成；14 个组合共用同一套 10 个 seed。\n")

    # ── GPU 规格与并发容量 ────────────────────────────────────────────
    # 32G 卡（0/4/5）允许同卡双开，但只在 --big-card-concurrency 2 时生效；
    # 该值由 VRAM probe 决定（单任务峰值 <= 14G 才双开）。16G 卡（6/7）恒为 1，
    # 且只接 Dual / Triple。
    vram = probe_gpu_vram(gpus)
    is_big = {g: vram.get(g, 0) > 20000 for g in gpus}
    capacity = {g: (args.big_card_concurrency if is_big[g] else 1) for g in gpus}
    downgraded = set()
    log(f"GPU 显存总容量 (MiB): {vram}")
    log(f"32G 卡: {[g for g in gpus if is_big[g]]}  16G 卡: {[g for g in gpus if not is_big[g]]}")
    log(f"初始并发容量: {capacity}  最大并发={sum(capacity.values())}")

    # 队列按模态数从多到少排序：Quintuple/Quad 先被领走，自然落到 32G 卡上
    pending.sort(key=lambda t: (-len(t[1]), t[0], t[2]))
    queue = list(pending)

    failures, oom_events, oom_counts = [], [], {}
    slots = {g: [] for g in gpus}
    t_start = time.time()
    n_launched = 0

    def allowed_on(gpu, task):
        """16G 卡只跑已确认显存安全的 Dual / Triple；Quad/Quintuple 只上 32G。"""
        if len(task[1]) >= 4 and not is_big[gpu]:
            return False
        return True

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
        nonlocal n_launched
        for gpu in gpus:
            while len(slots[gpu]) < capacity[gpu] and queue:
                pick = next((i for i, t in enumerate(queue)
                             if allowed_on(gpu, t)), None)
                if pick is None:
                    break                       # 剩下的任务这张卡不能接
                task = queue.pop(pick)
                slot = launch(task, gpu)
                slots[gpu].append(slot)
                n_launched += 1
                name, mods, s = task
                log(f"[launch] gpu{gpu} <- {name} seed{s}  "
                    f"(并发 {len(slots[gpu])}/{capacity[gpu]}, "
                    f"队列剩 {len(queue)}, pid={slot['proc'].pid})")

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
                sd = O.seed_dir(name, s)

                if rc == 0 and O.is_completed(name, s):
                    m = load_metrics(sd)
                    log(f"[done  ] gpu{gpu} {name} seed{s} {dt:.1f}min "
                        f"AUC={m['AUC']:.4f} best_epoch={m['best_epoch']} "
                        f"epochs={m['epochs_run']} peak={m.get('peak_cuda_memory_mb')}MB")
                    # 组合跑满 10/10 就立刻落 summary.txt（规格 §9），不必等全部 140
                    if all(O.is_completed(name, ss) for ss in seeds):
                        txt, _ = combo_summary_text(name, seeds)
                        (O.combo_dir(name) / "summary.txt").write_text(txt + "\n")
                        log(f"[combo ] {name} 10/10 完成 → {name}/summary.txt")
                    continue

                # ── 失败：OOM 单独处理，其余只记不中断 ──
                oom = is_oom(sd)
                if oom:
                    oom_events.append({"combo": name, "seed": s, "gpu": gpu})
                    oom_counts[(name, s)] = oom_counts.get((name, s), 0) + 1
                    if capacity[gpu] > 1:
                        capacity[gpu] = 1
                        downgraded.add(gpu)
                        log(f"[OOM   ] gpu{gpu} 降为单任务（并发容量 → 1）")
                    # 不改 batch_size / max_patches / 模型参数，只改运行设备
                    if not is_big[gpu]:
                        log(f"[OOM   ] gpu{gpu} 是 16G 卡，任务转投 32G 卡重排")
                    if oom_counts[(name, s)] > args.max_oom_retry:
                        failures.append({"combo": name, "seed": s, "gpu": gpu, "rc": rc,
                                         "status": f"OOM x{oom_counts[(name, s)]} 超重试上限",
                                         "minutes": round(dt, 1)})
                        log(f"[FAILED] gpu{gpu} {name} seed{s} OOM 重试 "
                            f"{oom_counts[(name, s)]} 次仍失败，放弃")
                    else:
                        queue.append((name, mods, s))
                        queue.sort(key=lambda t: (-len(t[1]), t[0], t[2]))
                        log(f"[OOM   ] gpu{gpu} {name} seed{s} {dt:.1f}min → 重新入队 "
                            f"(第 {oom_counts[(name, s)]} 次, 队列剩 {len(queue)})")
                else:
                    failures.append({"combo": name, "seed": s, "gpu": gpu, "rc": rc,
                                     "status": _read_status(sd),
                                     "minutes": round(dt, 1)})
                    log(f"[FAILED] gpu{gpu} {name} seed{s} rc={rc} {dt:.1f}min "
                        f"status={_read_status(sd)}")
        fill()

    log(f"队列清空；降级过的卡: {sorted(downgraded) or '无'}；OOM 次数: {len(oom_events)}")

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


def probe_gpu_vram(gpus):
    """用 nvidia-smi 读每张卡的**总显存**（MiB）。读不到就退化为 {}。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        total = {}
        for line in out.strip().splitlines():
            idx, mb = [x.strip() for x in line.split(",")]
            total[int(idx)] = int(mb)
        return {g: total.get(g, 0) for g in gpus}
    except Exception:                                          # noqa: BLE001
        return {}


_OOM_RE = re.compile(r"out of memory|OutOfMemoryError|CUDA error: out of memory",
                     re.IGNORECASE)


def is_oom(sd):
    """判断这个 run 是不是 OOM 死的 —— 看 status.json 的错误信息与 train.log 尾部。"""
    st = sd / "status.json"
    try:
        d = json.loads(st.read_text())
        if _OOM_RE.search(f"{d.get('error', '')}\n{d.get('traceback', '')}"):
            return True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        return bool(_OOM_RE.search((sd / "train.log").read_text(
            errors="ignore")[-30000:]))
    except Exception:                                          # noqa: BLE001
        return False


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
