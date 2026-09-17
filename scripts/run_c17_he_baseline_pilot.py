#!/usr/bin/env python
"""Phase A：用**历史 C17 HE baseline 的原始代码**跑 pilot seeds，确认能复现。

本脚本不重写训练循环 —— 它直接 ``exec`` 历史 runner
``work_results/comparative_exp/RRT+abMIL/run_experiment.py``，然后调用它自己的
``run_seed(C17_CONFIG, seed, out_dir, device)``。因此模型
(``RRT_ABMIL``)、数据 (``FeatureDataset``)、优化器 (AdamW + CosineAnnealingLR)、
早停 (patience 15 / min_epochs 10)、checkpoint 选择 (best test AUC) 全部是
历史实现本身，一字未改。

**不修改 RRT、不修改 ABMIL、不因结果不理想自动调参。**

产物::

    _pilot/HE_baseline/
        seed42/{train.log, best_model.pt, metrics.json, history.csv}
        seed123/...
        seed456/...
        summary.txt

用法::

    python scripts/run_c17_he_baseline_pilot.py --seeds 42,123,456 --gpus 6,7
"""
import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HIST_RUNNER = Path("/home/Public/lillan/work_results/comparative_exp/RRT+abMIL/run_experiment.py")
OUT_ROOT = Path("/home/Public/lillan/work_results/ours_main/C17+abmil")
PILOT_DIR = OUT_ROOT / "_pilot" / "HE_baseline"

#: 历史 10-seed 基线（从 comparative_exp 的 summary.txt / all_seeds.csv 读出）
HIST_AUC_MEAN, HIST_AUC_STD = 94.19, 0.37
HIST_SEEDS = {42: 0.9432, 83811: 0.9449, 14593: 0.9450, 3279: 0.9396, 97197: 0.9367,
              36049: 0.9455, 32099: 0.9405, 29257: 0.9360, 18290: 0.9465, 96531: 0.9413}

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s*\|\s*Train Loss=([\d.]+) Acc=([\d.]+) AUC=([\d.]+)\s*\|\s*"
    r"Test Loss=([\d.]+) Acc=([\d.]+) AUC=([\d.]+) F1=([\d.]+) Precision=([\d.]+)")


# ═══════════════════════════════════════════════════════════════════════════
# 子进程：真的跑一个 seed（在干净进程里 import 历史代码，避免包名冲突）
# ═══════════════════════════════════════════════════════════════════════════

def _child(seed, out_dir, device):
    spec = importlib.util.spec_from_file_location("hist_run_experiment", HIST_RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 历史代码原样加载

    cfg = dict(mod.C17_CONFIG)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    res = mod.run_seed(cfg, int(seed), out, device)

    # 历史代码把 log / model 放在 out/{logs,models}；搬到 seed 目录根下
    log_src = out / "logs" / f"seed_{seed}_train.log"
    mdl_src = out / "models" / f"model_seed_{seed}_best.pth"
    text = log_src.read_text() if log_src.is_file() else ""

    (out / "train.log").write_text(text)
    if mdl_src.is_file():
        mdl_src.replace(out / "best_model.pt")

    # 从 log 还原逐 epoch 曲线
    rows = []
    for m in EPOCH_RE.finditer(text):
        g = m.groups()
        rows.append({
            "epoch": int(g[0]),
            "train_loss": float(g[2]), "train_acc": float(g[3]), "train_auc": float(g[4]),
            "val_loss": float(g[5]), "val_accuracy": float(g[6]),
            "val_auc": float(g[7]), "val_f1_macro": float(g[8]),
            "val_precision_macro": float(g[9]),
        })
    with open(out / "history.csv", "w", newline="") as f:
        cols = ["epoch", "train_loss", "train_acc", "train_auc",
                "val_loss", "val_accuracy", "val_auc", "val_f1_macro",
                "val_precision_macro"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {
        "seed": int(seed),
        "source": "historical run_experiment.py + RRT_ABMIL (unmodified)",
        "config": {k: v for k, v in cfg.items()
                   if k not in ("train_patients", "test_patients")},
        "train_patients": f"{cfg['train_patients'][0]}..{cfg['train_patients'][-1]}"
                          f" ({len(cfg['train_patients'])})",
        "test_patients": f"{cfg['test_patients'][0]}..{cfg['test_patients'][-1]}"
                         f" ({len(cfg['test_patients'])})",
        "AUC": res["auc"], "Accuracy": res["acc"], "F1": res["f1"],
        "Precision": res["precision"], "Recall": res["sensitivity"],
        "Specificity": res["specificity"],
        "best_epoch": res["best_epoch"], "epochs_run": res["epochs"],
        "train_time_min": res["train_time_min"],
        "metric_definitions": {
            "AUC": "roc_auc_score(y_true, P(class 1))",
            "Accuracy": "accuracy_score",
            "Recall": "sensitivity = TP/(TP+FN) 正类召回",
            "Precision": "precision_score(zero_division=0) 正类精确率",
            "F1": "f1_score(zero_division=0) 正类 F1（sklearn 二分类默认 pos_label=1）",
            "Specificity": "TN/(TN+FP) 负类召回",
        },
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"seed": seed, "AUC": res["auc"], "done": True}))


# ═══════════════════════════════════════════════════════════════════════════
# 父进程：并行调度
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--gpus", default="6,7")
    ap.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child is not None:
        _child(args.child, args.out_dir, args.device)
        return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    PILOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[phaseA] seeds={seeds} gpus={gpus} out={PILOT_DIR}")
    pending, slots, failures = list(seeds), {}, []
    t0 = time.time()

    while pending or slots:
        for gpu in gpus:
            if gpu in slots or not pending:
                continue
            seed = pending.pop(0)
            sd = PILOT_DIR / f"seed{seed}"
            sd.mkdir(parents=True, exist_ok=True)
            logf = open(sd / "train.log", "a")
            logf.write(f"\n===== phaseA start {datetime.now().isoformat(timespec='seconds')} "
                       f"seed={seed} physical_gpu={gpu} =====\n")
            logf.flush()
            import os
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["OMP_NUM_THREADS"] = "4"
            cmd = [sys.executable, __file__, "--child", str(seed),
                   "--out-dir", str(sd), "--device", "cuda:0"]
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT))
            slots[gpu] = {"seed": seed, "proc": p, "t0": time.time(), "logf": logf}
            print(f"[launch] gpu{gpu} <- seed {seed} (pid={p.pid})", flush=True)

        time.sleep(10)
        for gpu in list(slots):
            s = slots[gpu]
            rc = s["proc"].poll()
            if rc is None:
                continue
            s["logf"].close()
            dt = (time.time() - s["t0"]) / 60
            m = PILOT_DIR / f"seed{s['seed']}" / "metrics.json"
            if rc == 0 and m.is_file():
                d = json.loads(m.read_text())
                print(f"[done  ] gpu{gpu} seed{s['seed']} {dt:.1f}min "
                      f"AUC={d['AUC']:.4f} best_epoch={d['best_epoch']}", flush=True)
            else:
                failures.append((s["seed"], gpu, rc))
                print(f"[FAILED] gpu{gpu} seed{s['seed']} rc={rc} {dt:.1f}min", flush=True)
            del slots[gpu]

    write_summary(seeds)
    print(f"\n[phaseA] 完成 {len(seeds)-len(failures)}/{len(seeds)}，"
          f"耗时 {(time.time()-t0)/60:.1f} min")
    if failures:
        for seed, gpu, rc in failures:
            print(f"  FAILED seed{seed} gpu{gpu} rc={rc}")
    return 0 if not failures else 2


def write_summary(seeds):
    rows = []
    for s in seeds:
        p = PILOT_DIR / f"seed{s}" / "metrics.json"
        if p.is_file():
            rows.append(json.loads(p.read_text()))

    L = []
    A = L.append
    A("=" * 78)
    A("Phase A — C17 HE + RRT + ABMIL 历史 baseline 复现（pilot 3 seeds）")
    A("=" * 78)
    A(f"生成时间        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"代码            : {HIST_RUNNER}（历史 run_experiment.py，未修改）")
    A("                  + /home/Public/lillan/RRT_ABMIL/{models,data,utils}")
    A(f"配置            : comparative_exp/RRT+abMIL/C17/config_full.yaml")
    A(f"seeds           : {seeds}")
    A("")
    A("⚠ 这 3 个 seed 只用于确认配置可复现，不是最终论文的 10-seed 统计。")
    A("")

    if not rows:
        A("没有任何完成的 run。")
        (PILOT_DIR / "summary.txt").write_text("\n".join(L) + "\n")
        return

    hdr = f"{'Seed':<10}{'AUC':>10}{'Accuracy':>10}{'F1':>10}{'Precision':>10}{'Recall':>10}{'Specificity':>12}{'BestEp':>8}{'Epochs':>8}"
    A(hdr)
    A("-" * len(hdr))
    for r in rows:
        A(f"{r['seed']:<10}{r['AUC']:>10.4f}{r['Accuracy']:>10.4f}{r['F1']:>10.4f}"
          f"{r['Precision']:>10.4f}{r['Recall']:>10.4f}{r['Specificity']:>12.4f}"
          f"{r['best_epoch']:>8}{r['epochs_run']:>8}")
    A("-" * len(hdr))
    A("")
    A("Mean ± Std (ddof=1):")
    for k in ["AUC", "Accuracy", "F1", "Precision", "Recall", "Specificity"]:
        v = np.array([r[k] for r in rows], dtype=float)
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        A(f"  {k:<12}: {v.mean():.4f} ± {sd:.4f}")
    A("")
    A("── 与历史 10-seed 基线对照 ──")
    A(f"  历史 10 seeds AUC : {HIST_AUC_MEAN:.2f} ± {HIST_AUC_STD:.2f}  (summary.txt)")
    A(f"  历史逐 seed AUC   : {HIST_SEEDS}")
    A("")
    for r in rows:
        if r["seed"] in HIST_SEEDS:
            d = (r["AUC"] - HIST_SEEDS[r["seed"]]) * 100
            A(f"  seed {r['seed']:<6}: 本次 {r['AUC']:.4f}  历史 {HIST_SEEDS[r['seed']]:.4f}  "
              f"Δ = {d:+.2f} AUC 点")
        else:
            A(f"  seed {r['seed']:<6}: 本次 {r['AUC']:.4f}  （历史 10 seeds 里没有这个 seed）")
    A("")
    mu = float(np.mean([r["AUC"] for r in rows]))
    A(f"  本次 3-seed 均值 {mu:.4f} vs 历史 10-seed 均值 {HIST_AUC_MEAN/100:.4f}  "
      f"→ Δ = {(mu - HIST_AUC_MEAN/100)*100:+.2f} AUC 点")
    A("")
    A("指标口径: AUC=roc_auc(P(class1)); Accuracy=accuracy_score; "
      "Recall=sensitivity=TP/(TP+FN) 正类召回;")
    A("          Precision=precision_score(zero_division=0) 正类; F1=f1_score 正类; "
      "Specificity=TN/(TN+FP) 负类召回。")
    A("          （C17 是二分类任务，以上均为正类/负类口径，不是 macro。）")
    A("=" * 78)
    (PILOT_DIR / "summary.txt").write_text("\n".join(L) + "\n")
    print(f"[phaseA] wrote {PILOT_DIR/'summary.txt'}")


if __name__ == "__main__":
    sys.exit(main())
