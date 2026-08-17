#!/usr/bin/env python3
"""诚实 plain-ABMIL 天花板驱动（Base_mil AttentionMIL，270/129 协议，5-seed）。

两个 recipe：
  abmil_recipe — Base_mil 原样（focal γ2.4 / lr 1.88e-4 / wd 3.8e-6 / dropout 0.101 / hidden 384）
  abmil_plain  — 与 R²T 基线同 recipe（CE / lr 1e-4 / wd 1e-5 / dropout 0.25 / hidden 384）

每 seed 复用 scripts/_run_abmil_honest.py。
用法: python scripts/run_abmil_honest.py --configs abmil_recipe,abmil_plain --seeds 42,123,456,789,1024 --gpus 6,7,6,7,6,7,6,7,6,7
"""
import os, sys, json, subprocess, argparse
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_abmil_honest.py"
BASE_OUT = PROJECT / "results" / "c16_official_train_test"
TRAIN_LABEL = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"

CONFIGS = {
    "abmil_recipe": {
        "model": {"hidden_dim": 384, "dropout_rate": 0.101},
        "training": {"focal_loss": True, "focal_gamma": 2.403,
                     "learning_rate": 1.8797e-4, "weight_decay": 3.829e-6},
    },
    "abmil_plain": {
        "model": {"hidden_dim": 384, "dropout_rate": 0.25},
        "training": {"focal_loss": False, "focal_gamma": 2.0,
                     "learning_rate": 1e-4, "weight_decay": 1e-5},
    },
}


def build_config(name, seed, out_dir):
    o = CONFIGS[name]
    return {
        "data": {
            "dataset_type": "c16", "modalities": ["PR"],
            "dir_mapping": {"PR": "C16_PR_features"},
            "train_label_file": TRAIN_LABEL,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": 42,
        },
        "model": {"hidden_dim": o["model"]["hidden_dim"],
                  "dropout_rate": o["model"]["dropout_rate"]},
        "training": {"batch_size": 1, "num_epochs": 25,
                     "learning_rate": o["training"]["learning_rate"],
                     "weight_decay": o["training"]["weight_decay"],
                     "scheduler": {"type": "cosine"},
                     "focal_loss": o["training"]["focal_loss"],
                     "focal_gamma": o["training"]["focal_gamma"]},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": str(out_dir / "ckpt"),
                   "log_dir": str(out_dir / "logs"),
                   "img_dir": str(out_dir / "img")},
    }


def launch(name, seed, gpu):
    out_dir = BASE_OUT / name / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(build_config(name, seed, out_dir), f, indent=2)
    env = dict(os.environ); env["EXP_GPU"] = str(gpu)
    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen([PYTHON, str(RUNNER), str(out_dir)],
                            stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT), env=env)
    return proc, log_f, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', type=str, default='abmil_recipe,abmil_plain')
    ap.add_argument('--seeds', type=str, default='42,123,456,789,1024')
    ap.add_argument('--gpus', type=str, default='6')
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) == 1:
        gpus = gpus * (len(configs) * len(seeds))
    assert len(gpus) == len(configs) * len(seeds)

    print("=" * 72)
    print("Honest plain-ABMIL ceiling（270/129, 25ep, cosine, last ckpt）")
    print(f"Configs: {configs}  Seeds: {seeds}")
    print("=" * 72)
    jobs = []
    k = 0
    for cfg_name in configs:
        for seed in seeds:
            proc, log_f, out_dir = launch(cfg_name, seed, gpus[k]); k += 1
            jobs.append((cfg_name, seed, proc, log_f, out_dir))
            print(f"  {cfg_name:>14} seed={seed:>4} GPU={gpus[k-1]}")
    print(f"\n▶ {len(jobs)} jobs. Waiting...\n")
    for cfg_name, seed, proc, log_f, out_dir in jobs:
        ret = proc.wait(); log_f.close()
        mf = out_dir / "metrics.json"
        if mf.exists():
            m = json.load(open(mf))
            print(f"  [{'✓' if ret == 0 else '✗'}] {cfg_name:>14} seed={seed:>4} "
                  f"AUC={m['auc']:.4f} Acc={m['accuracy']:.4f}")
        else:
            print(f"  [✗] {cfg_name:>14} seed={seed:>4} NO RESULT rc={ret}")
    print("\n" + "=" * 72)
    print("SUMMARY（样本标准差 ddof=1）")
    print("=" * 72)
    for cfg_name in configs:
        aucs, accs = [], []
        for seed in seeds:
            mf = BASE_OUT / cfg_name / f"seed{seed}" / "metrics.json"
            if mf.exists():
                m = json.load(open(mf)); aucs.append(m["auc"]); accs.append(m["accuracy"])
        if aucs:
            print(f"  {cfg_name:>14}: AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}  "
                  f"Acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} (n={len(aucs)})")
    print("\nDone!")


if __name__ == "__main__":
    main()
