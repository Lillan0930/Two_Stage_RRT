#!/usr/bin/env python3
"""
PR+RRT+ABMIL —— 270 train / 129 test-as-val early stopping（对齐 Base_mil 协议）。

train = c16_train_labels.csv (270)
val   = c16_test_labels.csv  (129)   # 官方 test 当 val，early stopping 直接在 test 上选 epoch
early_stopping: monitor val_auc, mode max, patience 5；num_epochs=80；cosine；存 best_model.pt。

recipe = R²T 最优（region 8/15/5, dropout 0.25, 纯 CE, lr 1e-4, wd 1e-5）。
报告 best_val_*（因为 val=test，即 test AUC/Acc/…）。

用法: python scripts/run_pr_rrt_testval.py --seeds 42,123,456,789,1024 --gpus 6,7,6,7,6
"""
import os, sys, json, subprocess, argparse
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_official_train_test_exp.py"
BASE_OUT = PROJECT / "results" / "c16_test_as_val"
TRAIN_LABEL = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
TEST_LABEL = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"


def build_config(seed, out_dir):
    return {
        "data": {
            "dataset_type": "c16",
            "modalities": ["PR"],
            "dir_mapping": {"PR": "C16_PR_features"},
            "train_label_file": TRAIN_LABEL,
            "val_label_file": TEST_LABEL,          # test-as-val
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": 42,
            "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 8, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 15,
            "crmsa_k": 5, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4, "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False, "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 5},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }


def launch(seed, gpu):
    out_dir = BASE_OUT / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(build_config(seed, out_dir), f, indent=2)
    env = dict(os.environ); env["EXP_GPU"] = str(gpu)
    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen([PYTHON, str(RUNNER), str(out_dir)],
                            stdout=log_f, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT), env=env)
    return proc, log_f, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42,123,456,789,1024')
    ap.add_argument('--gpus', type=str, default='6')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',')]
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) == 1:
        gpus = gpus * len(seeds)
    assert len(gpus) == len(seeds)

    print("=" * 72)
    print("PR+RRT+ABMIL — 270 train / 129 test-as-val early stopping")
    print(f"early_stopping: val_auc patience 5 | num_epochs 80 | cosine")
    print(f"Seeds: {seeds}  GPUs: {dict(zip(seeds, gpus))}")
    print("=" * 72)

    jobs = []
    for seed, gpu in zip(seeds, gpus):
        proc, log_f, out_dir = launch(seed, gpu)
        jobs.append((seed, proc, log_f, out_dir))
        print(f"  seed={seed:>4} GPU={gpu} → {out_dir}")

    print(f"\n▶ {len(jobs)} jobs. Waiting...\n")
    for seed, proc, log_f, out_dir in jobs:
        ret = proc.wait(); log_f.close()
        mf = out_dir / "metrics.json"
        if mf.exists():
            m = json.load(open(mf))
            print(f"  [{'✓' if ret == 0 else '✗'}] seed={seed:>4} "
                  f"AUC={m['auc']:.4f} Acc={m['accuracy']:.4f} "
                  f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f} "
                  f"(best_epoch={m['best_epoch']})")
        else:
            print(f"  [✗] seed={seed:>4} NO RESULT rc={ret}")

    print("\n" + "=" * 72)
    print("SUMMARY（样本标准差 ddof=1）")
    print("=" * 72)
    aucs, accs = [], []
    for seed in seeds:
        mf = BASE_OUT / f"seed{seed}" / "metrics.json"
        if mf.exists():
            m = json.load(open(mf)); aucs.append(m["auc"]); accs.append(m["accuracy"])
    if aucs:
        print(f"  AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}  "
              f"Acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f} (n={len(aucs)})")
    print("\nDone!")


if __name__ == "__main__":
    main()
