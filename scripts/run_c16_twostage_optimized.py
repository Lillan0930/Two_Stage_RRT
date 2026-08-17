#!/usr/bin/env python3
"""
Two-stage (HE+PR) optimized 5-seed official Test.

用 Stage2 网格的最优配置（crmsa_k=8 / drop_path=0.0）+ 固定 encoder
（HE 4/9/3, PR 8/15/5）+ 差分 lr（Stage1 1e-5 / Stage2 1e-4），
跑 5 seeds 官方 Test。协议与 c16_pr_optimized / c16_he_baseline 完全一致：
    train = data/C16_labels/fixed_split/train.csv (216)
    val   = data/C16_labels/fixed_split/val.csv   (54)
    test  = data/C16_labels/c16_test_labels.csv   (129)
    max_patches=2500, sampling=random, sample_seed=42

Usage (from TwoStageRRT/):
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_twostage_optimized.py

每 seed 一个 `scripts/_run_fixed_split_exp.py` 进程，经 EXP_GPU 分配到不同 GPU。
"""
import os, sys, json, subprocess
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_fixed_split_exp.py"
BASE_OUT = PROJECT / "results" / "c16_twostage_optimized"

SEEDS = [42, 123, 456, 789, 1024]
# 5 seeds → 2 个空闲 GPU（6/7），3+2 分配（每模型 ~1.2GB，16GB 卡足够并发）
GPUS = [6, 7, 6, 7, 6]

TRAIN_LABEL = str(PROJECT / "data/C16_labels/fixed_split/train.csv")
VAL_LABEL = str(PROJECT / "data/C16_labels/fixed_split/val.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"


def build_config(seed, out_dir):
    return {
        "data": {
            "dataset_type": "c16",
            "train_label_file": TRAIN_LABEL,
            "val_label_file": VAL_LABEL,
            "feature_base_dir": FEATURE_BASE,
            "modalities": ["HE", "PR"],
            "dir_mapping": {"HE": "C16_HE_features", "PR": "C16_PR_features"},
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": 42,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            # 共享值（two-stage 路径下被 encoder_cfg/stage2_cfg 覆盖；仅影响未用 rrt_encoder）
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 8, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            # 固定 Stage1 encoder（不要动）
            "encoder_cfg": {
                "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3},   # HE baseline
                "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5},  # PR tuned
            },
            # Stage2 CR-MSA 最优（网格 round 1 最佳，round 2 印证 k=8 为峰值）
            "stage2_cfg": {
                "crmsa_k": 8, "drop_path": 0.0, "region_num": 4,
                "crmsa_heads": 8, "epeg": True, "epeg_k": 9,
                "crmsa_mlp": False, "drop_out": 0.1,
            },
        },
        "training": {
            "batch_size": 1, "num_epochs": 25,
            "learning_rate": 1e-4,          # 名义 base；实际被 lr_stage1/lr_stage2 覆盖
            "lr_stage1": 1e-5, "lr_stage2": 1e-4,
            "weight_decay": 1e-5, "scheduler": {"type": "plateau"},
            "early_stopping": {"patience": 10, "monitor": "val_auc", "mode": "max"},
            "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
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

    env = dict(os.environ)
    env["EXP_GPU"] = str(gpu)
    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT), env=env,
    )
    return proc, log_f, out_dir


def main():
    print("=" * 72)
    print("Two-stage (HE+PR) optimized 5-seed official Test")
    print(f"Seeds: {SEEDS}")
    print(f"GPUs:  {dict(zip(SEEDS, GPUS))}")
    print("=" * 72)

    jobs = []
    for seed, gpu in zip(SEEDS, GPUS):
        proc, log_f, out_dir = launch(seed, gpu)
        jobs.append((seed, gpu, proc, log_f, out_dir))
        print(f"  seed={seed:>4}  GPU={gpu}  → {out_dir}")

    print(f"\n▶ {len(jobs)} jobs launched in parallel. Waiting...\n")

    for seed, gpu, proc, log_f, out_dir in jobs:
        ret = proc.wait()
        log_f.close()
        res_file = out_dir / "result.json"
        if res_file.exists():
            r = json.load(open(res_file))
            info = f"Val={r['best_val_auc']:.4f} Test={r['test_auc']:.4f}"
        else:
            info = f"NO RESULT (rc={ret})"
        print(f"  [{'✓' if ret == 0 else '✗'}] seed={seed:>4}  {info}")

    # ── Summary ──
    test_aucs, val_aucs = [], []
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for seed in SEEDS:
        res_file = BASE_OUT / f"seed{seed}" / "result.json"
        if not res_file.exists():
            continue
        r = json.load(open(res_file))
        test_aucs.append(r["test_auc"])
        val_aucs.append(r["best_val_auc"])
        print(f"  seed={seed:>4}: Val={r['best_val_auc']:.4f}  "
              f"Test AUC={r['test_auc']:.4f}  Acc={r['test_acc']:.4f}  "
              f"F1={r['test_f1']:.4f}")

    if test_aucs:
        print(f"  ─────────────────────────────────────────")
        print(f"  Val AUC : {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        print(f"  Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
