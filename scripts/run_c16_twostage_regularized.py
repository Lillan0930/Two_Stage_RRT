#!/usr/bin/env python3
"""
Two-stage (HE+PR) 正则化重试 — 降低泛化误差。

针对上一轮 overfit（val 0.908 / test 0.650）的诊断改造：
  1) 降 Stage2 容量：crmsa_k 8 -> 3（不再搜索）
  2) 增正则：model dropout 0.25 -> 0.5；Stage2 drop_out 0.1 -> 0.2、drop_path 0.0 -> 0.2；
     weight_decay 1e-5 -> 1e-4
  3) Stage2 别学太快：lr_stage1=1e-5, lr_stage2=1e-4 -> 1e-5 / 2e-5
  4) epochs 25 -> 50（early stop patience=10）

协议与基线一致：fixed_split 216/54, max_patches=2500, sampling=random, sample_seed=42。
用法（先跑 seed42 看方向，再扩到 5 seeds）：
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_twostage_regularized.py --seeds 42
    ... --seeds 42,123,456,789,1024 --gpus 6,7,6,7,6
"""
import os, sys, json, argparse, subprocess
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_fixed_split_exp.py"
BASE_OUT = PROJECT / "results" / "c16_twostage_regularized"

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
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.5, "use_gated": False,
            # 共享值（two-stage 下被 encoder_cfg/stage2_cfg 覆盖；仅影响未用 rrt_encoder）
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            # 固定 Stage1 encoder（不要动）
            "encoder_cfg": {
                "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3},
                "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5},
            },
            # Stage2 正则化：容量降 + dropout/droppath 升
            "stage2_cfg": {
                "crmsa_k": 3, "drop_path": 0.2, "region_num": 4,
                "crmsa_heads": 8, "epeg": True, "epeg_k": 9,
                "crmsa_mlp": False, "drop_out": 0.2,
            },
        },
        "training": {
            "batch_size": 1, "num_epochs": 50,
            "learning_rate": 1e-5,
            "lr_stage1": 1e-5, "lr_stage2": 2e-5,
            "weight_decay": 1e-4, "scheduler": {"type": "plateau"},
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--gpus', type=str, default='6')
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) == 1:
        gpus = gpus * len(seeds)
    assert len(gpus) == len(seeds), "gpus 数量须等于 seeds 数量（或单个 gpu 复用）"

    print("=" * 72)
    print("Two-stage regularized retry (降低泛化误差)")
    print(f"crmsa_k=3  dropout=0.5  drop_path=0.2  wd=1e-4  lr=1e-5/2e-5  ep=50")
    print(f"Seeds: {seeds}")
    print(f"GPUs:  {dict(zip(seeds, gpus))}")
    print("=" * 72)

    jobs = []
    for seed, gpu in zip(seeds, gpus):
        proc, log_f, out_dir = launch(seed, gpu)
        jobs.append((seed, gpu, proc, log_f, out_dir))
        print(f"  seed={seed:>4}  GPU={gpu}  → {out_dir}")

    print(f"\n▶ {len(jobs)} jobs launched. Waiting...\n")
    for seed, gpu, proc, log_f, out_dir in jobs:
        ret = proc.wait()
        log_f.close()
        res_file = out_dir / "result.json"
        if res_file.exists():
            r = json.load(open(res_file))
            print(f"  [{'✓' if ret == 0 else '✗'}] seed={seed:>4}  "
                  f"Val={r['best_val_auc']:.4f}  Test={r['test_auc']:.4f}  "
                  f"Sens={r['test_sensitivity']:.4f}  Spec={r['test_specificity']:.4f}")
        else:
            print(f"  [✗] seed={seed:>4}  NO RESULT (rc={ret})")

    # 汇总
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    ta, va, sens, spec = [], [], [], []
    for seed in seeds:
        res_file = BASE_OUT / f"seed{seed}" / "result.json"
        if not res_file.exists():
            continue
        r = json.load(open(res_file))
        ta.append(r["test_auc"]); va.append(r["best_val_auc"])
        sens.append(r["test_sensitivity"]); spec.append(r["test_specificity"])
    if ta:
        print(f"  Val AUC : {np.mean(va):.4f} ± {np.std(va):.4f}")
        print(f"  Test AUC: {np.mean(ta):.4f} ± {np.std(ta):.4f}")
        print(f"  Sens    : {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Spec    : {np.mean(spec):.4f} ± {np.std(spec):.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
