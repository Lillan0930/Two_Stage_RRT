#!/usr/bin/env python3
"""
C16 官方 Train/Test 协议 5-seed 驱动（270 train / 129 test，无 val）。

协议（与 pure ABMIL baseline 对齐）：
    train = data/C16_labels/c16_train_labels.csv (270)
    test  = data/C16_labels/c16_test_labels.csv  (129)
    no internal validation split；固定 num_epochs=25；cosine 调度；存 last checkpoint。
    max_patches=2500, sampling=random, sample_seed=42（train per-epoch random，test fixed）。

用法（从 TwoStageRRT/ 目录）：
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_official_train_test.py \
        --model he_rrt --seeds 42 --gpus 6

    --model he_rrt | pr_rrt | two_stage | ...（注册于 MODELS）
    --seeds 42,123,456,789,1024
    --gpus  6,7,6,7,6

每 seed 一个 `scripts/_run_official_train_test_exp.py` 进程，经 EXP_GPU 分配到不同 GPU。
结果写 results/c16_official_train_test/<MODEL>/seed<N>/{metrics.json,test_predictions.csv}
"""
import os, sys, json, subprocess
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_official_train_test_exp.py"
BASE_OUT = PROJECT / "results" / "c16_official_train_test"

TRAIN_LABEL = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
TEST_LABEL = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"

SEEDS_DEFAULT = [42, 123, 456, 789, 1024]


def _base_data(seed):
    return {
        "dataset_type": "c16",
        "train_label_file": TRAIN_LABEL,
        "feature_base_dir": FEATURE_BASE,
        "input_dim": 768, "num_classes": 2,
        "max_patches": 2500, "preload": False,
        "sampling": "random", "sample_seed": 42,
        "no_validation": True,
    }


def _base_training():
    return {
        "batch_size": 1, "num_epochs": 25,
        "learning_rate": 1e-4, "weight_decay": 1e-5,
        "scheduler": {"type": "cosine"},
        "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
        "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
        "no_validation": True,
    }


def build_config(model_key, seed, out_dir):
    data = _base_data(seed)
    training = _base_training()
    model = {
        "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
        "region_num": 4, "n_layers": 2, "n_heads": 4,
        "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
        "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
        "crmsa_heads": 8, "crmsa_mlp": False,
        "fusion_type": "two_stage_region", "fusion_stage": "middle",
        "use_gated_fusion": False, "abmil_hidden_dim": 256,
        "use_mclc": False, "aggregate_modalities": True,
    }

    if model_key == "he_rrt":
        # HE-only R²T+ABMIL（region 4/9/3，与 fixed_split baseline 同参）
        data["modalities"] = ["HE"]
        data["dir_mapping"] = {"HE": "C16_HE_features"}

    elif model_key == "pr_rrt":
        # PR-only R²T+ABMIL（region 8/15/5，PR 调参结果）
        data["modalities"] = ["PR"]
        data["dir_mapping"] = {"PR": "C16_PR_features"}
        model["region_num"] = 8
        model["epeg_k"] = 15
        model["crmsa_k"] = 5

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    return {
        "data": data,
        "model": model,
        "training": training,
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }


def launch(model_key, seed, gpu):
    out_dir = BASE_OUT / model_key / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(build_config(model_key, seed, out_dir), f, indent=2)

    env = dict(os.environ)
    env["EXP_GPU"] = str(gpu)
    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT), env=env,
    )
    return proc, log_f, out_dir


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='he_rrt')
    ap.add_argument('--seeds', type=str, default='42')
    ap.add_argument('--gpus', type=str, default='6')
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(',')]
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) == 1:
        gpus = gpus * len(seeds)
    assert len(gpus) == len(seeds), "gpus 数量须等于 seeds 数量（或单个 gpu 复用）"

    print("=" * 72)
    print("C16 Official Train/Test (270 train / 129 test, no val)")
    print(f"Model: {args.model}   num_epochs=25  cosine  max_patches=2500")
    print(f"Seeds: {seeds}")
    print(f"GPUs:  {dict(zip(seeds, gpus))}")
    print("=" * 72)

    jobs = []
    for seed, gpu in zip(seeds, gpus):
        proc, log_f, out_dir = launch(args.model, seed, gpu)
        jobs.append((seed, gpu, proc, log_f, out_dir))
        print(f"  seed={seed:>4}  GPU={gpu}  → {out_dir}")

    print(f"\n▶ {len(jobs)} jobs launched. Waiting...\n")
    for seed, gpu, proc, log_f, out_dir in jobs:
        ret = proc.wait()
        log_f.close()
        mf = out_dir / "metrics.json"
        if mf.exists():
            m = json.load(open(mf))
            print(f"  [{'✓' if ret == 0 else '✗'}] seed={seed:>4}  "
                  f"AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
        else:
            print(f"  [✗] seed={seed:>4}  NO RESULT (rc={ret})")

    print("\n" + "=" * 72)
    print(f"SUMMARY — {args.model}")
    print("=" * 72)
    aucs, accs, sens, spec = [], [], [], []
    for seed in seeds:
        mf = BASE_OUT / args.model / f"seed{seed}" / "metrics.json"
        if not mf.exists():
            continue
        m = json.load(open(mf))
        aucs.append(m["auc"]); accs.append(m["accuracy"])
        sens.append(m["sensitivity"]); spec.append(m["specificity"])
        print(f"  seed={seed:>4}: AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
    if aucs:
        print("  " + "-" * 50)
        print(f"  AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Acc  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Sens : {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Spec : {np.mean(spec):.4f} ± {np.std(spec):.4f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
