#!/usr/bin/env python3
"""
Patch Sampling — K Verification with Corrected Sampler
=======================================================
Train: per_epoch random (set_epoch called each epoch)
Val:   fixed deterministic random
Test:  fixed deterministic random

Compare K=2500 vs K=5000, seeds 42, 123.

Usage:
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_patch_sampling.py
"""
import os, sys, time, subprocess, json
import numpy as np
from pathlib import Path

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
BASE_OUT = PROJECT / "results" / "patch_sampling_v2"
SEEDS = [42, 123]
GPU = "7"
RUNNER = PROJECT / "scripts/_run_single_exp.py"
BASE_SEED = 42  # fixed base seed for all random sampling


def make_config(seed, max_patches, out_dir):
    return {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "feature_base_dir": "/home/Public/lillan/features_result/C16_features",
            "modalities": ["HE"],
            "dir_mapping": {"HE": "C16_HE_features"},
            "input_dim": 768, "num_classes": 2, "max_patches": max_patches,
            "preload": False, "val_ratio": 0.2,
            "sampling": "random", "sample_seed": BASE_SEED,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
        },
        "training": {
            "batch_size": 1, "num_epochs": 25, "learning_rate": 1e-4,
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


def run_one(name, seed, K):
    out_dir = BASE_OUT / name / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = make_config(seed, K, out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logf = open(out_dir / "logs" / "stdout.log", "w")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(PROJECT),
        timeout=1200,
    )
    elapsed = time.time() - t0
    logf.close()

    rf = out_dir / "result.json"
    if rf.exists():
        with open(rf) as f:
            result = json.load(f)
    else:
        import re
        text = (out_dir / "logs" / "stdout.log").read_text()
        vm = re.search(r'VAL_AUC=([\d.]+)', text)
        tm = re.search(r'TEST_AUC=([\d.]+)\s+ACC=([\d.]+)\s+F1=([\d.]+)', text)
        if vm and tm:
            result = {"val_auc": float(vm.group(1)), "test_auc": float(tm.group(1)),
                       "test_acc": float(tm.group(2)), "test_f1": float(tm.group(3))}
        else:
            result = None
    return result, elapsed, proc.returncode


def main():
    print("=" * 70)
    print("Patch Sampling K Verification — Corrected Sampler")
    print(f"Train: per_epoch random  Val/Test: fixed random")
    print(f"Seeds: {SEEDS}  Base seed: {BASE_SEED}  GPU: {GPU}")
    print("=" * 70)

    jobs = []
    for K in [2500, 5000]:
        name = f"rand{K}"
        for seed in SEEDS:
            jobs.append((f"{name}_s{seed}", name, seed, K))

    print(f"Total: {len(jobs)} jobs\n")

    results = {}
    for i, (tag, name, seed, K) in enumerate(jobs):
        print(f"[{i+1}/{len(jobs)}] {tag} ...", end=" ", flush=True)
        if (BASE_OUT / name / f'seed{seed}' / 'result.json').exists():
            with open(BASE_OUT / name / f'seed{seed}' / 'result.json') as f:
                results[tag] = json.load(f)
            print(f'SKIP (already done)', flush=True)
            continue
        result, elapsed, rc = run_one(name, seed, K)
        if result:
            results[tag] = result
            print(f"Val={result['val_auc']:.4f} Test={result['test_auc']:.4f} ({elapsed:.0f}s)", flush=True)
        else:
            print(f"FAIL(rc={rc}) ({elapsed:.0f}s)", flush=True)

    # Summary
    if results:
        print("\n" + "=" * 80)
        print(f"{'Experiment':<20} {'Seed':>5} {'Val AUC':>9} {'Test AUC':>10} {'Acc':>8} {'F1':>8}")
        print("-" * 80)
        for tag in sorted(results):
            r = results[tag]
            exp, seed_s = tag.rsplit("_s", 1)
            print(f"{exp:<20} {seed_s:>5} {r['val_auc']:>9.4f} {r['test_auc']:>10.4f} {r.get('test_acc',0):>8.4f} {r.get('test_f1',0):>8.4f}")

        print(f"\n{'='*60}")
        print(f"{'K':<10} {'Val AUC':>14} {'Test AUC':>14} {'Δ':>8}")
        print(f"{'='*60}")
        for K in [2500, 5000]:
            group = [results[k] for k in results if k.startswith(f"rand{K}_s")]
            if len(group) < 2:
                continue
            vm = np.mean([g["val_auc"] for g in group])
            vs = np.std([g["val_auc"] for g in group])
            tm = np.mean([g["test_auc"] for g in group])
            ts = np.std([g["test_auc"] for g in group])
            print(f"K={K:<7} {vm:>8.4f}±{vs:.3f} {tm:>8.4f}±{ts:.3f} {vm-tm:>8.4f}")

        with open(BASE_OUT / "all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {BASE_OUT}/all_results.json")

    print("Done!")


if __name__ == "__main__":
    main()
