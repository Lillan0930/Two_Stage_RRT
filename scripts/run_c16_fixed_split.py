#!/usr/bin/env python3
"""
C16 Fixed-Split — HE-only vs HE+PR Two-stage — 5 Seeds
========================================================
Protocol (LOCKED):
  Train: per_epoch random K=2500
  Val:   fixed random K=2500 (external CSV: data/C16_labels/fixed_split/val.csv)
  Test:  fixed random K=2500
  HE/PR: same patch indices (guaranteed by dataset)
  Split: fixed once (StratifiedShuffleSplit, random_state=42), all seeds use same split
  Seeds: 42, 123, 456, 789, 1024 — only affect model init, dropout, train randomness, per-epoch sampling
  No 5-fold, no OOF, no K search, no test-time ensemble, no Random PR, no new fusion modules

Saves per-slide test predictions and computes paired difference statistics.

Usage:
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_fixed_split.py
"""
import os, sys, time, json, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import stats

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

RUNNER = PROJECT / "scripts/_run_fixed_split_exp.py"
BASE_OUT = PROJECT / "results" / "c16_fixed_split"
K = 2500
SAMPLE_SEED = 42
SEEDS = [42, 123, 456, 789, 1024]
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"

FIXED_TRAIN = PROJECT / "data/C16_labels/fixed_split/train.csv"
FIXED_VAL = PROJECT / "data/C16_labels/fixed_split/val.csv"

MODEL_CONFIGS = {
    "HE_only": {
        "modalities": ["HE"],
        "dir_mapping": {"HE": "C16_HE_features"},
    },
    "Two_stage": {
        "modalities": ["HE", "PR"],
        "dir_mapping": {"HE": "C16_HE_features", "PR": "C16_PR_features"},
    },
}


def make_config(model_name, modalities, dir_mapping, seed, out_dir):
    num_mod = len(modalities)
    return {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(FIXED_TRAIN),
            "val_label_file": str(FIXED_VAL),  # external fixed val
            "feature_base_dir": FEATURE_BASE,
            "modalities": modalities,
            "dir_mapping": dir_mapping,
            "input_dim": 768, "num_classes": 2, "max_patches": K,
            "preload": False,
            "sampling": "random", "sample_seed": SAMPLE_SEED,
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


def train_one(out_dir, cfg):
    """Train via subprocess, return result dict or None."""
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logf = open(out_dir / "logs" / "stdout.log", "w")
    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(PROJECT), timeout=1800,
    )
    elapsed = time.time() - t0
    logf.close()

    rf = out_dir / "result.json"
    if rf.exists():
        with open(rf) as f:
            return json.load(f), elapsed
    return None, elapsed


def paired_stats(values_a, values_b):
    """Compute paired t-test and mean difference CI."""
    diffs = np.array(values_a) - np.array(values_b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    ci95 = stats.t.ppf(0.975, n - 1) * se_diff
    t_stat, p_val = stats.ttest_rel(values_a, values_b)
    return {
        "mean_diff": float(mean_diff),
        "std_diff": float(std_diff),
        "se_diff": float(se_diff),
        "ci95_lower": float(mean_diff - ci95),
        "ci95_upper": float(mean_diff + ci95),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
    }


def main():
    print("=" * 70)
    print("C16 Fixed-Split — HE-only vs Two-stage — 5 Seeds")
    print(f"K={K}  per_epoch train  fixed val/test  seeds={SEEDS}")
    print(f"Train CSV: {FIXED_TRAIN}")
    print(f"Val   CSV: {FIXED_VAL}")
    print("=" * 70)

    # Verify split files
    train_df = pd.read_csv(FIXED_TRAIN)
    val_df = pd.read_csv(FIXED_VAL)
    train_ids = set(train_df["slide_id"])
    val_ids = set(val_df["slide_id"])
    assert len(train_ids & val_ids) == 0, "Train/Val overlap!"
    print(f"Split: train={len(train_df)} ({dict(train_df['label'].value_counts())})")
    print(f"       val={len(val_df)}   ({dict(val_df['label'].value_counts())})")
    test_overlap = train_ids | val_ids
    test_df = pd.read_csv(PROJECT / "data/C16_labels/c16_test_labels.csv")
    test_ids = set(test_df["slide_id"])
    assert len(test_overlap & test_ids) == 0, "Train/Val overlaps with Test!"
    print(f"       test={len(test_df)} ({dict(test_df['label'].value_counts())})")
    print()

    all_jobs = []
    for model_name in MODEL_CONFIGS:
        for seed in SEEDS:
            all_jobs.append((model_name, seed))

    print(f"Total: {len(all_jobs)} experiments\n")

    # ── Run all experiments sequentially ──
    all_results = {}  # key: "model_s{seed}" → result dict
    preds = {}  # key: "model_s{seed}" → dataframe of per-slide predictions

    for i, (model_name, seed) in enumerate(all_jobs):
        mcfg = MODEL_CONFIGS[model_name]
        out_dir = BASE_OUT / model_name / f"seed{seed}"
        tag = f"{model_name}_s{seed}"
        print(f"[{i+1}/{len(all_jobs)}] {tag} ...", end=" ", flush=True)

        if (out_dir / "result.json").exists():
            with open(out_dir / "result.json") as f:
                all_results[tag] = json.load(f)
            pred_csv = out_dir / "test_predictions.csv"
            if pred_csv.exists():
                preds[tag] = pd.read_csv(pred_csv)
            print(f"SKIP (already done)", flush=True)
            continue

        cfg = make_config(model_name, mcfg["modalities"], mcfg["dir_mapping"], seed, out_dir)
        result, elapsed = train_one(out_dir, cfg)

        if result is not None:
            all_results[tag] = result
            pred_csv = out_dir / "test_predictions.csv"
            if pred_csv.exists():
                preds[tag] = pd.read_csv(pred_csv)
            print(f"Val={result['best_val_auc']:.4f} Test AUC={result['test_auc']:.4f} ({elapsed:.0f}s)", flush=True)
        else:
            print(f"FAILED ({elapsed:.0f}s)", flush=True)

    if not all_results:
        print("\nNo successful results. Exiting.")
        return

    # ── Per-model summary ──
    print("\n" + "=" * 80)
    print("PER-SEED RESULTS")
    print("=" * 80)
    header = f"{'Model':<15} {'Seed':>5} {'Best Epoch':>10} {'Val AUC':>9} {'Test AUC':>10} {'Acc':>8} {'F1':>8} {'Sens':>8} {'Spec':>8} {'Time':>8}"
    print(header)
    print("-" * len(header))

    for model_name in MODEL_CONFIGS:
        for seed in SEEDS:
            tag = f"{model_name}_s{seed}"
            if tag not in all_results:
                print(f"{model_name:<15} {seed:>5} {'—':>10} {'—':>9} {'—':>10} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
                continue
            r = all_results[tag]
            print(f"{model_name:<15} {seed:>5} {r.get('best_epoch',0):>10} "
                  f"{r['best_val_auc']:>9.4f} {r['test_auc']:>10.4f} "
                  f"{r.get('test_acc',0):>8.4f} {r.get('test_f1',0):>8.4f} "
                  f"{r.get('test_sensitivity',0):>8.4f} {r.get('test_specificity',0):>8.4f} "
                  f"{r.get('train_time_s',0):>8.0f}")

    # ── Aggregated summary ──
    print("\n" + "=" * 80)
    print("AGGREGATED SUMMARY (mean ± std across 5 seeds)")
    print("=" * 80)
    agg_header = f"{'Model':<15} {'Val AUC':>18} {'Test AUC':>18} {'Acc':>18} {'Sens':>18} {'Spec':>18}"
    print(agg_header)
    print("-" * len(agg_header))

    agg = {}
    for model_name in MODEL_CONFIGS:
        vals = []
        test_aucs = []
        accs = []
        sens = []
        specs = []
        for seed in SEEDS:
            tag = f"{model_name}_s{seed}"
            if tag not in all_results:
                continue
            r = all_results[tag]
            vals.append(r['best_val_auc'])
            test_aucs.append(r['test_auc'])
            accs.append(r.get('test_acc', 0))
            sens.append(r.get('test_sensitivity', 0))
            specs.append(r.get('test_specificity', 0))
        if vals:
            agg[model_name] = {
                "val_auc": (np.mean(vals), np.std(vals)),
                "test_auc": (np.mean(test_aucs), np.std(test_aucs)),
                "acc": (np.mean(accs), np.std(accs)),
                "sensitivity": (np.mean(sens), np.std(sens)),
                "specificity": (np.mean(specs), np.std(specs)),
            }
            print(f"{model_name:<15} "
                  f"{np.mean(vals):>8.4f}±{np.std(vals):.4f}   "
                  f"{np.mean(test_aucs):>8.4f}±{np.std(test_aucs):.4f}   "
                  f"{np.mean(accs):>8.4f}±{np.std(accs):.4f}   "
                  f"{np.mean(sens):>8.4f}±{np.std(sens):.4f}   "
                  f"{np.mean(specs):>8.4f}±{np.std(specs):.4f}")

    # ── Paired difference analysis ──
    if "HE_only" in agg and "Two_stage" in agg:
        he_test_aucs = []
        ts_test_aucs = []
        for seed in SEEDS:
            he_tag = f"HE_only_s{seed}"
            ts_tag = f"Two_stage_s{seed}"
            if he_tag in all_results and ts_tag in all_results:
                he_test_aucs.append(all_results[he_tag]["test_auc"])
                ts_test_aucs.append(all_results[ts_tag]["test_auc"])

        print("\n" + "=" * 80)
        print("PAIRED DIFFERENCE ANALYSIS (Two_stage − HE_only)")
        print("=" * 80)
        if len(he_test_aucs) >= 3:
            ps = paired_stats(ts_test_aucs, he_test_aucs)
            print(f"  HE_only Test AUC:    {np.mean(he_test_aucs):.4f} ± {np.std(he_test_aucs):.4f}")
            print(f"  Two_stage Test AUC:  {np.mean(ts_test_aucs):.4f} ± {np.std(ts_test_aucs):.4f}")
            print(f"  Mean Δ (TS − HE):    {ps['mean_diff']:+.4f}")
            print(f"  95% CI:              [{ps['ci95_lower']:+.4f}, {ps['ci95_upper']:+.4f}]")
            print(f"  Paired t-test:       t={ps['t_statistic']:.3f}, p={ps['p_value']:.4f}")
            if ps['p_value'] < 0.05:
                print(f"  → Significant at α=0.05")
            else:
                print(f"  → NOT significant at α=0.05")

        # Per-slide agreement
        print("\n  Per-slide prediction correlation:")
        all_slide_ids = None
        for seed in SEEDS:
            he_tag = f"HE_only_s{seed}"
            ts_tag = f"Two_stage_s{seed}"
            if he_tag in preds and ts_tag in preds:
                he_df = preds[he_tag]
                ts_df = preds[ts_tag]
                merged = he_df.merge(ts_df, on="slide_id", suffixes=("_he", "_ts"))
                if len(merged) > 0:
                    from sklearn.metrics import cohen_kappa_score
                    agree = (merged["prediction_he"] == merged["prediction_ts"]).mean()
                    kappa = cohen_kappa_score(merged["prediction_he"], merged["prediction_ts"])
                    prob_corr = merged["probability_he"].corr(merged["probability_ts"])
                    print(f"    Seed {seed}: agreement={agree:.3f}, κ={kappa:.4f}, "
                          f"prob corr={prob_corr:.4f}")
                    if all_slide_ids is None:
                        all_slide_ids = merged["slide_id"].values

        # Save paired results
        paired_results = {
            "he_only": {
                "test_auc_mean": float(np.mean(he_test_aucs)),
                "test_auc_std": float(np.std(he_test_aucs)),
            },
            "two_stage": {
                "test_auc_mean": float(np.mean(ts_test_aucs)),
                "test_auc_std": float(np.std(ts_test_aucs)),
            },
            "paired_stats": ps,
            "per_seed_aucs": {
                "seeds": SEEDS,
                "HE_only": [all_results[f"HE_only_s{s}"]["test_auc"] if f"HE_only_s{s}" in all_results else None for s in SEEDS],
                "Two_stage": [all_results[f"Two_stage_s{s}"]["test_auc"] if f"Two_stage_s{s}" in all_results else None for s in SEEDS],
            },
        }
        with open(BASE_OUT / "paired_analysis.json", "w") as f:
            json.dump(paired_results, f, indent=2)

    # ── Save full summary ──
    save_all = {}
    for k, v in all_results.items():
        save_all[k] = v
    with open(BASE_OUT / "all_results.json", "w") as f:
        json.dump(save_all, f, indent=2)

    print(f"\nResults saved to {BASE_OUT}/")
    print("Done!")


if __name__ == "__main__":
    main()
