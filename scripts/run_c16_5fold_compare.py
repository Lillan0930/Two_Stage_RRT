#!/usr/bin/env python3
"""
C16 5-Fold HE-only vs HE+PR Two-stage — Corrected Sampler
============================================================
Protocol (LOCKED):
  Train: per_epoch random K=2500
  Val:   fixed random K=2500
  Test:  fixed random K=2500
  HE/PR: same patch indices (guaranteed by dataset)
  Seeds: 42/123 (2 seeds × 5 folds × 2 models = 20 runs)

All folds identical between HE-only and Two-stage.

Usage:
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_5fold_compare.py
"""
import os, sys, time, json, subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from train import build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
RUNNER = PROJECT / "scripts/_run_single_exp.py"
BASE_OUT = PROJECT / "results" / "c16_5fold_compare"
N_FOLDS = 5
K = 2500
SAMPLE_SEED = 42
SEEDS = [42, 123]
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"

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


def make_config(model_name, modalities, dir_mapping, seed, train_csv, out_dir):
    num_mod = len(modalities)
    return {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(train_csv),
            "feature_base_dir": FEATURE_BASE,
            "modalities": modalities,
            "dir_mapping": dir_mapping,
            "input_dim": 768, "num_classes": 2, "max_patches": K,
            "preload": False, "val_ratio": 0.2,
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
        cwd=str(PROJECT), timeout=1200,
    )
    elapsed = time.time() - t0
    logf.close()

    rf = out_dir / "result.json"
    if rf.exists():
        with open(rf) as f:
            return json.load(f), elapsed
    return None, elapsed


def predict_fold(ckpt_path, modalities, dir_mapping, csv_path, per_epoch=False):
    """Predict probabilities for a given CSV using a trained checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
    mc = ckpt["config"]["model"]

    model = MM_RRT_ABMIL(
        num_modalities=len(modalities), input_dim=768, num_classes=2,
        mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
        n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
        drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
        epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
        crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
        all_shortcut=mc.get("all_shortcut", True),
        crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
        fusion_type=mc.get("fusion_type", "two_stage_region"),
        abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.cuda().eval()

    feature_dirs = build_feature_dirs(FEATURE_BASE, modalities, dir_mapping)
    ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=str(csv_path),
        max_patches=K, preload=False, verbose=False,
        sampling="random", sample_seed=SAMPLE_SEED, per_epoch=per_epoch,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=c16_multimodal_collate_fn, num_workers=0,
                    pin_memory=True)

    probs, slide_ids_list, labels_list = [], [], []
    with torch.no_grad():
        for batch in dl:
            feats = [torch.stack(m).cuda() for m in batch["features"]]
            logits, _, _, _ = model(feats)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probs.append(prob)
            slide_ids_list.append(batch["slide_ids"][0])
            labels_list.append(batch["labels"].item())

    return dict(zip(slide_ids_list, probs)), dict(zip(slide_ids_list, labels_list))


def main():
    print("=" * 70)
    print("5-Fold HE-only vs Two-stage — Corrected Sampler")
    print(f"K={K}  per_epoch train  fixed val/test  seeds={SEEDS}")
    print("=" * 70)

    label_df = pd.read_csv(PROJECT / "data/C16_labels/c16_train_labels.csv")
    slide_ids_all = label_df["slide_id"].values
    labels_all = label_df["label"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_results = {}

    for model_name, mcfg in MODEL_CONFIGS.items():
        modalities = mcfg["modalities"]
        dir_mapping = mcfg["dir_mapping"]
        print(f"\n{'='*60}")
        print(f"  {model_name}: {modalities}")
        print(f"{'='*60}")

        oof_probs = np.full(len(slide_ids_all), np.nan)
        test_probs_all = {}  # seed → [probs]

        for seed in SEEDS:
            fold_oof_probs = np.full(len(slide_ids_all), np.nan)
            seed_results = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(slide_ids_all, labels_all)):
                out_dir = BASE_OUT / model_name / f"seed{seed}" / f"fold{fold_idx+1}"
                out_dir.mkdir(parents=True, exist_ok=True)
                train_csv = out_dir / "train_fold.csv"
                val_csv = out_dir / "val_fold.csv"
                label_df.iloc[train_idx].to_csv(train_csv, index=False)
                label_df.iloc[val_idx].to_csv(val_csv, index=False)

                cfg = make_config(model_name, modalities, dir_mapping, seed, train_csv, out_dir)

                tag = f"{model_name}_s{seed}_f{fold_idx+1}"
                result, elapsed = train_one(out_dir, cfg)

                if result is None:
                    print(f"  [{tag}] FAILED", flush=True)
                    seed_results.append(None)
                    continue

                # OOF prediction on hold-out fold
                ckpt_path = out_dir / "ckpt" / "best_model.pt"
                oof_map, _ = predict_fold(ckpt_path, modalities, dir_mapping, val_csv)
                for i, idx in enumerate(val_idx):
                    sid = label_df.iloc[idx]["slide_id"]
                    if sid in oof_map:
                        fold_oof_probs[idx] = oof_map[sid]

                valid = ~np.isnan(fold_oof_probs[val_idx])
                if valid.sum() > 1:
                    fold_oof_auc = roc_auc_score(
                        labels_all[val_idx][valid], fold_oof_probs[val_idx][valid]
                    )
                else:
                    fold_oof_auc = float('nan')

                print(f"  [{tag}] Val={result['val_auc']:.4f} OOF={fold_oof_auc:.4f} ({elapsed:.0f}s)", flush=True)
                seed_results.append({
                    "fold": fold_idx + 1,
                    "internal_val_auc": result["val_auc"],
                    "oof_auc": fold_oof_auc,
                    "test_auc_from_runner": result.get("test_auc", 0),
                })

            # Seed-level OOF
            valid_all = ~np.isnan(fold_oof_probs)
            seed_oof_auc = roc_auc_score(labels_all[valid_all], fold_oof_probs[valid_all]) if valid_all.sum() > 1 else float('nan')
            oof_probs = fold_oof_probs  # overwrite with last seed for final summary

            print(f"  Seed {seed} Pooled OOF: {seed_oof_auc:.4f}", flush=True)
            all_results[f"{model_name}_s{seed}"] = {
                "seed": seed,
                "fold_results": seed_results,
                "pooled_oof_auc": float(seed_oof_auc),
                "mean_internal_val": float(np.nanmean([r["internal_val_auc"] for r in seed_results if r])),
                "mean_oof": float(np.nanmean([r["oof_auc"] for r in seed_results if r])),
            }

        # ── Independent test (best seed checkpoint per fold) ──
        print(f"\n  {model_name} Independent Test (5-fold ensemble):")
        test_csv = PROJECT / "data/C16_labels/c16_test_labels.csv"
        test_df = pd.read_csv(test_csv)
        test_slide_ids = test_df["slide_id"].values
        test_labels = test_df["label"].values

        # Use seed 42 for final test eval (deterministic)
        test_ensemble_probs = None
        for fold_idx in range(N_FOLDS):
            ckpt_path = BASE_OUT / model_name / "seed42" / f"fold{fold_idx+1}" / "ckpt" / "best_model.pt"
            if not ckpt_path.exists():
                continue
            fold_map, _ = predict_fold(ckpt_path, modalities, dir_mapping, test_csv)
            fold_probs = np.array([fold_map.get(sid, 0.5) for sid in test_slide_ids])
            if test_ensemble_probs is None:
                test_ensemble_probs = fold_probs
            else:
                test_ensemble_probs += fold_probs
        test_ensemble_probs /= N_FOLDS

        test_auc = roc_auc_score(test_labels, test_ensemble_probs)
        test_acc = accuracy_score(test_labels, (test_ensemble_probs > 0.5).astype(int))
        print(f"    Test AUC (5-fold ensemble): {test_auc:.4f}")
        print(f"    Test Acc (5-fold ensemble): {test_acc:.4f}")

        all_results[f"{model_name}_test"] = {
            "test_auc_ensemble": float(test_auc),
            "test_acc_ensemble": float(test_acc),
        }

    # ── Final Summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Seed':>5} {'Int Val':>9} {'OOF Mean':>9} {'OOF Pooled':>10} {'Test Ens':>9}")
    print("-" * 65)
    for model_name in MODEL_CONFIGS:
        for seed in SEEDS:
            key = f"{model_name}_s{seed}"
            if key not in all_results:
                continue
            r = all_results[key]
            print(f"{model_name:<15} {seed:>5} {r['mean_internal_val']:>9.4f} {r['mean_oof']:>9.4f} {r['pooled_oof_auc']:>10.4f} {'—':>9}")
        tkey = f"{model_name}_test"
        if tkey in all_results:
            tr = all_results[tkey]
            print(f"{'':>21} {'—':>5} {'—':>9} {'—':>9} {'—':>10} {tr['test_auc_ensemble']:>9.4f}")

    with open(BASE_OUT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {BASE_OUT}/summary.json")
    print("Done!")


if __name__ == "__main__":
    main()
