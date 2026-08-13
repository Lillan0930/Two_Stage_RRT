#!/usr/bin/env python3
"""
C16 5-Fold Out-of-Fold (OOF) Evaluation — HE-only, corrected sampler
======================================================================
Train: per_epoch random K=2500
Val:   fixed random K=2500 (internal StratifiedShuffleSplit 80/20)
OOF:   fixed random K=2500 (hold-out 1/5 fold, truly unseen)

Usage (from TwoStageRRT/):
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_5fold_oof.py
"""
import os, sys, time, json, subprocess
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from train import build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
RUNNER = PROJECT / "scripts/_run_single_exp.py"
BASE_OUT = PROJECT / "results" / "c16_5fold_oof_v2"
DATA_SPLIT_SEED = 42
N_FOLDS = 5
K = 2500
SAMPLE_SEED = 42

# ── Load data ──
label_df = pd.read_csv(PROJECT / "data/C16_labels/c16_train_labels.csv")
slide_ids = label_df["slide_id"].values
labels = label_df["label"].values
print(f"Total official train: {len(slide_ids)} ({sum(labels==0)} normal, {sum(labels==1)} tumor)")

# ── Stratified 5-fold ──
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=DATA_SPLIT_SEED)

fold_results = []
all_oof_probs = np.full(len(slide_ids), np.nan)
all_oof_labels = labels.copy()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(slide_ids, labels)):
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")
    print(f"  val labels: {dict(zip(*np.unique(labels[val_idx], return_counts=True)))}")

    out_dir = BASE_OUT / f"fold{fold_idx+1}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)

    # Write fold-specific train CSV (Trainer does internal val split from this)
    train_csv = out_dir / "train_fold.csv"
    val_csv = out_dir / "val_fold.csv"   # UNSEEN hold-out, never given to Trainer
    label_df.iloc[train_idx].to_csv(train_csv, index=False)
    label_df.iloc[val_idx].to_csv(val_csv, index=False)

    cfg = {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(train_csv),
            "feature_base_dir": "/home/Public/lillan/features_result/C16_features",
            "modalities": ["HE"],
            "dir_mapping": {"HE": "C16_HE_features"},
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
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": DATA_SPLIT_SEED + fold_idx},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Train via subprocess ──
    t0 = time.time()
    logf = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.run(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=str(PROJECT), timeout=1200,
    )
    logf.close()
    elapsed = time.time() - t0

    result_file = out_dir / "result.json"
    if not result_file.exists():
        print(f"  Fold {fold_idx+1}: FAILED (no result.json)")
        continue
    with open(result_file) as f:
        train_result = json.load(f)

    internal_val_auc = train_result.get("val_auc", 0)
    print(f"  Fold {fold_idx+1}: internal Val AUC={internal_val_auc:.4f} ({elapsed:.0f}s)")

    # ── OOF predict on hold-out fold ──
    ckpt_path = out_dir / "ckpt" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
    mc = ckpt["config"]["model"]

    model = MM_RRT_ABMIL(
        num_modalities=1, input_dim=768, num_classes=2,
        mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
        n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
        drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
        epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
        crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
        all_shortcut=mc.get("all_shortcut", True),
        crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
        fusion_type="two_stage_region",
        abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.cuda().eval()

    feature_dirs = build_feature_dirs(
        "/home/Public/lillan/features_result/C16_features",
        ["HE"], {"HE": "C16_HE_features"},
    )
    # OOF prediction: fixed random, same K and seed as val
    oof_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=str(val_csv),
        max_patches=K, preload=False, verbose=False,
        sampling="random", sample_seed=SAMPLE_SEED, per_epoch=False,
    )
    oof_dl = DataLoader(oof_ds, batch_size=1, shuffle=False,
                        collate_fn=c16_multimodal_collate_fn, num_workers=0,
                        pin_memory=True)

    fold_probs = []
    fold_slide_ids = []
    with torch.no_grad():
        for batch in oof_dl:
            feats = [torch.stack(m).cuda() for m in batch["features"]]
            logits, _, _, _ = model(feats)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            fold_probs.append(prob)
            fold_slide_ids.append(batch["slide_ids"][0])

    # Map predictions back by slide_id
    val_slide_ids = label_df.iloc[val_idx]["slide_id"].values
    oof_probs_by_id = dict(zip(fold_slide_ids, fold_probs))
    for i, idx in enumerate(val_idx):
        sid = label_df.iloc[idx]["slide_id"]
        if sid in oof_probs_by_id:
            all_oof_probs[idx] = oof_probs_by_id[sid]

    fold_auc = roc_auc_score(
        [labels[idx] for idx in val_idx if not np.isnan(all_oof_probs[idx])],
        [all_oof_probs[idx] for idx in val_idx if not np.isnan(all_oof_probs[idx])],
    )
    fold_results.append({
        "fold": fold_idx + 1,
        "internal_val_auc": internal_val_auc,
        "oof_auc": fold_auc,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    })
    print(f"  Fold {fold_idx+1}: OOF AUC={fold_auc:.4f}")

# ── OOF Summary ──
valid_mask = ~np.isnan(all_oof_probs)
oof_auc = roc_auc_score(all_oof_labels[valid_mask], all_oof_probs[valid_mask])
oof_acc = accuracy_score(all_oof_labels[valid_mask], (np.array(all_oof_probs[valid_mask]) > 0.5).astype(int))

print(f"\n{'='*60}")
print(f"5-Fold OOF Results (HE-only, K={K}, corrected sampler)")
print(f"{'='*60}")
for r in fold_results:
    print(f"  Fold {r['fold']}: Internal Val={r['internal_val_auc']:.4f}  OOF={r['oof_auc']:.4f}  (train={r['n_train']}, val={r['n_val']})")
print(f"  {'─'*50}")
print(f"  Mean Internal Val: {np.mean([r['internal_val_auc'] for r in fold_results]):.4f} ± {np.std([r['internal_val_auc'] for r in fold_results]):.4f}")
print(f"  Mean OOF AUC:      {np.mean([r['oof_auc'] for r in fold_results]):.4f} ± {np.std([r['oof_auc'] for r in fold_results]):.4f}")
print(f"  Pooled OOF AUC:    {oof_auc:.4f}")
print(f"  OOF Acc:           {oof_acc:.4f}")
print(f"  Valid:             {valid_mask.sum()}/{len(slide_ids)}")

# ── Test eval using all 5 checkpoints (ensemble by averaging probs) ──
print(f"\n{'='*60}")
print(f"Independent Test — 5-Fold Ensemble")
print(f"{'='*60}")

feature_dirs = build_feature_dirs(
    "/home/Public/lillan/features_result/C16_features",
    ["HE"], {"HE": "C16_HE_features"},
)
test_ds = C16MultimodalDataset(
    feature_dirs=feature_dirs,
    label_file=str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
    max_patches=K, preload=False, verbose=False,
    sampling="random", sample_seed=SAMPLE_SEED, per_epoch=False,
)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                     collate_fn=c16_multimodal_collate_fn, num_workers=0,
                     pin_memory=True)

all_test_probs = []
all_test_labels = []
with torch.no_grad():
    for batch in test_dl:
        feats = [torch.stack(m).cuda() for m in batch["features"]]
        fold_probs_for_sample = []
        for fold_idx in range(N_FOLDS):
            ckpt_path = BASE_OUT / f"fold{fold_idx+1}" / "ckpt" / "best_model.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
            mc = ckpt["config"]["model"]
            model = MM_RRT_ABMIL(
                num_modalities=1, input_dim=768, num_classes=2,
                mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
                n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
                drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
                epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
                crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
                all_shortcut=mc.get("all_shortcut", True),
                crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
                fusion_type="two_stage_region",
                abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model = model.cuda().eval()
            logits, _, _, _ = model(feats)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            fold_probs_for_sample.append(prob)
        avg_prob = np.mean(fold_probs_for_sample) if fold_probs_for_sample else 0.5
        all_test_probs.append(avg_prob)
        all_test_labels.append(batch["labels"].item())

test_auc = roc_auc_score(all_test_labels, all_test_probs)
test_acc = accuracy_score(all_test_labels, (np.array(all_test_probs) > 0.5).astype(int))
print(f"  5-Fold Ensemble Test AUC: {test_auc:.4f}")
print(f"  5-Fold Ensemble Test Acc: {test_acc:.4f}")

# Save
oof_df = pd.DataFrame({
    "slide_id": slide_ids,
    "label": labels,
    "oof_prob": all_oof_probs,
})
oof_df.to_csv(BASE_OUT / "oof_predictions.csv", index=False)

summary = {
    "K": K, "sample_seed": SAMPLE_SEED,
    "sampling": "random", "train": "per_epoch", "val_test": "fixed",
    "fold_results": fold_results,
    "internal_val_mean": float(np.mean([r["internal_val_auc"] for r in fold_results])),
    "internal_val_std": float(np.std([r["internal_val_auc"] for r in fold_results])),
    "oof_mean": float(np.mean([r["oof_auc"] for r in fold_results])),
    "oof_std": float(np.std([r["oof_auc"] for r in fold_results])),
    "pooled_oof_auc": float(oof_auc),
    "test_auc_5fold_ensemble": float(test_auc),
    "test_acc_5fold_ensemble": float(test_acc),
    "n_valid": int(valid_mask.sum()),
    "n_total": len(slide_ids),
}
with open(BASE_OUT / "oof_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {BASE_OUT}/")
print("Done!")
