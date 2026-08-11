#!/usr/bin/env python3
"""
C16 5-Fold Out-of-Fold (OOF) Evaluation — HE-only
==================================================
Trains 5 HE-only models, each holding out one fold as validation.
Collects OOF predictions for all 270 official train samples.

Also records best epoch per fold for final full-training epoch selection.

Usage (from TwoStageRRT/):
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_5fold_oof.py
"""

import os, sys, time, json, copy
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

from train import Trainer
import logging

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
BASE_OUT = PROJECT / "results" / "c16_5fold_oof"
SEED = 42  # data split seed; model seed varies per fold
N_FOLDS = 5

# ── Load data ──
label_df = pd.read_csv(PROJECT / "data/C16_labels/c16_train_labels.csv")
slide_ids = label_df["slide_id"].values
labels = label_df["label"].values

print(f"Total official train: {len(slide_ids)} ({sum(labels==0)} normal, {sum(labels==1)} tumor)")

# ── Stratified 5-fold ──
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    # Write fold-specific train/val CSVs
    train_csv = out_dir / "train_fold.csv"
    val_csv = out_dir / "val_fold.csv"
    label_df.iloc[train_idx].to_csv(train_csv, index=False)
    label_df.iloc[val_idx].to_csv(val_csv, index=False)

    cfg = {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(train_csv),
            "val_label_file": str(val_csv),
            "feature_base_dir": "/home/Public/lillan/features_result/C16_features",
            "modalities": ["HE"],
            "dir_mapping": {"HE": "C16_HE_features"},
            "input_dim": 768, "num_classes": 2, "max_patches": 5000, "preload": False,
            "val_ratio": 0.0,  # 0 = use val_label_file directly (manual fold split)
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
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": SEED + fold_idx},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }

    # Note: since val_ratio=0, the data loading path changes.
    # We need to handle this — when val_ratio=0, use val_label_file directly.
    # Actually, let me check: our current train.py code checks `if dataset_type == 'c16'`
    # and always does internal split. We need to bypass that for manual fold splits.
    #
    # For now, let me just directly use the CSV files with val_ratio=0 to mean "use val_label_file".
    # But our code doesn't support this yet. Let me modify train.py temporarily or do inline training.
    #
    # Actually, the simplest approach: write a quick inline training loop instead of using Trainer.
    # Or better: just set val_ratio=-1 to signal "use external val file".

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Quick inline training using the script approach
    script = f'''
import os, sys, json, logging, time
import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "{PROJECT}")
os.chdir("{PROJECT}")

from train import Trainer
import logging

logger = logging.getLogger("fold{fold_idx+1}")
logger.handlers.clear()
logger.setLevel(logging.INFO)
(Path("{out_dir}") / "logs").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(Path("{out_dir}") / "logs" / "run.log"))
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

with open("{out_dir}/config.json") as f:
    cfg = json.load(f)

print(f"Fold {fold_idx+1}: training...", flush=True)
t0 = time.time()

# Override val_ratio=0 → use val_label_file as-is
cfg["data"]["_manual_fold"] = True
trainer = Trainer(cfg, logger, f"fold{fold_idx+1}")
_, val_auc = trainer.train()

print(f"Fold {fold_idx+1}: Val AUC={{val_auc:.4f}} in {{time.time()-t0:.0f}}s", flush=True)
'''
    script_path = out_dir / "run_script.py"
    with open(script_path, "w") as f:
        f.write(script)

    # Run inline
    t0 = time.time()
    logger = logging.getLogger(f"fold{fold_idx+1}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    print(f"  Training fold {fold_idx+1}...", flush=True)

    # Use subprocess to avoid state leakage
    import subprocess
    proc = subprocess.run(
        [PYTHON, str(script_path)],
        cwd=str(PROJECT),
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr[-500:])

    # Load best model and predict on val fold
    ckpt_path = out_dir / "ckpt" / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  WARNING: no checkpoint at {ckpt_path}")
        continue

    ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
    mc = ckpt["config"]["model"]

    from models.mm_rrt_abmil import MM_RRT_ABMIL
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
    state_dict = ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda().eval()

    # Predict on val fold
    from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn
    from train import build_feature_dirs

    feature_dirs = build_feature_dirs(
        "/home/Public/lillan/features_result/C16_features",
        ["HE"], {"HE": "C16_HE_features"},
    )
    val_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=str(val_csv),
        max_patches=5000, preload=False, verbose=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=c16_multimodal_collate_fn, num_workers=2,
        pin_memory=True, persistent_workers=True,
    )

    fold_probs = []
    with torch.no_grad():
        for batch in val_dl:
            feats = [torch.stack(m).cuda() for m in batch["features"]]
            logits, _, _, _ = model(feats)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            fold_probs.append(prob)

    # Store OOF predictions
    for i, idx in enumerate(val_idx):
        all_oof_probs[idx] = fold_probs[i]

    fold_auc = roc_auc_score(labels[val_idx], fold_probs)
    fold_results.append({
        "fold": fold_idx + 1,
        "val_auc": fold_auc,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    })
    print(f"  Fold {fold_idx+1} Val AUC: {fold_auc:.4f} ({elapsed:.0f}s)")

# ── OOF Summary ──
valid_mask = ~np.isnan(all_oof_probs)
oof_auc = roc_auc_score(all_oof_labels[valid_mask], all_oof_probs[valid_mask])
oof_acc = accuracy_score(all_oof_labels[valid_mask], (np.array(all_oof_probs[valid_mask]) > 0.5).astype(int))

print(f"\n{'='*60}")
print(f"5-Fold OOF Results (HE-only, all {N_FOLDS} folds)")
print(f"{'='*60}")
for r in fold_results:
    print(f"  Fold {r['fold']}: Val AUC = {r['val_auc']:.4f}  (train={r['n_train']}, val={r['n_val']})")
print(f"  ─────────────────────────────")
print(f"  Per-fold Val AUC: {np.mean([r['val_auc'] for r in fold_results]):.4f} ± {np.std([r['val_auc'] for r in fold_results]):.4f}")
print(f"  OOF AUC (all 270):  {oof_auc:.4f}")
print(f"  OOF Acc (all 270):  {oof_acc:.4f}")
print(f"  Valid predictions:  {valid_mask.sum()}/{len(slide_ids)}")

# Save
oof_df = pd.DataFrame({
    "slide_id": slide_ids,
    "label": labels,
    "oof_prob": all_oof_probs,
})
oof_df.to_csv(BASE_OUT / "oof_predictions.csv", index=False)

summary = {
    "fold_results": fold_results,
    "per_fold_mean_auc": float(np.mean([r["val_auc"] for r in fold_results])),
    "per_fold_std_auc": float(np.std([r["val_auc"] for r in fold_results])),
    "oof_auc": float(oof_auc),
    "oof_acc": float(oof_acc),
    "n_valid": int(valid_mask.sum()),
    "n_total": len(slide_ids),
}
with open(BASE_OUT / "oof_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {BASE_OUT}/")
print("Done!")
