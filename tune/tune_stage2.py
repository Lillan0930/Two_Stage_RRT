#!/usr/bin/env python
"""
Two-stage (HE+PR) Stage2 CrossStainingCRMSA 网格调参 — 3-fold 内部 CV

固定 Stage1 两个 encoder（HE=baseline 4/9/3，PR=调优 8/15/5），只调 Stage2。
第一轮网格: Stage2 crmsa_k ∈ {1,3,5,8} × drop_path ∈ {0,0.1,0.2}
            （Stage2 region_num=4, crmsa_heads=8 固定）。

objective = 3-fold 内部 CV 的 mean val AUC（不污染官方 Test，不依赖单次 Val）。

训练方式 = 差分 lr（方案A）:
    encoder（rrt_he/rrt_ihc/patch_to_emb/rrt_encoder）  lr = 1e-5
    Stage2+ABMIL（cross_region_mod/mil）                lr = 1e-4

轻量化: 不落 checkpoint / 曲线，只写 CSV + best_config.json，复用 tune/folds/。

用法:
    cd /home/Public/lillan/Two_Sage_RRT-/TwoStageRRT
    PY=/home/cxl/miniconda3/envs/rrtmil/bin/python
    $PY tune/tune_stage2.py --gpu 6
"""

import os
import sys
import gc
import json
import csv
import argparse
import warnings
from pathlib import Path

# 抑制 torch.load(weights_only=True) 的 TypedStorage deprecation 噪音
warnings.filterwarnings('ignore', message='.*TypedStorage is deprecated.*')

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from train import build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

# ── 常量 ─────────────────────────────────────────────────────────────
TUNE_DIR = PROJECT / "tune"
FOLDS_DIR = TUNE_DIR / "folds"

TRAIN_LABEL_FILE = PROJECT / "data/C16_labels/c16_train_labels.csv"  # 官方 train 270
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}

INPUT_DIM = 768
NUM_CLASSES = 2
MAX_PATCHES = 2500
SAMPLE_SEED = 42

NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 10
SCHED_PATIENCE = 5
SCHED_FACTOR = 0.5
BATCH_SIZE = 1
NUM_WORKERS = 2

# 固定共享超参（锁定）
FIXED = dict(
    mlp_dim=512, n_layers=2, n_heads=4, trans_dropout=0.1, epeg=True,
    cr_msa=True, all_shortcut=True, crmsa_heads=8, crmsa_mlp=False,
    fusion_type='two_stage_region', abmil_hidden_dim=256, dropout=0.25,
)

# 固定 Stage1 encoder（不要动）—— 结构参数各自 best config
ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3},   # HE-only baseline (Test 0.8080)
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5},  # PR-only tuned   (Test 0.7425)
}

# Stage2 固定（第一轮） + 网格
STAGE2_FIXED = dict(region_num=4, crmsa_heads=8, epeg=True, epeg_k=9,
                    crmsa_mlp=False, drop_out=0.1)
CRMSA_K_CHOICES = [1, 3, 5, 8]
DROP_PATH_CHOICES = [0.0, 0.1, 0.2]

# 差分 lr（方案A）
LR_STAGE1 = 1e-5
LR_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-5

STAGE1_PREFIXES = ('rrt_he.', 'rrt_ihc.', 'patch_to_emb.', 'rrt_encoder.')
STAGE2_PREFIXES = ('cross_region_mod.', 'mil.')


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def make_folds(n_folds=3, val_ratio=0.2, seed=42):
    """对官方 train (270) 做 n_folds 次独立分层 216/54 划分。"""
    df = pd.read_csv(TRAIN_LABEL_FILE)
    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=val_ratio, random_state=seed)
    folds = []
    for tr_idx, va_idx in sss.split(df, df['label']):
        folds.append((
            df.iloc[tr_idx].reset_index(drop=True),
            df.iloc[va_idx].reset_index(drop=True),
        ))
    return folds


def write_folds(folds):
    """把 fold 划分写成 CSV（一次性，复用）。返回 [(train_csv, val_csv), ...]。"""
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for f, (tr, va) in enumerate(folds):
        tp = FOLDS_DIR / f"train_f{f}.csv"
        vp = FOLDS_DIR / f"val_f{f}.csv"
        tr.to_csv(tp, index=False)
        va.to_csv(vp, index=False)
        paths.append((str(tp), str(vp)))
    return paths


def build_model(crmsa_k, drop_path):
    """Two-stage（HE+PR）：encoder 固定，Stage2 用给定 crmsa_k/drop_path。"""
    stage2_cfg = dict(STAGE2_FIXED)
    stage2_cfg['crmsa_k'] = crmsa_k
    stage2_cfg['drop_path'] = drop_path
    return MM_RRT_ABMIL(
        num_modalities=len(MODALITIES), modality_list=MODALITIES,
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
        mlp_dim=FIXED['mlp_dim'], dropout=FIXED['dropout'],
        # 以下 shared 值对 two-stage 路径无实际作用（被 encoder_cfg/stage2_cfg 覆盖），
        # 但会影响未使用的 self.rrt_encoder；encoders 的 drop_path 恒为 0.0。
        region_num=STAGE2_FIXED['region_num'], n_layers=FIXED['n_layers'],
        n_heads=FIXED['n_heads'], drop_path=0.0, trans_dropout=FIXED['trans_dropout'],
        epeg=FIXED['epeg'], epeg_k=STAGE2_FIXED['epeg_k'], crmsa_k=3,
        cr_msa=FIXED['cr_msa'], all_shortcut=FIXED['all_shortcut'],
        crmsa_heads=FIXED['crmsa_heads'], crmsa_mlp=FIXED['crmsa_mlp'],
        fusion_type=FIXED['fusion_type'], abmil_hidden_dim=FIXED['abmil_hidden_dim'],
        encoder_cfg=ENCODER_CFG,
        stage2_cfg=stage2_cfg,
    )


def _make_loaders(train_csv, val_csv):
    feature_dirs = build_feature_dirs(FEATURE_BASE, MODALITIES, DIR_MAPPING)
    train_ds = C16MultimodalDataset(
        feature_dirs, train_csv, max_patches=MAX_PATCHES, preload=False,
        verbose=False, sampling='random', sample_seed=SAMPLE_SEED, per_epoch=False,
    )
    val_ds = C16MultimodalDataset(
        feature_dirs, val_csv, max_patches=MAX_PATCHES, preload=False,
        verbose=False, sampling='random', sample_seed=SAMPLE_SEED, per_epoch=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=c16_multimodal_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=c16_multimodal_collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=False,
    )
    return train_ds, val_ds, train_loader, val_loader


def _class_weights(train_ds, device):
    labels = [s['label'] for s in train_ds.samples]
    counts = {c: labels.count(c) for c in range(NUM_CLASSES)}
    n_total = len(labels)
    w = [n_total / (NUM_CLASSES * max(counts.get(c, 1), 1)) for c in range(NUM_CLASSES)]
    return torch.tensor(w, device=device)


def _make_optimizer(model):
    """差分 lr（方案A）：Stage1 encoder 1e-5，Stage2+ABMIL 1e-4。"""
    stage1, stage2 = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(pre) for pre in STAGE1_PREFIXES):
            stage1.append(p)
        elif any(name.startswith(pre) for pre in STAGE2_PREFIXES):
            stage2.append(p)
        else:
            stage2.append(p)  # 兜底：未归类参数归入 Stage2
    n_stage1 = sum(p.numel() for p in stage1)
    n_stage2 = sum(p.numel() for p in stage2)
    print(f"  [lr-groups] Stage1(1e-5)={n_stage1} params | Stage2(1e-4)={n_stage2} params", flush=True)
    return torch.optim.Adam([
        {'params': stage1, 'lr': LR_STAGE1},
        {'params': stage2, 'lr': LR_STAGE2},
    ], weight_decay=WEIGHT_DECAY)


def _evaluate(model, val_loader, device):
    model.eval()
    probs, labels = [], []
    with torch.inference_mode():
        for batch in val_loader:
            feats = [torch.stack(m).to(device) for m in batch['features']]
            labels.append(batch['labels'])
            logits, _, _, _ = model(feats)
            probs.append(torch.softmax(logits, dim=-1))
    y = torch.cat(labels).cpu().numpy()
    p = torch.cat(probs, dim=0)[:, 1].cpu().numpy()
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return 0.5


def train_one_fold(model, train_csv, val_csv, device):
    train_ds, val_ds, train_loader, val_loader = _make_loaders(train_csv, val_csv)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds, device))
    optimizer = _make_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=SCHED_PATIENCE, factor=SCHED_FACTOR,
    )

    best_val_auc = 0.0
    patience_left = EARLY_STOP_PATIENCE
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            feats = [torch.stack(m).to(device) for m in batch['features']]
            labels = batch['labels'].to(device)
            logits, _, _, _ = model(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_auc = _evaluate(model, val_loader, device)
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_auc


def main():
    ap = argparse.ArgumentParser(description="Two-stage Stage2 CRMSA grid tuning")
    ap.add_argument('--gpu', type=str, default='6')
    ap.add_argument('--n-folds', type=int, default=3)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={args.gpu})", flush=True)

    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    folds = make_folds(args.n_folds, 0.2, seed=42)
    fold_paths = write_folds(folds)
    for f, (tr, va) in enumerate(folds):
        print(f"Fold {f}: train={len(tr)} val={len(va)} ({dict(va['label'].value_counts())})", flush=True)

    grid = [(k, dp) for k in CRMSA_K_CHOICES for dp in DROP_PATH_CHOICES]
    print(f"\nGrid: {len(grid)} configs = crmsa_k {CRMSA_K_CHOICES} x drop_path {DROP_PATH_CHOICES}", flush=True)

    results = []
    for gi, (crmsa_k, drop_path) in enumerate(grid):
        fold_aucs = []
        for f, (train_csv, val_csv) in enumerate(fold_paths):
            set_seed(42)
            model = build_model(crmsa_k, drop_path).to(device)
            auc = train_one_fold(model, train_csv, val_csv, device)
            fold_aucs.append(auc)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        results.append({
            'crmsa_k': crmsa_k, 'drop_path': drop_path,
            'fold_aucs': [round(x, 4) for x in fold_aucs],
            'mean_auc': round(mean_auc, 4), 'std_auc': round(std_auc, 4),
        })
        print(f"[{gi+1}/{len(grid)}] crmsa_k={crmsa_k} drop_path={drop_path} "
              f"folds={[f'{x:.4f}' for x in fold_aucs]} mean={mean_auc:.4f} ± {std_auc:.4f}",
              flush=True)

    # ── 写 CSV ──
    csv_path = TUNE_DIR / "stage2_grid_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crmsa_k", "drop_path", "fold0", "fold1", "fold2", "mean_auc", "std_auc"])
        for r in results:
            fa = r['fold_aucs']
            w.writerow([r['crmsa_k'], r['drop_path'],
                        fa[0], fa[1], fa[2], r['mean_auc'], r['std_auc']])
    print(f"\nSaved -> {csv_path}", flush=True)

    # ── best config ──
    best = max(results, key=lambda r: r['mean_auc'])
    best_cfg = {
        "objective": "mean_3fold_val_auc",
        "best_stage2": {
            "crmsa_k": best['crmsa_k'], "drop_path": best['drop_path'],
            **STAGE2_FIXED,
        },
        "best_value": best['mean_auc'],
        "fold_aucs": best['fold_aucs'], "fold_std": best['std_auc'],
        "fixed_encoder_cfg": ENCODER_CFG,
        "fixed_shared": FIXED,
        "training": {"lr_stage1": LR_STAGE1, "lr_stage2": LR_STAGE2,
                     "weight_decay": WEIGHT_DECAY, "num_epochs": NUM_EPOCHS,
                     "early_stop_patience": EARLY_STOP_PATIENCE},
        "data": {"modalities": MODALITIES, "dir_mapping": DIR_MAPPING,
                 "max_patches": MAX_PATCHES, "sample_seed": SAMPLE_SEED,
                 "n_folds": args.n_folds},
        "note": "下一步：把 best_stage2 + fixed_encoder_cfg 写进 two-stage config.json，跑 5 seeds 官方 Test。",
    }
    with open(TUNE_DIR / "stage2_best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"Best: crmsa_k={best['crmsa_k']} drop_path={best['drop_path']} "
          f"mean={best['mean_auc']} ± {best['std_auc']}", flush=True)
    print(f"Saved -> {TUNE_DIR / 'stage2_best_config.json'}", flush=True)


if __name__ == '__main__':
    main()
