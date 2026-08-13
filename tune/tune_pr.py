#!/usr/bin/env python
"""
PR-only R²T+ABMIL 超参数调优 — Optuna (3-fold 内部 CV)

目标 (objective) = 3-fold 内部 CV 的 mean val AUC
    * 不使用官方 Test（绝不污染 Test）
    * 不使用单次 Val（避免 "幸运 split" 虚高）
    * 每个 trial 训练 3 次 (216 train / 54 val)，返回 3 个 val AUC 的均值

搜索空间 (30 trials 左右):
    region_num   : categorical [2, 4, 8]
    epeg_k       : categorical [5, 9, 15]
    crmsa_k      : categorical [1, 3, 5]
    dropout      : float [0.1, 0.5]      (uniform)
    lr           : float [1e-5, 3e-4]    (log)
    weight_decay : float [1e-6, 1e-3]    (log)

协议与 results/c16_pr_baseline 一致:
    K=2500, sampling='random', sample_seed=42, 单模态官方 RRTEncoder + ABMIL
    batch_size=1, num_epochs=25, early_stop(patience=10, monitor=val_auc),
    scheduler=ReduceLROnPlateau(patience=5, factor=0.5), Adam, seed=42

轻量化:
    * 不保存任何 checkpoint (.pt) / 训练曲线 / per-trial 日志
    * 结果全部存 Optuna SQLite (tune/tune.db)
    * 3 份 fold CSV 只生成一次 (tune/folds/)，复用

用法:
    cd /home/Public/lillan/Two_Sage_RRT-/TwoStageRRT
    PY=/home/cxl/miniconda3/envs/rrtmil/bin/python
    $PY tune/tune_pr.py --n-trials 30 --gpu 7
    $PY tune/tune_pr.py --n-trials 30 --gpu 7 --study-name pr_rrt_tune_v2   # 新搜索
"""

import os
import sys
import gc
import json
import argparse
import time
import warnings
from pathlib import Path

# 抑制 torch.load(weights_only=True) 的 TypedStorage deprecation 噪音（每次加载 .pt 都会触发）
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

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train import build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

# ── 固定常量（与 PR-only 基线一致）─────────────────────────────────────
TUNE_DIR = PROJECT / "tune"
FOLDS_DIR = TUNE_DIR / "folds"

TRAIN_LABEL_FILE = PROJECT / "data/C16_labels/c16_train_labels.csv"  # 官方 train 270
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITY = "PR"
DIR_MAPPING = {"PR": "C16_PR_features"}

INPUT_DIM = 768
NUM_CLASSES = 2
MAX_PATCHES = 2500          # K
SAMPLE_SEED = 42

NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 10
SCHED_PATIENCE = 5
SCHED_FACTOR = 0.5
BATCH_SIZE = 1
NUM_WORKERS = 2

# 非调参的固定超参（锁定，与基线 config.json 相同）
FIXED = dict(
    mlp_dim=512,
    n_layers=2,
    n_heads=4,
    drop_path=0.0,
    trans_dropout=0.1,
    epeg=True,
    cr_msa=True,
    all_shortcut=True,
    crmsa_heads=8,
    crmsa_mlp=False,
    fusion_type='two_stage_region',
    abmil_hidden_dim=256,
)

# 默认搜索空间（第一轮）。可用 --space JSON 覆盖（第二轮扩展上界、复用已有 trial）。
SPACE = {
    "region_num":   {"type": "cat",   "choices": [2, 4, 8]},
    "epeg_k":       {"type": "cat",   "choices": [5, 9, 15]},
    "crmsa_k":      {"type": "cat",   "choices": [1, 3, 5]},
    "dropout":      {"type": "float", "low": 0.1,  "high": 0.5,  "log": False},
    "lr":           {"type": "float", "low": 1e-5, "high": 3e-4, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
}


def _suggest(trial, name):
    """按 SPACE 定义采样一个参数。"""
    spec = SPACE[name]
    if spec["type"] == "cat":
        return trial.suggest_categorical(name, spec["choices"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec["log"])


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def make_folds(n_folds=3, val_ratio=0.2, seed=42):
    """对官方 train (270) 做 n_folds 次独立分层 216/54 划分，返回 (train_df, val_df) 列表。"""
    df = pd.read_csv(TRAIN_LABEL_FILE)
    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=val_ratio,
                                 random_state=seed)
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


def build_model(region_num, epeg_k, crmsa_k, dropout):
    """单模态 (PR) RRTEncoder + ABMIL。非调参项全部锁定为 FIXED。"""
    return MM_RRT_ABMIL(
        num_modalities=1,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        mlp_dim=FIXED['mlp_dim'],
        dropout=dropout,                    # 投影 dropout + ABMIL dropout_rate
        region_num=region_num,
        n_layers=FIXED['n_layers'],
        n_heads=FIXED['n_heads'],
        drop_path=FIXED['drop_path'],
        trans_dropout=FIXED['trans_dropout'],
        epeg=FIXED['epeg'],
        epeg_k=epeg_k,
        crmsa_k=crmsa_k,
        cr_msa=FIXED['cr_msa'],
        all_shortcut=FIXED['all_shortcut'],
        crmsa_heads=FIXED['crmsa_heads'],
        crmsa_mlp=FIXED['crmsa_mlp'],
        fusion_type=FIXED['fusion_type'],
        abmil_hidden_dim=FIXED['abmil_hidden_dim'],
    )


def _make_loaders(train_csv, val_csv):
    feature_dirs = build_feature_dirs(FEATURE_BASE, [MODALITY], DIR_MAPPING)
    # per_epoch=False → 固定 random (epoch=0)。与基线 Trainer 的有效行为一致
    #（基线 persistent_workers=True 使 set_epoch 不生效，等价于 epoch 恒为 0）。
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
    """inverse-frequency class weights，与基线 train() 一致。"""
    labels = [s['label'] for s in train_ds.samples]
    counts = {c: labels.count(c) for c in range(NUM_CLASSES)}
    n_total = len(labels)
    w = [n_total / (NUM_CLASSES * max(counts.get(c, 1), 1)) for c in range(NUM_CLASSES)]
    return torch.tensor(w, device=device)


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


def train_one_fold(model, train_csv, val_csv, lr, wd, device):
    """训练一个 fold，返回 best val AUC（不落盘任何 checkpoint）。"""
    train_ds, val_ds, train_loader, val_loader = _make_loaders(train_csv, val_csv)

    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
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

    # 显式释放 dataloader / worker 资源
    del train_loader, val_loader, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    return best_val_auc


def main():
    ap = argparse.ArgumentParser(description="Optuna PR-only R²T+ABMIL tuning")
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--study-name', type=str, default='pr_rrt_tune')
    ap.add_argument('--gpu', type=str, default='7')
    ap.add_argument('--n-folds', type=int, default=3)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--n-startup-trials', type=int, default=8,
                    help='MedianPruner: 前 N 个 trial 不剪枝')
    ap.add_argument('--space', type=str, default=None,
                    help='JSON 文件覆盖默认搜索空间（第二轮扩展搜索用）')
    args = ap.parse_args()

    if args.space:
        global SPACE
        with open(args.space) as f:
            SPACE = json.load(f)
        print(f"Search space overridden by {args.space}", flush=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} (CUDA_VISIBLE_DEVICES={args.gpu})", flush=True)

    # ── 生成 3-fold（一次性写盘）──
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    folds = make_folds(args.n_folds, args.val_ratio, seed=42)
    fold_paths = write_folds(folds)
    for f, (tr, va) in enumerate(folds):
        print(f"Fold {f}: train={len(tr)} val={len(va)} "
              f"({dict(va['label'].value_counts())})", flush=True)

    # ── Optuna study (SQLite, 轻量) ──
    storage = f"sqlite:///{TUNE_DIR}/tune.db"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=args.n_startup_trials,
                            n_warmup_steps=1, interval_steps=1),
        load_if_exists=True,
    )
    print(f"Study: {args.study_name}  storage={storage}", flush=True)

    def objective(trial):
        region_num = _suggest(trial, 'region_num')
        epeg_k = _suggest(trial, 'epeg_k')
        crmsa_k = _suggest(trial, 'crmsa_k')
        dropout = _suggest(trial, 'dropout')
        lr = _suggest(trial, 'lr')
        wd = _suggest(trial, 'weight_decay')

        fold_aucs = []
        for f, (train_csv, val_csv) in enumerate(fold_paths):
            set_seed(42)  # 每个 fold 从相同 init 重新训练（标准 CV 做法）
            model = build_model(region_num, epeg_k, crmsa_k, dropout).to(device)
            auc = train_one_fold(model, train_csv, val_csv, lr, wd, device)
            fold_aucs.append(auc)
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # 折间中间值上报（只上报前 n-1 折；末折直接返回，避免 completed trial 被剪掉）
            # MedianPruner(n_warmup_steps=1) → 只在 step>=1（跑完 ≥2 folds）后才可能剪枝
            if f < len(fold_paths) - 1:
                running_mean = float(np.mean(fold_aucs))
                trial.report(running_mean, step=f)
                if trial.should_prune():
                    print(f"  [pruned] trial={trial.number} "
                          f"fold_aucs={[f'{x:.4f}' for x in fold_aucs]}", flush=True)
                    raise optuna.exceptions.TrialPruned()

        mean_auc = float(np.mean(fold_aucs))
        trial.set_user_attr('fold_aucs', [round(x, 4) for x in fold_aucs])
        trial.set_user_attr('fold_std', round(float(np.std(fold_aucs)), 4))
        print(f"[trial {trial.number}] region_num={region_num} epeg_k={epeg_k} "
              f"crmsa_k={crmsa_k} dropout={dropout:.3f} lr={lr:.2e} wd={wd:.2e} "
              f"-> folds={[f'{x:.4f}' for x in fold_aucs]} mean={mean_auc:.4f}",
              flush=True)
        return mean_auc

    # 支持断点续跑：只补足到 n_trials
    finished = sum(1 for t in study.trials if t.state.is_finished())
    remaining = max(0, args.n_trials - finished)
    print(f"Existing finished trials: {finished}, will run {remaining} more.", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    # ── 汇总 ──
    print("\n" + "=" * 70, flush=True)
    print(f"Study: {args.study_name}  Trials: {len(study.trials)}", flush=True)
    try:
        best = study.best_trial
    except ValueError:
        print("No completed trials — nothing to summarize.", flush=True)
        return
    print(f"Best objective (mean 3-fold val AUC): {best.value:.4f}", flush=True)
    print(f"Best fold AUCs: {best.user_attrs.get('fold_aucs')} "
          f"(std={best.user_attrs.get('fold_std')})", flush=True)
    print("Best params:", flush=True)
    for k, v in best.params.items():
        print(f"  {k}: {v}", flush=True)

    best_cfg = {
        "study_name": args.study_name,
        "objective": "mean_3fold_val_auc",
        "best_value": best.value,
        "best_params": best.params,
        "fold_aucs": best.user_attrs.get('fold_aucs'),
        "fold_std": best.user_attrs.get('fold_std'),
        "fixed": FIXED,
        "data": {
            "modality": MODALITY, "dir_mapping": DIR_MAPPING,
            "max_patches": MAX_PATCHES, "sample_seed": SAMPLE_SEED,
            "n_folds": args.n_folds, "val_ratio": args.val_ratio,
        },
        "note": "下一步：把 best_params 写进 config.json，用 5 seeds 跑官方 Test。",
    }
    with open(TUNE_DIR / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\nSaved best config -> {TUNE_DIR / 'best_config.json'}", flush=True)


if __name__ == '__main__':
    main()
