#!/usr/bin/env python
"""
MM-RRT-Light + ABMIL Training Script
======================================
Config-driven multi-modal MIL training with logging, visualization,
and rich metrics. Supports 1–5 modalities specified in a YAML config.

Usage:
    python train.py --config config/config_dual_raw_er.yaml --exp_name dual_raw_er
"""

import os
import sys
import gc
import time
import yaml
import logging
import argparse
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.multimodal_dataset import MultiModalFeatureDataset, multimodal_collate_fn
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn
from utils.metrics import calculate_metrics, format_confusion_matrix, save_checkpoint


class FocalLoss(nn.Module):
    """Focal Loss — 自动聚焦难分类样本，减少易分类样本的梯度"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # 注：去掉 deterministic=True 以启用 cuDNN 最优 kernel 选择
    # 训练结果仍通过 manual_seed 保证可复现
    torch.backends.cudnn.benchmark = True


def he_to_pr_confidence_kd_loss(
    logits_he,
    logits_pr,
    T=2.0,
    conf_threshold=0.5,
    use_agreement_gate=False,
    eps=1e-6,
):
    """
    HE → PR confidence-weighted KD.

    logits_he: [B, C] — teacher, must be detached before calling
    logits_pr: [B, C] — student

    KD only updates PR branch; HE teacher is frozen.
    Higher HE confidence → higher KD weight.

    When use_agreement_gate=True, only distills samples where
    HE and PR predict in the same direction (margin sign agreement).
    """
    with torch.no_grad():
        p_he_T = F.softmax(logits_he.detach() / T, dim=-1)
        p_he = F.softmax(logits_he.detach(), dim=-1)
        conf = p_he.max(dim=-1, keepdim=True).values  # [B, 1]
        num_classes = logits_he.size(-1)
        min_conf = 1.0 / num_classes
        conf_weight = (conf - min_conf) / (1.0 - min_conf + eps)
        conf_weight = conf_weight.clamp(0.0, 1.0)
        if conf_threshold is not None:
            mask = (conf >= conf_threshold).float()
            conf_weight = conf_weight * mask

        # ── Sample-level agreement gate ──
        agree_ratio = torch.tensor(1.0)
        if use_agreement_gate and num_classes == 2:
            margin_he = logits_he.detach()[:, 1] - logits_he.detach()[:, 0]
            margin_pr = logits_pr.detach()[:, 1] - logits_pr.detach()[:, 0]
            agree = (margin_he * margin_pr > 0).float().unsqueeze(1)
            conf_weight = conf_weight * agree
            agree_ratio = agree.float().mean().detach()

    log_p_pr_T = F.log_softmax(logits_pr / T, dim=-1)
    kd_each = F.kl_div(log_p_pr_T, p_he_T, reduction="none").sum(dim=-1, keepdim=True)
    loss_kd = (conf_weight * kd_each).sum() / (conf_weight.sum() + eps)
    loss_kd = loss_kd * (T * T)

    stats = {
        "kd_conf_mean": conf.mean().detach(),
        "kd_weight_mean": conf_weight.mean().detach(),
        "kd_active_ratio": (conf_weight > 0).float().mean().detach(),
        "kd_agree_ratio": agree_ratio.detach() if isinstance(agree_ratio, torch.Tensor) else agree_ratio,
    }
    return loss_kd, stats


def setup_logging(log_dir, exp_name):
    """设置日志，返回 (logger, timestamp)"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d%H%M')  # 月日时分
    log_file = os.path.join(log_dir, f"{timestamp}_{exp_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__), timestamp


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_feature_dirs(feature_base_dir, modalities, dir_mapping=None):
    """
    根据特征根目录和模态列表构建 feature_dirs 字典。

    Args:
        feature_base_dir: 特征根目录
        modalities: 模态名称列表，如 ['RAW', 'ER', 'PR']
        dir_mapping: 可选的目录名覆盖映射，如 {'HE': 'C17_HE_new_features'}

    Returns:
        dict: {modality_name: feature_directory_path}
    """
    dir_mapping = dir_mapping or {}
    feature_dirs = {}
    for mod in modalities:
        if mod in dir_mapping:
            feature_dirs[mod] = f"{feature_base_dir}/{dir_mapping[mod]}"
        elif mod == 'RAW':
            feature_dirs[mod] = f"{feature_base_dir}/C17_raw_features"
        else:
            feature_dirs[mod] = f"{feature_base_dir}/C17_{mod}_features"
    return feature_dirs


class Trainer:
    """MM-RRT-Light + ABMIL 训练器"""

    def __init__(self, config, logger, timestamp):
        self.config = config
        self.logger = logger
        self.timestamp = timestamp
        self.device = torch.device(
            config['environment']['device'] if torch.cuda.is_available() else 'cpu'
        )

        # 创建输出目录
        os.makedirs(config['output']['save_dir'], exist_ok=True)
        os.makedirs(config['output']['img_dir'], exist_ok=True)

        # 模态信息
        self.modalities = config['data']['modalities']
        self.num_modalities = len(self.modalities)

        # 训练历史记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.val_aucs = []

        # 早停监控
        early_stop_cfg = config['training']['early_stopping']
        self.monitor_metric_name = early_stop_cfg.get('monitor', 'val_auc')
        self.monitor_mode = early_stop_cfg.get('mode', 'max')
        self.best_val_metric = 0.0 if self.monitor_mode == 'max' else float('inf')
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        self.best_val_sensitivity = 0.0
        self.best_val_specificity = 0.0
        self.best_val_precision = 0.0
        self.best_epoch = 0

        # 存储最佳模型验证集的概率和标签（用于ROC曲线）
        self.best_val_probs = None
        self.best_val_labels = None

        # 模型参数信息
        self.model_total_params = 0
        self.model_trainable_params = 0

        # ── 多模态增强配置 ──
        # Modality Dropout: 训练时随机丢弃弱模态，强制模型不依赖单一模态
        self.modality_dropout = config['training'].get('modality_dropout', 0.3)
        # 辅助单模态监督损失权重 (0=禁用)
        self.aux_loss_weight = config['training'].get('aux_loss_weight', 0.1)
        # 辅助分类头 (每个模态一个，在 create_model 中初始化)
        self.aux_classifiers = None

        # ── KD: HE → PR confidence-weighted distillation ──
        self.kd_enabled = config['training'].get('kd_enabled', False)
        self.kd_lambda = config['training'].get('kd_lambda', 0.01)
        self.kd_T = config['training'].get('kd_T', 2.0)
        self.kd_start_epoch = config['training'].get('kd_start_epoch', 5)
        self.kd_ramp_epochs = config['training'].get('kd_ramp_epochs', 10)
        self.kd_conf_threshold = config['training'].get('kd_conf_threshold', 0.5)
        self.kd_he_auc_threshold = config['training'].get('kd_he_auc_threshold', 0.90)
        self.kd_use_agreement_gate = config['training'].get('kd_use_agreement_gate', False)
        self.last_val_auc_he = None  # updated after each validation

        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Modalities ({self.num_modalities}): {self.modalities}")
        if self.num_modalities >= 2:
            self.logger.info(f"Modality dropout: {self.modality_dropout}, aux_loss_weight: {self.aux_loss_weight}")
        if self.kd_enabled:
            self.logger.info(
                f"KD (HE→PR): λ={self.kd_lambda}, T={self.kd_T}, "
                f"start_epoch={self.kd_start_epoch}, ramp={self.kd_ramp_epochs}, "
                f"conf_thresh={self.kd_conf_threshold}, "
                f"he_auc_thresh={self.kd_he_auc_threshold}, "
                f"agree_gate={self.kd_use_agreement_gate}"
            )

    def _get_kd_weight(self):
        """KD warmup + global HE reliability gate.

        Returns 0 if:
        - KD disabled
        - Before warmup start epoch
        - HE teacher AUC below reliability threshold (global gate)
        """
        if not self.kd_enabled:
            return 0.0
        epoch = getattr(self, 'current_epoch', 0)
        if epoch < self.kd_start_epoch:
            return 0.0
        progress = (epoch - self.kd_start_epoch + 1) / max(self.kd_ramp_epochs, 1)
        base_weight = self.kd_lambda * min(1.0, progress)

        # ── Global reliability gate: only allow KD when HE teacher is reliable ──
        he_auc_threshold = getattr(self, 'kd_he_auc_threshold', 0.90)
        last_he_auc = getattr(self, 'last_val_auc_he', None)
        if last_he_auc is not None and last_he_auc < he_auc_threshold:
            return 0.0

        return base_weight

    def _get_patients_from_labels(self, label_file):
        """
        从标签文件中提取所有唯一的 patient_id 及其标签。

        Returns:
            patients: 排序后的患者ID列表
            labels: 对应的标签列表
        """
        df = pd.read_csv(label_file)
        if 'patient_id' in df.columns:
            patients = sorted(df['patient_id'].unique())
            labels = [df[df['patient_id'] == p]['label'].iloc[0] for p in patients]
        else:
            # 从 slide_id 提取 patient_id（如 patient_001_node_1 → patient_001）
            df['patient_id'] = df['slide_id'].apply(
                lambda x: '_'.join(x.split('_')[:2])
            )
            patients = sorted(df['patient_id'].unique())
            labels = [df[df['patient_id'] == p]['label'].iloc[0] for p in patients]

        self.logger.info(f"Total unique patients: {len(patients)}")
        return patients, labels

    def _split_patients(self, patients, labels):
        """
        按 val_start 划分训练集和验证集。

        Returns:
            train_patients, val_patients, train_labels, val_labels
        """
        val_start = self.config['data_split'].get('val_start', 100)

        # 从 patient_XXX 中提取编号
        def _patient_num(p):
            try:
                return int(p.split('_')[-1])
            except (ValueError, IndexError):
                return 0

        train_patients = []
        val_patients = []
        train_labels = []
        val_labels = []

        for p, lbl in zip(patients, labels):
            if _patient_num(p) < val_start:
                train_patients.append(p)
                train_labels.append(lbl)
            else:
                val_patients.append(p)
                val_labels.append(lbl)

        self.logger.info(
            f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}"
        )
        return train_patients, val_patients, train_labels, val_labels

    def create_dataloaders(self):
        """创建训练集和验证集 DataLoader"""
        data_cfg = self.config['data']
        train_cfg = self.config['training']
        env_cfg = self.config['environment']
        dataset_type = data_cfg.get('dataset_type', 'c17')

        # ── C16 模式：独立的 train/test label 文件 ──
        if dataset_type == 'c16':
            from data.c16_multimodal_dataset import (
                C16MultimodalDataset, c16_multimodal_collate_fn,
            )

            feature_dirs = build_feature_dirs(
                data_cfg['feature_base_dir'], data_cfg['modalities'],
                dir_mapping=data_cfg.get('dir_mapping', None)
            )

            train_dataset = C16MultimodalDataset(
                feature_dirs=feature_dirs,
                label_file=data_cfg['train_label_file'],
                max_patches=data_cfg.get('max_patches', 10000),
                preload=data_cfg.get('preload', False),
                verbose=True,
            )
            val_dataset = C16MultimodalDataset(
                feature_dirs=feature_dirs,
                label_file=data_cfg['val_label_file'],
                max_patches=data_cfg.get('max_patches', 10000),
                preload=data_cfg.get('preload', False),
                verbose=True,
            )
            collate_fn = c16_multimodal_collate_fn

            train_loader = DataLoader(
                train_dataset,
                batch_size=train_cfg['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=env_cfg['num_workers'],
                pin_memory=True,
                persistent_workers=(env_cfg['num_workers'] > 0)
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=train_cfg['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=env_cfg['num_workers'],
                pin_memory=True,
                persistent_workers=(env_cfg['num_workers'] > 0)
            )

            self.logger.info(
                f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
            )
            return train_loader, val_loader

        # ── C17 默认模式 ──

        feature_dirs = build_feature_dirs(
            data_cfg['feature_base_dir'], data_cfg['modalities'],
            dir_mapping=data_cfg.get('dir_mapping', None)
        )

        # 获取患者列表并按 val_start 划分
        patients, labels = self._get_patients_from_labels(data_cfg['label_file'])
        train_patients, val_patients, _, _ = self._split_patients(patients, labels)

        # 创建数据集
        train_dataset = MultiModalFeatureDataset(
            feature_dirs=feature_dirs,
            label_file=data_cfg['label_file'],
            patient_list=train_patients,
            max_patches=data_cfg.get('max_patches', 10000),
            preload=data_cfg.get('preload', False),
            verbose=True
        )

        val_dataset = MultiModalFeatureDataset(
            feature_dirs=feature_dirs,
            label_file=data_cfg['label_file'],
            patient_list=val_patients,
            max_patches=data_cfg.get('max_patches', 10000),
            preload=data_cfg.get('preload', False),
            verbose=True
        )

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            collate_fn=multimodal_collate_fn,
            num_workers=env_cfg['num_workers'],
            pin_memory=True,
            persistent_workers=(env_cfg['num_workers'] > 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            collate_fn=multimodal_collate_fn,
            num_workers=env_cfg['num_workers'],
            pin_memory=True,
            persistent_workers=(env_cfg['num_workers'] > 0)
        )

        self.logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        return train_loader, val_loader

    def create_model(self):
        """创建 MM_RRT_ABMIL 模型"""
        model_cfg = self.config['model']
        data_cfg = self.config['data']

        model = MM_RRT_ABMIL(
            # 模态配置
            num_modalities=self.num_modalities,
            modality_list=self.modalities,

            # 特征维度
            input_dim=data_cfg['input_dim'],
            mlp_dim=model_cfg.get('mlp_dim', 512),
            num_classes=data_cfg['num_classes'],

            # 投影层
            dropout=model_cfg.get('dropout', 0.25),

            # RRT Encoder 参数
            region_num=model_cfg.get('region_num', 8),
            n_layers=model_cfg.get('n_layers', 2),
            n_heads=model_cfg.get('n_heads', 8),
            drop_path=model_cfg.get('drop_path', 0.0),
            trans_dropout=model_cfg.get('trans_dropout', 0.1),
            epeg=model_cfg.get('epeg', True),
            epeg_k=model_cfg.get('epeg_k', 15),
            crmsa_k=model_cfg.get('crmsa_k', 3),
            cr_msa=model_cfg.get('cr_msa', True),
            all_shortcut=model_cfg.get('all_shortcut', False),

            # 模态融合参数
            fusion_type=model_cfg.get('fusion_type', 'self_attention'),
            fusion_stage=model_cfg.get('fusion_stage', 'middle'),
            fusion_kwargs={},
            stage2_type=model_cfg.get('stage2_type', 'staining_msa'),
            use_gated_fusion=model_cfg.get('use_gated_fusion', False),
            use_per_layer_fusion=model_cfg.get('use_per_layer_fusion', True),
            use_logit_fusion=model_cfg.get('use_logit_fusion', False),
            fixed_beta=model_cfg.get('fixed_beta', None),
            fixed_alpha=model_cfg.get('fixed_alpha', None),
            use_consistency_fusion=model_cfg.get('use_consistency_fusion', False),
            use_arlc_fusion=model_cfg.get('use_arlc_fusion', False),
            use_correction_only=model_cfg.get('use_correction_only', False),
            use_logit_attn=model_cfg.get('use_logit_attn', False),
            pretrained_he_ckpt=model_cfg.get('pretrained_he_ckpt', None),
            alpha_mode=model_cfg.get('alpha_mode', 'feature'),
            use_lowrank_correction=model_cfg.get('use_lowrank_correction', False),
            use_srp_fusion=model_cfg.get('use_srp_fusion', False),
            srp_beta=model_cfg.get('srp_beta', 0.1),
            srp_mode=model_cfg.get('srp_mode', 'residual'),
            use_shared_rrt=model_cfg.get('use_shared_rrt', False),
            shared_rrt_alpha=model_cfg.get('shared_rrt_alpha', 0.02),
            use_partial_align=model_cfg.get('use_partial_align', False),
            use_mclc=model_cfg.get('use_mclc', False),
            freeze_mclc=model_cfg.get('freeze_mclc', False),
            he_only=model_cfg.get('he_only', False),

            # MIL 参数
            mil_type=model_cfg.get('mil_type', 'abmil'),
            abmil_hidden_dim=model_cfg.get('abmil_hidden_dim', 128),
            use_gated=model_cfg.get('use_gated', False),
        ).to(self.device)

        # 参数统计
        self.model_total_params = sum(p.numel() for p in model.parameters())
        self.model_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # ── 辅助单模态分类头 ──
        # 对多模态场景，每个模态加一个轻量分类头（mean pool → linear），
        # 确保弱模态也被迫学到有用的分类特征
        if self.num_modalities >= 2 and self.aux_loss_weight > 0:
            mlp_dim = model_cfg.get('mlp_dim', 512)
            num_classes = data_cfg['num_classes']
            self.aux_classifiers = nn.ModuleList([
                nn.Linear(mlp_dim, num_classes)
                for _ in range(self.num_modalities)
            ]).to(self.device)
            self.logger.info(f"Aux classifiers: {self.num_modalities} heads")

        self.logger.info(
            f"Model: MM_RRT_ABMIL ({self.num_modalities} modalities, "
            f"fusion={model_cfg.get('fusion_type', 'N/A')}, "
            f"stage={model_cfg.get('fusion_stage', 'N/A')}, "
            f"mil={model_cfg.get('mil_type', 'abmil')})"
        )
        self.logger.info(
            f"Total Params: {self.model_total_params:,}, "
            f"Trainable: {self.model_trainable_params:,}"
        )

        return model

    def create_optimizer_scheduler(self, model):
        """创建优化器和学习率调度器"""
        train_cfg = self.config['training']
        lr = train_cfg['learning_rate']
        wd = train_cfg['weight_decay']

        # Layer-wise LR for correction-only mode
        if getattr(model, 'use_correction_only', False):
            he_encoder, correction, he_classifier, emb = [], [], [], []
            for name, p in model.named_parameters():
                if not p.requires_grad: continue
                if 'rrt_encoder' in name:
                    he_encoder.append(p)
                elif 'correction_nets' in name:
                    correction.append(p)
                elif 'mil' in name or 'classifier' in name:
                    he_classifier.append(p)
                else:
                    emb.append(p)
            param_groups = []
            if he_encoder: param_groups.append({'params': he_encoder, 'lr': lr * 0.1, 'name': 'he_encoder'})
            if he_classifier: param_groups.append({'params': he_classifier, 'lr': lr, 'name': 'he_cls'})
            if correction: param_groups.append({'params': correction, 'lr': lr, 'name': 'correction'})
            if emb: param_groups.append({'params': emb, 'lr': lr * 0.5, 'name': 'embedding'})
            params = param_groups
        else:
            # MCLC logit_mixer gets its own LR
            lr_logit_mixer = train_cfg.get('lr_logit_mixer', lr)
            logit_mixer_params = []
            other_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if 'logit_mixer' in name:
                    logit_mixer_params.append(p)
                else:
                    other_params.append(p)
            if self.aux_classifiers is not None:
                other_params += list(self.aux_classifiers.parameters())
            param_groups = [{'params': other_params, 'lr': lr}]
            if logit_mixer_params:
                param_groups.append({
                    'params': logit_mixer_params,
                    'lr': lr_logit_mixer,
                    'weight_decay': 0.0,
                    'name': 'logit_mixer',
                })
            params = param_groups

        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

        scheduler_cfg = train_cfg.get('scheduler', {})
        if scheduler_cfg.get('type') == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=5, factor=0.5
            )
        elif scheduler_cfg.get('type') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=train_cfg['num_epochs']
            )
        elif scheduler_cfg.get('type') == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_cfg.get('step_size', 30),
                gamma=scheduler_cfg.get('gamma', 0.5)
            )
        else:
            scheduler = None

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, criterion, optimizer, scaler=None):
        """训练一个 epoch（支持 AMP、Modality Dropout、辅助单模态监督）"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        use_aux = self.aux_classifiers is not None and self.aux_loss_weight > 0
        use_mdrop = self.modality_dropout > 0 and self.num_modalities >= 2

        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            features_list_by_modality = batch['features']  # [M, B]
            labels = batch['labels'].to(self.device, non_blocking=True)

            batch_size = len(labels)
            num_modalities = len(features_list_by_modality)

            # ── Modality Dropout: 训练时随机丢弃弱模态，强制模型不依赖单一模态 ──
            if use_mdrop:
                # HE 模态丢弃率减半，弱模态丢弃率高
                drop_probs = [self.modality_dropout * 0.33 if m == 'HE'
                              else self.modality_dropout for m in self.modalities]
                # 保证至少保留 1 个模态
                keep_mask = torch.bernoulli(
                    torch.tensor([1 - p for p in drop_probs])
                ).bool()  # [M]
                if not keep_mask.any():
                    keep_mask[0] = True  # 至少保留第一个模态
                features_list_by_modality = [
                    f if keep_mask[i] else [torch.zeros_like(ff) for ff in f]
                    for i, f in enumerate(features_list_by_modality)
                ]

            # 重新组织为 list of [sample_features_list]
            features_list_by_sample = []
            for i in range(batch_size):
                sample_features = []
                for j in range(num_modalities):
                    sample_features.append(features_list_by_modality[j][i])
                features_list_by_sample.append(sample_features)

            optimizer.zero_grad()
            kd_loss = torch.tensor(0.0, device=self.device)
            kd_stats = {}

            if scaler is not None:
                with autocast():
                    batch_logits = []
                    all_aux_logits = [[] for _ in range(num_modalities)] if use_aux else None
                    logits_he_batch, logits_ihc_lists_batch = [], []
                    for sample_features in features_list_by_sample:
                        sample_features = [f.to(self.device) for f in sample_features]
                        out = model(sample_features, return_features=use_aux)
                        if use_aux:
                            logits = out['logits']
                            emb_list = out['embedded_features']  # list of [1,N,D]
                            for mi in range(num_modalities):
                                aux_feat = emb_list[mi].mean(dim=1)  # [1, D]
                                all_aux_logits[mi].append(self.aux_classifiers[mi](aux_feat))
                            gate_stats = out.get('fusion_stats', None)
                        else:
                            logits = out[0]
                            gate_stats = out[3] if len(out) > 3 else None
                        batch_logits.append(logits)
                        # Collect HE/IHC logits for KD (multi-IHC → average)
                        if isinstance(gate_stats, dict) and 'logits_ihc_list' in gate_stats:
                            logits_he_batch.append(gate_stats['logits_he'].detach())
                            logits_ihc_lists_batch.append(gate_stats['logits_ihc_list'])
                        elif isinstance(gate_stats, dict) and 'logits_pr' in gate_stats:
                            logits_he_batch.append(gate_stats['logits_he'].detach())
                            logits_ihc_lists_batch.append([gate_stats['logits_pr']])
                    batch_logits = torch.cat(batch_logits, dim=0)
                    # Partial alignment aux losses
                    pa_loss = 0.0
                    if gate_stats is not None and isinstance(gate_stats, dict) and 'P_s' in gate_stats:
                        P_s, P_p, H_s, delta = gate_stats['P_s'], gate_stats['P_p'], gate_stats['H_s'], gate_stats['delta']
                        pa_loss = 0.02 * (1.0 - torch.nn.functional.cosine_similarity(P_s, H_s, dim=-1).mean())
                        pa_loss += 0.01 * (torch.nn.functional.normalize(P_s,dim=-1) *
                            torch.nn.functional.normalize(P_p,dim=-1)).sum(-1).pow(2).mean()
                        pa_loss += 0.001 * delta.pow(2).mean()
                    # MCLC identity loss
                id_loss = 0.0
                if gate_stats and isinstance(gate_stats, dict) and 'W' in gate_stats:
                    W = gate_stats['W']
                    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                    id_loss = 0.001 * ((W - I) ** 2).mean()
                loss = criterion(batch_logits, labels) + pa_loss + id_loss
                # ── HE→PR confidence-weighted KD ──
                kd_loss = torch.tensor(0.0, device=self.device)
                kd_stats = {}
                if self.kd_enabled and logits_he_batch:
                    kd_weight = self._get_kd_weight()
                    if kd_weight > 0:
                        all_he = torch.cat(logits_he_batch, dim=0)
                        num_ihc = len(logits_ihc_lists_batch[0]) if logits_ihc_lists_batch else 0
                        kd_losses = []
                        for i in range(num_ihc):
                            all_ihc_i = torch.cat([lst[i] for lst in logits_ihc_lists_batch], dim=0)
                            kd_i, kd_stats_i = he_to_pr_confidence_kd_loss(
                                logits_he=all_he,
                                logits_pr=all_ihc_i,
                                T=self.kd_T,
                                conf_threshold=self.kd_conf_threshold,
                                use_agreement_gate=self.kd_use_agreement_gate,
                            )
                            kd_losses.append(kd_i)
                        kd_loss = sum(kd_losses) / max(num_ihc, 1)  # average, not sum
                        kd_stats = kd_stats_i  # use last IHC's stats for display
                        loss = loss + kd_weight * kd_loss
                    # Consistency-guided KD loss
                fs_amp = None
                if isinstance(out, dict): fs_amp = out.get('fusion_stats')
                elif isinstance(out, tuple) and len(out) > 3: fs_amp = out[3]
                if fs_amp and 'p_ihc' in fs_amp:
                    p_he = torch.softmax(fs_amp['logits_he'], dim=-1)
                    p_teacher = fs_amp['p_ihc'].mean(dim=1)
                    kl = (p_teacher * (p_teacher.log() - p_he.log())).sum(dim=-1).mean()
                    loss = loss + 0.1 * fs_amp['reliability'].mean() * kl
                    # 辅助单模态监督
                    if use_aux:
                        aux_loss = 0.0
                        for mi in range(num_modalities):
                            aux_logits = torch.cat(all_aux_logits[mi], dim=0)
                            aux_loss += criterion(aux_logits, labels)
                        aux_loss = aux_loss / num_modalities
                        loss = loss + self.aux_loss_weight * aux_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_logits = []
                all_aux_logits = [[] for _ in range(num_modalities)] if use_aux else None
                logits_he_batch, logits_ihc_lists_batch = [], []
                for sample_features in features_list_by_sample:
                    sample_features = [f.to(self.device) for f in sample_features]
                    out = model(sample_features, return_features=use_aux)
                    if use_aux:
                        logits = out['logits']
                        emb_list = out['embedded_features']
                        for mi in range(num_modalities):
                            aux_feat = emb_list[mi].mean(dim=1)
                            all_aux_logits[mi].append(self.aux_classifiers[mi](aux_feat))
                        gate_stats = out.get('fusion_stats', None)
                    else:
                        logits = out[0]
                        gate_stats = out[3] if len(out) > 3 else None
                    batch_logits.append(logits)
                    # Collect HE/IHC logits for KD (multi-IHC → average)
                    if isinstance(gate_stats, dict) and 'logits_ihc_list' in gate_stats:
                        logits_he_batch.append(gate_stats['logits_he'].detach())
                        logits_ihc_lists_batch.append(gate_stats['logits_ihc_list'])
                    elif isinstance(gate_stats, dict) and 'logits_pr' in gate_stats:
                        logits_he_batch.append(gate_stats['logits_he'].detach())
                        logits_ihc_lists_batch.append([gate_stats['logits_pr']])
                batch_logits = torch.cat(batch_logits, dim=0)
                # Partial alignment aux losses
                pa_loss = 0.0
                if gate_stats is not None and isinstance(gate_stats, dict) and 'P_s' in gate_stats:
                    P_s, P_p, H_s, delta = gate_stats['P_s'], gate_stats['P_p'], gate_stats['H_s'], gate_stats['delta']
                    pa_loss = 0.02 * (1.0 - torch.nn.functional.cosine_similarity(P_s, H_s, dim=-1).mean())
                    pa_loss += 0.01 * (torch.nn.functional.normalize(P_s,dim=-1) *
                        torch.nn.functional.normalize(P_p,dim=-1)).sum(-1).pow(2).mean()
                    pa_loss += 0.001 * delta.pow(2).mean()
                # MCLC identity loss
                id_loss = 0.0
                if gate_stats and isinstance(gate_stats, dict) and 'W' in gate_stats:
                    W = gate_stats['W']
                    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                    id_loss = 0.001 * ((W - I) ** 2).mean()
                loss = criterion(batch_logits, labels) + pa_loss + id_loss
                # ── HE→PR confidence-weighted KD ──
                kd_loss = torch.tensor(0.0, device=self.device)
                kd_stats = {}
                if self.kd_enabled and logits_he_batch:
                    kd_weight = self._get_kd_weight()
                    if kd_weight > 0:
                        all_he = torch.cat(logits_he_batch, dim=0)
                        num_ihc = len(logits_ihc_lists_batch[0]) if logits_ihc_lists_batch else 0
                        kd_losses = []
                        for i in range(num_ihc):
                            all_ihc_i = torch.cat([lst[i] for lst in logits_ihc_lists_batch], dim=0)
                            kd_i, kd_stats_i = he_to_pr_confidence_kd_loss(
                                logits_he=all_he,
                                logits_pr=all_ihc_i,
                                T=self.kd_T,
                                conf_threshold=self.kd_conf_threshold,
                                use_agreement_gate=self.kd_use_agreement_gate,
                            )
                            kd_losses.append(kd_i)
                        kd_loss = sum(kd_losses) / max(num_ihc, 1)  # average, not sum
                        kd_stats = kd_stats_i  # use last IHC's stats for display
                        loss = loss + kd_weight * kd_loss
                # Consistency-guided KD loss (legacy)
                fs = out[3] if isinstance(out, tuple) and len(out) > 3 else None
                if fs and 'p_ihc' in fs:
                    p_he = torch.softmax(fs['logits_he'], dim=-1)
                    p_teacher = fs['p_ihc'].mean(dim=1)  # [B, 2]
                    kl = (p_teacher * (p_teacher.log() - p_he.log())).sum(dim=-1).mean()
                    loss = loss + 0.1 * fs['reliability'].mean() * kl
                if use_aux:
                    aux_loss = 0.0
                    for mi in range(num_modalities):
                        aux_logits = torch.cat(all_aux_logits[mi], dim=0)
                        aux_loss += criterion(aux_logits, labels)
                    aux_loss = aux_loss / num_modalities
                    loss = loss + self.aux_loss_weight * aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # 指标收集
            with torch.no_grad():
                probs = torch.softmax(batch_logits, dim=1)
                preds = torch.argmax(batch_logits, dim=1)
                all_preds.append(preds)
                all_labels.append(labels)
                all_probs.append(probs)

            if self.kd_enabled and kd_stats:
                pfix = {
                    'loss': f'{loss.item():.4f}',
                    'kd': f'{kd_loss.item():.4f}',
                    'conf': f'{kd_stats["kd_conf_mean"].item():.3f}',
                }
                if 'kd_agree_ratio' in kd_stats:
                    pfix['agree'] = f'{kd_stats["kd_agree_ratio"]:.2f}'
                pbar.set_postfix(pfix)
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        # epoch 结束时一次性 concat + 转 CPU（减少 per-batch 的 GPU-CPU 同步）
        all_labels_np = torch.cat(all_labels).cpu().numpy()
        all_preds_np = torch.cat(all_preds).cpu().numpy()
        all_probs_np = torch.cat(all_probs).cpu().numpy()
        # 显式释放 GPU tensor 列表，防止跨 epoch 内存累积
        del all_labels, all_preds, all_probs
        metrics = calculate_metrics(
            all_labels_np, all_preds_np,
            num_classes=self.config['data']['num_classes'],
            y_prob=all_probs_np
        )

        return avg_loss, metrics

    def validate(self, model, val_loader, criterion, return_probs=False):
        """验证"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        all_he_probs = []   # accumulate for per-path AUC
        all_pr_probs = []   # accumulate for PR-path AUC
        all_ihc_probs = []  # accumulate for per-path AUC (legacy IHC)
        gate_stats = None

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc='Validating'):
                features_list_by_modality = batch['features']
                labels = batch['labels'].to(self.device, non_blocking=True)

                batch_size = len(labels)
                num_modalities = len(features_list_by_modality)

                features_list_by_sample = []
                for i in range(batch_size):
                    sample_features = []
                    for j in range(num_modalities):
                        sample_features.append(features_list_by_modality[j][i])
                    features_list_by_sample.append(sample_features)

                batch_logits = []
                for sample_features in features_list_by_sample:
                    sample_features = [f.to(self.device) for f in sample_features]
                    out = model(sample_features)
                    logits = out[0]
                    gate_stats = out[3] if len(out) > 3 else None  # gate stats from residual fusion
                    aux_loss = out[4] if len(out) > 4 else torch.tensor(0.0, device=logits.device)
                    batch_logits.append(logits)
                    if gate_stats and 'logits_he' in gate_stats:
                        all_he_probs.append(torch.softmax(gate_stats['logits_he'].float(), dim=-1))
                        if 'logits_ihc_list' in gate_stats:
                            for li in gate_stats['logits_ihc_list']:
                                if not hasattr(self, '_ihc_probs_lists'):
                                    self._ihc_probs_lists = [[] for _ in gate_stats['logits_ihc_list']]
                            # Note: per-IHC AUC computed in validation summary
                        if 'logits_pr' in gate_stats:
                            all_pr_probs.append(torch.softmax(gate_stats['logits_pr'].float(), dim=-1))
                        if 'logits_ihc' in gate_stats:
                            all_ihc_probs.append(torch.softmax(gate_stats['logits_ihc'].float(), dim=-1))

                batch_logits = torch.cat(batch_logits, dim=0)
                # Partial alignment aux losses
                pa_loss = 0.0
                if gate_stats is not None and isinstance(gate_stats, dict) and 'P_s' in gate_stats:
                    P_s, P_p, H_s, delta = gate_stats['P_s'], gate_stats['P_p'], gate_stats['H_s'], gate_stats['delta']
                    pa_loss = 0.02 * (1.0 - torch.nn.functional.cosine_similarity(P_s, H_s, dim=-1).mean())
                    pa_loss += 0.01 * (torch.nn.functional.normalize(P_s,dim=-1) *
                        torch.nn.functional.normalize(P_p,dim=-1)).sum(-1).pow(2).mean()
                    pa_loss += 0.001 * delta.pow(2).mean()
                # MCLC identity loss
                id_loss = 0.0
                if gate_stats and isinstance(gate_stats, dict) and 'W' in gate_stats:
                    W = gate_stats['W']
                    I = torch.eye(W.size(0), device=W.device, dtype=W.dtype)
                    id_loss = 0.001 * ((W - I) ** 2).mean()
                loss = criterion(batch_logits, labels) + pa_loss + id_loss

                probs = torch.softmax(batch_logits.float(), dim=1)

                total_loss += loss.item()
                preds = torch.argmax(batch_logits, dim=1)
                all_preds.append(preds)
                all_labels.append(labels)
                all_probs.append(probs)

        # Log fusion statistics
        if 'gate_stats' in dir() and gate_stats is not None:
            if 'W' in gate_stats:
                W = gate_stats['W'].cpu()
                self.logger.info(f"MCLC W: HE→HE={W[0,0]:.3f} HE→PR={W[0,1]:.3f}")
            elif 'P_norm' in gate_stats:
                g = gate_stats.get('gate_mean', 0)
                d = gate_stats.get('delta_norm', 0)
                self.logger.info(f"SRP: |P|={gate_stats['P_norm']:.1f} g={g:.3f} |Δ|={d:.1f}")
            elif 'weight_mean' in gate_stats:
                w = gate_stats['weight_mean'].numpy()
                w_str = ' '.join(f'{x:.3f}' for x in w)
                self.logger.info(f"Attn: [{w_str}] T={gate_stats['temperature']:.2f}")
            elif 'r' in gate_stats:
                self.logger.info(f"ARLC: r={gate_stats['r']:.4f} δ={gate_stats['delta']:.4f}")
            elif 'n_ihc' in gate_stats:
                self.logger.info(f"Multi-Corr ({gate_stats['n_ihc']}IHC): δ={gate_stats['delta']:.4f}")
            elif 'delta' in gate_stats and 'r' not in gate_stats:
                self.logger.info(f"Correction: δ={gate_stats['delta']:.4f}")
            elif 'beta' in gate_stats:
                self.logger.info(f"Fusion: β={gate_stats['beta']:.4f} (HE weight)")
                if all_he_probs:
                    from sklearn.metrics import roc_auc_score
                    labels_np = torch.cat(all_labels).cpu().numpy()
                    he_probs = torch.cat(all_he_probs, dim=0)[:,1].cpu().numpy()
                    ihc_probs = torch.cat(all_ihc_probs, dim=0)[:,1].cpu().numpy()
                    try:
                        auc_he = roc_auc_score(labels_np, he_probs)
                        auc_ihc = roc_auc_score(labels_np, ihc_probs)
                        self.logger.info(f"  AUC_HE={auc_he:.4f}  AUC_IHC={auc_ihc:.4f}  Δ={auc_he-auc_ihc:+.4f}")
                    except Exception as e:
                        self.logger.info(f"  AUC per-path error: {e}")
            elif 'orth_norm' in gate_stats:
                self.logger.info(
                    f"Orth: α={gate_stats.get('alpha',0):.4f} "
                    f"|orth|={gate_stats.get('orth_norm',0):.3f} "
                    f"|par|={gate_stats.get('parallel_norm',0):.3f} "
                    f"cos(he,orth)={gate_stats.get('cos_after',0):.4f}"
                )
            else:
                self.logger.info(
                    f"Fusion: α={gate_stats.get('alpha',0):.4f} "
                    f"delta_norm={gate_stats.get('delta_norm',0):.3f} "
                    f"att_norm={gate_stats.get('att_norm',0):.3f}"
                )

        avg_loss = total_loss / len(val_loader)
        all_labels_np = torch.cat(all_labels).cpu().numpy()
        all_probs_np = torch.cat(all_probs).cpu().numpy()
        all_preds_np = torch.cat(all_preds).cpu().numpy()
        # 显式释放 GPU tensor 列表，防止跨 epoch 内存累积
        del all_labels, all_preds, all_probs
        metrics = calculate_metrics(
            all_labels_np, all_preds_np,
            num_classes=self.config['data']['num_classes'],
            y_prob=all_probs_np
        )

        # ── Per-path AUC (HE / PR) ──
        if all_he_probs:
            try:
                from sklearn.metrics import roc_auc_score
                he_probs_np = torch.cat(all_he_probs, dim=0)[:, 1].cpu().numpy()
                metrics['auc_he'] = float(roc_auc_score(all_labels_np, he_probs_np))
            except Exception:
                metrics['auc_he'] = 0.0
        if all_pr_probs:
            try:
                from sklearn.metrics import roc_auc_score
                pr_probs_np = torch.cat(all_pr_probs, dim=0)[:, 1].cpu().numpy()
                metrics['auc_pr'] = float(roc_auc_score(all_labels_np, pr_probs_np))
            except Exception:
                metrics['auc_pr'] = 0.0
        del all_he_probs, all_pr_probs, all_ihc_probs

        if return_probs:
            return avg_loss, metrics, all_probs_np, all_labels_np
        return avg_loss, metrics

    def plot_training_curves(self):
        """
        绘制训练曲线，包含 Loss、Metrics、ROC 曲线和摘要信息。
        布局: 2x2
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # ===== 1. Loss 曲线 =====
        ax1 = axes[0, 0]
        ax1.plot(self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        if len(self.val_losses) > 0:
            best_val_epoch = np.argmin(self.val_losses)
            ax1.axvline(x=best_val_epoch, color='g', linestyle='--', alpha=0.5,
                       label=f'Best Val Loss @ Epoch {best_val_epoch}')
            ax1.scatter([best_val_epoch], [self.val_losses[best_val_epoch]],
                       color='green', s=100, zorder=5)

        # ===== 2. Accuracy / AUC 曲线 =====
        ax2 = axes[0, 1]
        ax2.plot(self.train_accs, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(self.val_accs, 'r-', label='Val Acc', linewidth=2)
        ax2.plot(self.val_aucs, 'g-', label='Val AUC', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Training and Validation Metrics', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])

        if len(self.val_aucs) > 0:
            best_auc_epoch = np.argmax(self.val_aucs)
            ax2.axvline(x=best_auc_epoch, color='purple', linestyle='--', alpha=0.5)
            ax2.scatter([best_auc_epoch], [self.val_aucs[best_auc_epoch]],
                       color='purple', s=100, zorder=5)

        # ===== 3. ROC 曲线 =====
        ax3 = axes[1, 0]
        num_classes = self.config['data']['num_classes']

        if self.best_val_probs is not None and self.best_val_labels is not None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            if num_classes == 2:
                fpr, tpr, _ = roc_curve(
                    self.best_val_labels, self.best_val_probs[:, 1]
                )
                roc_auc = auc(fpr, tpr)
                ax3.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC (AUC = {roc_auc:.4f})')
            else:
                for i in range(num_classes):
                    y_true_bin = (self.best_val_labels == i).astype(int)
                    fpr, tpr, _ = roc_curve(
                        y_true_bin, self.best_val_probs[:, i]
                    )
                    roc_auc = auc(fpr, tpr)
                    ax3.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                            label=f'Class {i} (AUC = {roc_auc:.4f})')

            ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate', fontsize=12)
            ax3.set_ylabel('True Positive Rate', fontsize=12)
            ax3.set_title('ROC Curves (Best Model)', fontsize=14)
            ax3.legend(fontsize=9, loc='lower right')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'ROC curve not available',
                    ha='center', va='center', fontsize=12,
                    transform=ax3.transAxes)
            ax3.set_title('ROC Curves', fontsize=14)

        # ===== 4. 摘要信息 =====
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = (
            f"Best Model Summary\n"
            f"{'='*20}\n"
            f"Best Epoch: {self.best_epoch + 1}\n"
            f"Val AUC: {self.best_val_auc:.4f}\n"
            f"Val Acc: {self.best_val_acc:.4f}\n"
            f"Val F1: {self.best_val_f1:.4f}\n"
            f"Val Sensitivity: {self.best_val_sensitivity:.4f}\n"
            f"Val Specificity: {self.best_val_specificity:.4f}\n\n"
            f"Model Info\n"
            f"{'='*20}\n"
            f"Modalities: {self.num_modalities}\n"
            f"Params: {self.model_total_params:,}\n"
            f"Trainable: {self.model_trainable_params:,}"
        )
        ax4.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 底部超参数文本框
        hyperparams_text = self._format_hyperparams()
        fig.text(0.5, 0.01, hyperparams_text, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        img_path = os.path.join(
            self.config['output']['img_dir'],
            f'training_curves_{self.timestamp}.png'
        )
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Saved training curves to {img_path}")
        plt.close()

    def _format_hyperparams(self):
        """格式化超参数字符串用于图表标注"""
        cfg = self.config
        mod_str = '+'.join(cfg['data']['modalities'])
        return (
            f"Modalities: {mod_str} | "
            f"Fusion: {cfg['model'].get('fusion_type', 'N/A')} | "
            f"Stage: {cfg['model'].get('fusion_stage', 'N/A')} | "
            f"MLP-Dim: {cfg['model'].get('mlp_dim', 512)} | "
            f"Dropout: {cfg['model'].get('dropout', 0.25)} | "
            f"LR: {cfg['training']['learning_rate']} | "
            f"WD: {cfg['training']['weight_decay']} | "
            f"Epochs: {cfg['training']['num_epochs']}"
        )

    def train(self, optuna_trial=None):
        """主训练循环"""
        set_seed(self.config['environment']['seed'])

        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders()

        # 创建模型
        model = self.create_model()

        # 创建优化器和调度器
        optimizer, scheduler = self.create_optimizer_scheduler(model)

        # 混合精度
        use_amp = self.config['training'].get('use_amp', False)
        scaler = GradScaler() if use_amp else None

        # 损失函数 — Auto class weights + FocalLoss 支持
        num_classes = self.config['data']['num_classes']
        from collections import Counter
        train_labels = [s['label'] for s in train_loader.dataset.samples]
        label_counts = Counter(train_labels)
        n_total = len(train_labels)
        weights = []
        for c in range(num_classes):
            count_c = label_counts.get(c, 1)
            weights.append(n_total / (num_classes * count_c))
        class_weights = torch.tensor(weights).to(self.device)
        self.logger.info(f"Auto class_weights (inverse freq): {weights}")

        use_focal = self.config['training'].get('focal_loss', False)
        focal_gamma = self.config['training'].get('focal_gamma', 2.0)
        label_smoothing = self.config['training'].get('label_smoothing', 0.0)
        if use_focal:
            criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            self.logger.info(f"Using FocalLoss (gamma={focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            if label_smoothing > 0:
                self.logger.info(f"Using CrossEntropyLoss with label_smoothing={label_smoothing}")
            else:
                self.logger.info("Using CrossEntropyLoss with auto class_weights")

        # 早停配置
        early_stop_patience = self.config['training']['early_stopping']['patience']
        early_stop_counter = 0
        num_epochs = self.config['training']['num_epochs']

        self.logger.info("=" * 60)
        self.logger.info("Starting Training...")
        self.logger.info(f"Total Epochs: {num_epochs}")
        self.logger.info(f"Modalities: {self.modalities}")
        self.logger.info(f"Fusion: {self.config['model'].get('fusion_type', 'N/A')}")
        self.logger.info("=" * 60)

        # Warmup for correction: α=0 for first N epochs, then α=0.1
        warmup_epochs = self.config['training'].get('correction_warmup', 5)
        correction_alpha_final = self.config['training'].get('correction_alpha', 0.1)

        for epoch in range(num_epochs):
            if hasattr(model, 'correction_alpha'):
                if warmup_epochs == 0:
                    model.correction_alpha = correction_alpha_final
                elif epoch < warmup_epochs:
                    model.correction_alpha = 0.0
                else:
                    model.correction_alpha = correction_alpha_final

            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.logger.info("-" * 40)

            # 训练
            self.current_epoch = epoch
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(
                model, train_loader, criterion, optimizer, scaler=scaler
            )
            t_train = time.time() - t0
            train_acc = train_metrics['accuracy']
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
                f"[{t_train:.1f}s]"
            )

            # 验证
            t0 = time.time()
            val_loss, val_metrics = self.validate(
                model, val_loader, criterion
            )
            t_val = time.time() - t0
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            val_auc = val_metrics.get('auc', 0.0)
            val_auc_he = val_metrics.get('auc_he', None)
            val_auc_pr = val_metrics.get('auc_pr', None)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_aucs.append(val_auc)
            # ── Update HE reliability for KD gate ──
            self.last_val_auc_he = val_auc_he
            extra = ""
            if val_auc_he is not None:
                extra += f"AUC_HE={val_auc_he:.4f} "
            if val_auc_pr is not None:
                extra += f"AUC_PR={val_auc_pr:.4f} "
            kd_w = self._get_kd_weight() if self.kd_enabled else 0.0
            if self.kd_enabled:
                extra += f"kd_w={kd_w:.4f} "
            self.logger.info(
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
                f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} "
                f"[{t_val:.1f}s]" + (f" {extra}" if extra else "")
            )

            # 学习率调度
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_auc)
                else:
                    scheduler.step()

            # 保存最佳模型
            if self.monitor_metric_name == 'val_acc':
                monitor_metric = val_acc
            elif self.monitor_metric_name == 'val_f1':
                monitor_metric = val_f1
            elif self.monitor_metric_name == 'val_auc':
                monitor_metric = val_auc
            elif self.monitor_metric_name == 'val_loss':
                monitor_metric = val_loss
            else:
                monitor_metric = val_auc

            is_best = (
                (self.monitor_mode == 'max' and monitor_metric > self.best_val_metric)
                or (self.monitor_mode == 'min' and monitor_metric < self.best_val_metric)
            )
            if is_best:
                self.best_val_metric = monitor_metric
                self.best_val_acc = val_acc
                self.best_val_auc = val_metrics.get('auc', 0.0)
                self.best_val_f1 = val_f1
                self.best_val_sensitivity = val_metrics.get('sensitivity_macro', 0.0)
                self.best_val_specificity = val_metrics.get('specificity_macro', 0.0)
                self.best_val_precision = val_metrics.get('precision', 0.0)
                self.best_epoch = epoch
                early_stop_counter = 0

                save_path = os.path.join(
                    self.config['output']['save_dir'], 'best_model.pt'
                )
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': self.config,
                    'best_val_metric': self.best_val_metric,
                    'best_val_acc': self.best_val_acc,
                    'best_val_auc': self.best_val_auc,
                    'best_val_f1': self.best_val_f1,
                    'best_val_sensitivity': self.best_val_sensitivity,
                    'best_val_specificity': self.best_val_specificity,
                    'best_val_precision': self.best_val_precision,
                    'modalities': self.modalities,
                    'num_modalities': self.num_modalities,
                    'aux_classifiers_state': self.aux_classifiers.state_dict() if self.aux_classifiers is not None else None,
                }
                torch.save(checkpoint, save_path)
                self.logger.info(
                    f"Saved best model "
                    f"({self.monitor_metric_name}: {self.best_val_metric:.4f})"
                )
            else:
                early_stop_counter += 1

            # --- Optuna prune check ---
            if optuna_trial is not None:
                optuna_trial.report(monitor_metric, epoch)
                if optuna_trial.should_prune():
                    self.logger.info(f"Trial pruned at epoch {epoch + 1}")
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            # 早停检查
            if early_stop_counter >= early_stop_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break

            # 每个 epoch 结束后强制内存回收，防止跨 epoch 累积
            gc.collect()
            torch.cuda.empty_cache()

        # 训练结束 — 加载最佳模型做最终评估
        best_model_path = os.path.join(
            self.config['output']['save_dir'], 'best_model.pt'
        )
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if (self.aux_classifiers is not None
                    and checkpoint.get('aux_classifiers_state') is not None):
                self.aux_classifiers.load_state_dict(checkpoint['aux_classifiers_state'])

            _, val_metrics, best_probs, best_labels = self.validate(
                model, val_loader, criterion, return_probs=True
            )
            self.best_val_probs = best_probs
            self.best_val_labels = best_labels
            self.best_val_auc = val_metrics.get('auc', 0.0)

            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            val_auc = val_metrics.get('auc', float('nan'))
            val_sensitivity = val_metrics.get(
                'sensitivity_macro', val_metrics.get('recall', 0.0)
            )
            val_specificity = val_metrics.get('specificity_macro', 0.0)
            self.best_val_precision = val_metrics.get('precision', 0.0)

            num_classes = self.config['data']['num_classes']
            per_class_lines = []
            for i in range(num_classes):
                sens = val_metrics.get(f'sensitivity_class_{i}', 0.0)
                spec = val_metrics.get(f'specificity_class_{i}', 0.0)
                prec = val_metrics.get(f'precision_class_{i}', 0.0)
                per_class_lines.append(
                    f"  Class {i}: Sens={sens:.4f}, Spec={spec:.4f}, Prec={prec:.4f}"
                )

            per_class_str = "\n".join(per_class_lines)
            cm = val_metrics['confusion_matrix']
            class_names = [f"Class_{i}" for i in range(num_classes)]

            self.logger.info("\n" + "=" * 60)
            self.logger.info("Training Completed!")
            self.logger.info(f"Best Epoch: {self.best_epoch + 1}")
            self.logger.info(
                f"Best {self.monitor_metric_name}: {self.best_val_metric:.4f}"
            )
            self.logger.info(
                f"Best Val Acc: {val_acc:.4f}  AUC: {val_auc:.4f}"
            )
            self.logger.info(
                f"Sensitivity: {val_sensitivity:.4f}  "
                f"Specificity: {val_specificity:.4f}"
            )
            self.logger.info(f"Per-Class:\n{per_class_str}")
            self.logger.info(
                f"Confusion Matrix:\n{format_confusion_matrix(cm, class_names)}"
            )
            self.logger.info(
                f"Model Params: {self.model_total_params:,}"
            )
            self.logger.info("=" * 60)
        else:
            self.logger.info("\n" + "=" * 60)
            self.logger.info("Training Completed! (no best model saved)")
            self.logger.info(
                f"Best {self.monitor_metric_name}: {self.best_val_metric:.4f} "
                f"at Epoch {self.best_epoch + 1}"
            )
            self.logger.info("=" * 60)

        # 保存训练曲线
        self.plot_training_curves()
        self.logger.info("Done.")
        return model, self.best_val_metric


def main():
    parser = argparse.ArgumentParser(
        description='MM-RRT-Light + ABMIL Training'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--exp_name', type=str, default='mmrrt_abmil',
        help='Experiment name (used for log file naming)'
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    logger, timestamp = setup_logging(
        config['output']['log_dir'], args.exp_name
    )
    logger.info(f"Configuration: {config}")
    logger.info(f"Modalities: {config['data']['modalities']}")

    # 创建训练器并训练
    trainer = Trainer(config, logger, timestamp)
    model, best_metric = trainer.train()

    logger.info(f"Final Best Metric: {best_metric:.4f}")


if __name__ == '__main__':
    main()
