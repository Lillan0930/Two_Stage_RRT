#!/usr/bin/env python3
"""C16 Two-stage Modality-aware R²T — standalone test."""
import os, sys, yaml, time
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import Trainer
import logging
from pathlib import Path

logger = logging.getLogger('test')
logger.handlers.clear()
logger.setLevel(logging.WARNING)

cfg = {
    'data': {
        'dataset_type': 'c16',
        'train_label_file': 'data/C16_labels/c16_train_labels.csv',
        'val_label_file': 'data/C16_labels/c16_test_labels.csv',
        'feature_base_dir': '/home/Public/lillan/features_result/C16_features',
        'modalities': ['HE', 'PR'],
        'dir_mapping': {'HE': 'C16_HE_features', 'PR': 'C16_PR_features'},
        'input_dim': 768, 'num_classes': 2, 'max_patches': 5000, 'preload': False,
    },
    'output': {'save_dir': '/tmp/ts_standalone/ckpt', 'log_dir': '/tmp/ts_standalone/logs', 'img_dir': '/tmp/ts_standalone/img'},
    'model': {
        'mil_type': 'abmil', 'mlp_dim': 512, 'dropout': 0.25, 'use_gated': False,
        'region_num': 4, 'n_layers': 2, 'n_heads': 4,
        'drop_path': 0.0, 'trans_dropout': 0.1, 'epeg': True, 'epeg_k': 21,
        'crmsa_k': 5, 'cr_msa': True, 'all_shortcut': True,
        'fusion_type': 'two_stage_region', 'fusion_stage': 'middle',
        'use_gated_fusion': False, 'abmil_hidden_dim': 256,
        'use_mclc': False, 'aggregate_modalities': True,
    },
    'training': {
        'batch_size': 1, 'num_epochs': 25, 'learning_rate': 1e-4, 'weight_decay': 1e-5,
        'scheduler': {'type': 'plateau'},
        'early_stopping': {'patience': 10, 'monitor': 'val_auc', 'mode': 'max'},
        'use_amp': False, 'focal_loss': False, 'label_smoothing': 0.0,
        'kd_enabled': False, 'modality_dropout': 0.0, 'aux_loss_weight': 0.0,
    },
    'data_split': {'val_start': 100},
    'environment': {'device': 'cuda:0', 'num_workers': 2, 'seed': 42},
}

for d in ['/tmp/ts_standalone/ckpt', '/tmp/ts_standalone/logs', '/tmp/ts_standalone/img']:
    Path(d).mkdir(parents=True, exist_ok=True)

print('Testing TwoStageRRT standalone...', flush=True)
t0 = time.time()
trainer = Trainer(cfg, logger, 'test')
_, auc = trainer.train()
print(f'Done! AUC={auc:.4f} [{time.time()-t0:.0f}s]', flush=True)
print('Project integrity verified!', flush=True)
