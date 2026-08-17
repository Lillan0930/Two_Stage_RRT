#!/usr/bin/env python3
"""
C16 单模态 R²T+ABMIL Optuna worker（通用，按 --task 选模态）。

协议（对齐 Base_mil）：
  train = c16_train_labels.csv (270)  val = c16_test_labels.csv (129)  # test-as-val
  objective = best_val_auc；early_stopping val_auc/max/patience 10；num_epochs=80/cosine；seed 42。

搜索空间（6 参数，结构/损失已由 PR 网格确定后固定）：
  n_heads {2,4,8} | abmil_hidden_dim {128,256,384} | drop_path [0,0.3]
  lr log[3e-5,3e-4] | wd log[1e-6,1e-4] | label_smoothing [0,0.2]

用法（由 run_c16_rrt_optuna.py 启动）：
  CUDA_VISIBLE_DEVICES=0 python scripts/_run_c16_rrt_optuna_worker.py \
      --task er --study er_rrt_abmil_optuna --storage sqlite:///.../er.db --n-trials 15
"""
import os, sys, json, shutil, logging, argparse
from pathlib import Path
from copy import deepcopy

import optuna
from optuna.storages import RDBStorage

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
SCRATCH_ROOT = PROJECT / "results" / "c16_rrt_optuna"

TASKS = {
    "pr":   {"modality": "PR",   "feature_dir": "C16_PR_features"},
    "er":   {"modality": "ER",   "feature_dir": "C16_ER_features"},
    "her2": {"modality": "HER2", "feature_dir": "C16_HER2_features"},
    "ki67": {"modality": "Ki67", "feature_dir": "C16_Ki67_features"},
}


def _base_config(task):
    mod = TASKS[task]["modality"]
    feat = TASKS[task]["feature_dir"]
    return {
        "data": {
            "dataset_type": "c16", "modalities": [mod],
            "dir_mapping": {mod: feat},
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "val_label_file": str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": 42, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 8, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 15,
            "crmsa_k": 5, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4, "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False, "focal_gamma": 2.0,
            "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": 42},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }


def _suggest(trial, cfg):
    cfg["model"]["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
    cfg["model"]["abmil_hidden_dim"] = trial.suggest_categorical(
        "abmil_hidden_dim", [128, 256, 384])
    cfg["model"]["drop_path"] = trial.suggest_float("drop_path", 0.0, 0.3)
    cfg["training"]["learning_rate"] = trial.suggest_float(
        "lr", 3e-5, 3e-4, log=True)
    cfg["training"]["weight_decay"] = trial.suggest_float(
        "wd", 1e-6, 1e-4, log=True)
    cfg["training"]["label_smoothing"] = trial.suggest_float(
        "label_smoothing", 0.0, 0.2)
    return cfg


def objective(trial):
    from train import Trainer

    cfg = _suggest(trial, deepcopy(BASE_CONFIG))
    trial_dir = SCRATCH_ROOT / TASK / f"trial_{trial.number}"
    for d in ["ckpt", "logs", "img"]:
        (trial_dir / d).mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_dir"] = str(trial_dir / "ckpt")
    cfg["output"]["log_dir"] = str(trial_dir / "logs")
    cfg["output"]["img_dir"] = str(trial_dir / "img")

    logger = logging.getLogger(f"trial_{trial.number}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(trial_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    hp = ", ".join(f"{k}={v}" for k, v in trial.params.items())
    logger.info(f"Trial {trial.number} | {hp}")

    try:
        trainer = Trainer(cfg, logger, f"trial{trial.number}")
        trainer.train(optuna_trial=trial)
        auc = float(trainer.best_val_auc)
        trial.set_user_attr("val_acc", float(trainer.best_val_acc))
        trial.set_user_attr("val_auc", auc)
        trial.set_user_attr("val_sensitivity", float(trainer.best_val_sensitivity))
        trial.set_user_attr("val_specificity", float(trainer.best_val_specificity))
        trial.set_user_attr("best_epoch", int(trainer.best_epoch))
        logger.info(f"Trial {trial.number} done: val_auc={auc:.4f}")
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} pruned")
        raise
    finally:
        shutil.rmtree(trial_dir / "ckpt", ignore_errors=True)
        shutil.rmtree(trial_dir / "img", ignore_errors=True)

    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--study", required=True)
    ap.add_argument("--storage", required=True)
    ap.add_argument("--n-trials", type=int, required=True)
    args = ap.parse_args()

    global TASK, BASE_CONFIG
    TASK = args.task
    BASE_CONFIG = _base_config(TASK)

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU {gpu}] worker start task={TASK} study={args.study}", flush=True)

    storage = RDBStorage(args.storage, engine_kwargs={"connect_args": {"timeout": 60}})
    study = optuna.load_study(study_name=args.study, storage=storage)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print(f"[GPU {gpu}] worker done task={TASK}", flush=True)


if __name__ == "__main__":
    main()
