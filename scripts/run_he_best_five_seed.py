#!/usr/bin/env python3
"""HE-RRT-only 270/129 × 5 seeds 基线（对齐 two-stage 的比较目标）。

用 HE-RRT-only 的 best 配置（region_num=4 / epeg_k=9 / crmsa_k=3 / n_heads=4 / drop_path=0.0，
来自 results/c16_official_train_test/he_rrt/seed42/config.json），但切到与 two-stage 完全一致的
270/129 test-as-val 协议（early-stop val_auc/max/patience 10，num_epochs=80，cosine，sampling random）。

跑 5 个 seed（42/123/456/789/1024），汇报 test-as-val AUC 的 mean±std，作为 two-stage 要超过的基线。

用法:
  python scripts/run_he_best_five_seed.py --gpu 2
"""
import os, sys, json, logging, shutil, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

# 先解析 --gpu，在 import train 前设置 CUDA_VISIBLE_DEVICES（避免 CUDA 提前初始化）
_pre_ap = argparse.ArgumentParser(add_help=False)
_pre_ap.add_argument("--gpu", type=int, required=True)
_pre_gpu, _ = _pre_ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_gpu.gpu)

from train import Trainer  # noqa: E402

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
SEEDS = [42, 123, 456, 789, 1024]
OUT_ROOT = PROJECT / "results" / "HE_rrt_abmil"


def build_config(seed: int):
    return {
        "data": {
            "dataset_type": "c16", "modalities": ["HE"],
            "dir_mapping": {"HE": "C16_HE_features"},
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "val_label_file": str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4, "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False,
            "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }


def run_seed(seed: int):
    out_dir = OUT_ROOT / f"seed{seed}"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg = build_config(seed)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger = logging.getLogger(f"he_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"HE-RRT-only 270/129 baseline | seed={seed}")

    trainer = Trainer(cfg, logger, f"seed{seed}")
    trainer.train()

    result = {
        "seed": seed,
        "val_auc": float(trainer.best_val_auc),
        "val_acc": float(trainer.best_val_acc),
        "val_sensitivity": float(trainer.best_val_sensitivity),
        "val_specificity": float(trainer.best_val_specificity),
        "best_epoch": int(trainer.best_epoch),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[HE seed {seed}] val_auc={result['val_auc']:.4f} "
          f"acc={result['val_acc']:.4f} sens={result['val_sensitivity']:.4f} "
          f"spec={result['val_specificity']:.4f} epoch={result['best_epoch']}",
          flush=True)

    shutil.rmtree(out_dir / "ckpt", ignore_errors=True)
    shutil.rmtree(out_dir / "img", ignore_errors=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    print(f"HE-RRT-only 270/129 baseline (GPU {args.gpu}), seeds={SEEDS}", flush=True)

    results = [run_seed(s) for s in SEEDS]

    aucs = [r["val_auc"] for r in results]
    accs = [r["val_acc"] for r in results]
    sens = [r["val_sensitivity"] for r in results]
    specs = [r["val_specificity"] for r in results]

    print("\n" + "=" * 64, flush=True)
    print("SUMMARY HE-RRT-only (270/129 test-as-val)", flush=True)
    for r in results:
        print(f"  seed={r['seed']:>4}: AUC={r['val_auc']:.4f} acc={r['val_acc']:.4f} "
              f"sens={r['val_sensitivity']:.4f} spec={r['val_specificity']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
    print(f"  Sens: {np.mean(sens):.4f} ± {np.std(sens):.4f}", flush=True)

    summary = {
        "task": "he_rrt_abmil",
        "modality": "HE",
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc",
                     "mode": "max", "patience": 10, "num_epochs": 80,
                     "scheduler": "cosine", "sampling": "random"},
        "seeds": SEEDS,
        "per_seed": results,
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_sensitivity": float(np.mean(sens)),
        "std_sensitivity": float(np.std(sens)),
        "mean_specificity": float(np.mean(specs)),
        "std_specificity": float(np.std(specs)),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_ROOT / 'summary.json'}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
