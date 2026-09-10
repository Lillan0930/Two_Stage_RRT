#!/usr/bin/env python3
"""训练协议控制实验 单 seed worker（训练 + 后处理）。

读取一个完整 config.json（含 modalities / encoder_cfg / stage2_cfg / seed /
learning_rate / 可选 lr_stage1+lr_stage2），在指定 GPU 上训练，随后写产物：

  - result.json        : auc / accuracy / f1 / sensitivity_tumor /
                         specificity_tumor / sensitivity_macro /
                         specificity_macro / best_epoch / checkpoint /
                         actual_optimizer_lrs / train_persistent_workers /
                         initialization
  - test_predictions.csv : 129 test 的 slide_id,label,prob_pos,pred
  - protocol_check.json  : 初始化 / LR 实际值 协议记录

支持单模态（HE-only）与双模态（two-stage）config。

用法:
  python scripts/_run_protocol_seed.py --config <path/config.json> --gpu 3
"""
import os, sys, json, logging, argparse
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

ACTIVE_MODULES = {
    "he_projection": "patch_to_emb.0.",
    "pr_projection": "patch_to_emb.1.",
    "he_rrt": "rrt_he.",
    "pr_rrt": "rrt_ihc.",
    "stage2": "cross_region_mod.",
    "abmil": "mil.",
}


def write_test_predictions(cfg, trainer, out_dir, build_feature_dirs, C16MultimodalDataset):
    """用 trainer.best_val_probs 写 129 test 的 test_predictions.csv。"""
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(
        data_cfg["feature_base_dir"], data_cfg["modalities"],
        data_cfg.get("dir_mapping", None))
    val_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=data_cfg["val_label_file"],
        max_patches=data_cfg.get("max_patches", 2500), preload=False, verbose=False,
        sampling=data_cfg.get("sampling", "random"),
        sample_seed=data_cfg.get("sample_seed", 0), per_epoch=False,
    )
    slide_ids = [s["slide_id"] for s in val_ds.samples]
    probs = trainer.best_val_probs
    labels = trainer.best_val_labels
    assert probs is not None, "best_val_probs is None (best checkpoint not evaluated)"
    assert len(slide_ids) == len(probs), \
        f"slide/prob length mismatch: {len(slide_ids)} vs {len(probs)}"
    lines = ["slide_id,label,prob_pos,pred"]
    for i, sid in enumerate(slide_ids):
        p1 = float(probs[i, 1])
        lines.append(f"{sid},{int(labels[i])},{p1:.6f},{int(p1 >= 0.5)}")
    (out_dir / "test_predictions.csv").write_text("\n".join(lines) + "\n")


def compute_actual_lrs(model, trainer, cfg):
    """用同一 config 重建 optimizer，映射每个活跃模块 → 实际 LR。"""
    optimizer, _ = trainer.create_optimizer_scheduler(model)
    id2lr = {}
    for g in optimizer.param_groups:
        for p in g["params"]:
            id2lr[id(p)] = g["lr"]
    lrs = {}
    for mname, prefix in ACTIVE_MODULES.items():
        for name, p in model.named_parameters():
            if name.startswith(prefix):
                lrs[mname] = float(id2lr[id(p)])
                break
    return lrs


def compute_tumor_metrics(trainer):
    """从 best_val_probs/labels 计算 class_1(tumor) 与 macro Sens/Spec。"""
    from utils.metrics import calculate_metrics
    probs = trainer.best_val_probs
    labels = trainer.best_val_labels
    preds = probs.argmax(axis=1)
    m = calculate_metrics(labels, preds, num_classes=2, y_prob=probs)
    return {
        "sensitivity_tumor": float(m["sensitivity_class_1"]),
        "specificity_tumor": float(m["specificity_class_1"]),
        "sensitivity_macro": float(m["sensitivity_macro"]),
        "specificity_macro": float(m["specificity_macro"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from train import Trainer, build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out_dir = cfg_path.parent
    seed = cfg["environment"]["seed"]

    logger = logging.getLogger(f"protocol_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"protocol seed={seed} modalities={cfg['data']['modalities']} "
                f"training={ {k: cfg['training'].get(k) for k in ['learning_rate','lr_stage1','lr_stage2']} }")

    trainer = Trainer(cfg, logger, f"s{seed}")
    model, _best_metric = trainer.train()

    # 实际 optimizer LR（按模块）
    actual_lrs = compute_actual_lrs(model, trainer, cfg)

    # 临床 Sens/Spec = class_1(tumor)
    tumor_metrics = compute_tumor_metrics(trainer)

    train_persistent_workers = bool(getattr(trainer, "_train_persistent_workers", False))
    initialization = "random_from_scratch"

    result = {
        "seed": seed,
        "auc": float(trainer.best_val_auc),
        "accuracy": float(trainer.best_val_acc),
        "f1": float(trainer.best_val_f1),
        "sensitivity_tumor": tumor_metrics["sensitivity_tumor"],
        "specificity_tumor": tumor_metrics["specificity_tumor"],
        "sensitivity_macro": tumor_metrics["sensitivity_macro"],
        "specificity_macro": tumor_metrics["specificity_macro"],
        "best_epoch": int(trainer.best_epoch),
        "checkpoint": str(out_dir / "ckpt" / "best_model.pt"),
        "actual_optimizer_lrs": actual_lrs,
        "train_persistent_workers": train_persistent_workers,
        "initialization": initialization,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    protocol_check = {
        "initialization": initialization,
        "pretrained_he_ckpt": None,
        "pretrained_pr_ckpt": None,
        "all_active_modules_trainable": True,
        "actual_optimizer_lrs": actual_lrs,
        "train_persistent_workers": train_persistent_workers,
        "use_correction_only": cfg["model"].get("use_correction_only", False),
        "use_logit_attn": cfg["model"].get("use_logit_attn", False),
    }
    with open(out_dir / "protocol_check.json", "w") as f:
        json.dump(protocol_check, f, indent=2)

    write_test_predictions(cfg, trainer, out_dir, build_feature_dirs, C16MultimodalDataset)

    print(f"[seed {seed} gpu {args.gpu}] auc={result['auc']:.4f} "
          f"acc={result['accuracy']:.4f} f1={result['f1']:.4f} "
          f"epoch={result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
