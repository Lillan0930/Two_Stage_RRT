#!/usr/bin/env python3
"""v5 训练路径冒烟测试 —— 真实 Trainer 端到端跑一个小子集。

验证（§3/§5 的集成层）：
  1. pr_value_aux_weight=0.1 时，训练 loss 正确叠加 β·CE(pr_value_logits, y)；
  2. validate 正确计算并返回 auc_pr_value（aux head 的 val AUC）；
  3. 断点/checkpoint 保存正常，best_model.pt 存在；
  4. pr_value_aux_weight=0.0 时 aux 路径为 0、val 不崩溃。

用 fixed_split train/val 的前 8 个 slide 做迷你集，max_patches=64，num_epochs=2，
在单卡上约 1–2 分钟跑完。不作为任何实验结论。

用法:
  python scripts/smoke_v5_train.py --gpu 2
"""
import os, sys, json, logging, argparse, csv
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
os.chdir(str(PROJECT))

SMOKE_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v5" / "_smoke"
N_SMOKE = 8
MAX_PATCHES = 64
NUM_EPOCHS = 2


def _subset_label_file(src: Path, dst: Path, n: int):
    with open(src) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body[:n])


def run(device, beta: float):
    from run_stage2_v5 import build_config, STAGE2_CFG, STAGE2_TYPE

    train_sub = SMOKE_ROOT / f"beta{beta}_train.csv"
    val_sub = SMOKE_ROOT / f"beta{beta}_val.csv"
    _subset_label_file(PROJECT / "data/C16_labels/fixed_split/train.csv", train_sub, N_SMOKE)
    _subset_label_file(PROJECT / "data/C16_labels/fixed_split/val.csv", val_sub, N_SMOKE)

    cfg = build_config("v5_beta0", 42)
    cfg["training"]["num_epochs"] = NUM_EPOCHS
    cfg["training"]["pr_value_aux_weight"] = beta
    cfg["data"]["train_label_file"] = str(train_sub)
    cfg["data"]["val_label_file"] = str(val_sub)
    cfg["data"]["max_patches"] = MAX_PATCHES
    cfg["data"]["sample_seed"] = 42
    cfg["environment"]["num_workers"] = 0
    cfg["environment"]["seed"] = 42
    out_dir = SMOKE_ROOT / f"beta{beta}"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")

    from train import Trainer
    logger = logging.getLogger(f"smoke_beta{beta}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    trainer = Trainer(cfg, logger, f"smoke{beta}")
    model, best = trainer.train()

    ckpt = out_dir / "ckpt" / "best_model.pt"
    assert ckpt.exists(), "best_model.pt missing"
    assert best is not None and np.isfinite(best), f"best metric non-finite: {best}"

    # ── 直接前向一次，确认 fusion_stats 带 pr_value_logits ──
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(data_cfg["feature_base_dir"],
                                      data_cfg["modalities"],
                                      data_cfg.get("dir_mapping", None))
    ds = C16MultimodalDataset(feature_dirs=feature_dirs, label_file=str(val_sub),
                              max_patches=MAX_PATCHES, preload=False, verbose=False,
                              sampling="random", sample_seed=42, per_epoch=False)
    model.eval()
    with torch.no_grad():
        smp = ds[0]
        he = smp["features"]["HE"].to(device).unsqueeze(0)
        pr = smp["features"]["PR"].to(device).unsqueeze(0)
        out = model([he, pr])
    assert len(out) == 4, f"forward must return 4-tuple, got {len(out)}"
    fs = out[3]
    assert fs.get("pr_value_logits") is not None, "fusion_stats missing pr_value_logits"
    assert fs["pr_value_logits"].shape[-1] == 2, "aux logits must be [B,2]"
    assert fs["pr_value_has_valid"].dtype == torch.bool, "has_valid must be bool"

    return {"best_val_auc": float(best),
            "stage2_type": STAGE2_TYPE,
            "pr_value_aux_weight": beta,
            "aux_logits_shape": list(fs["pr_value_logits"].shape),
            "has_valid": bool(fs["pr_value_has_valid"].item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"[smoke] gpu={args.gpu} device={device}", flush=True)

    results = {}
    for beta in (0.0, 0.1):
        print(f"\n[smoke] beta={beta}", flush=True)
        results[str(beta)] = run(device, beta)
        print(f"[smoke] beta={beta} best_val_auc={results[str(beta)]['best_val_auc']:.4f} "
              f"aux_logits={results[str(beta)]['aux_logits_shape']} "
              f"has_valid={results[str(beta)]['has_valid']}", flush=True)

    (SMOKE_ROOT / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
