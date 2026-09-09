#!/usr/bin/env python3
"""v6 训练路径冒烟测试 —— 真实 Trainer 端到端跑一个小子集（§6「覆盖真实训练分支」）。

验证（§3/§4/§5 的集成层）：
  1. is_v6 生效，训练走 _train_epoch_v6 双 loss 分支（CE_HE + CE_fused）；
  2. forward_v6 返回 he_logits / fused_logits / fused_features（fused 带梯度）；
  3. best_model.pt 保存、best_val_auc 有限；
  4. 末 epoch 的 v6_avg_loss_he / v6_avg_loss_fused 均为有限值；
  5. validate 用正常 eval 前向（fused logits）不崩溃。

用 270/129 协议文件的前 8 个 slide 做迷你集，max_patches=64，num_epochs=2，
单卡约 1–2 分钟。不作为任何实验结论。

用法:
  python scripts/smoke_v6_train.py --gpu 2
"""
import os, sys, json, logging, argparse, csv
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
os.chdir(str(PROJECT))

SMOKE_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v6" / "_smoke"
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


def run(device):
    from run_stage2_v6 import build_config, CONDITIONS

    train_sub = SMOKE_ROOT / "train.csv"
    val_sub = SMOKE_ROOT / "val.csv"
    _subset_label_file(PROJECT / "data/C16_labels/c16_train_labels.csv", train_sub, N_SMOKE)
    _subset_label_file(PROJECT / "data/C16_labels/c16_test_labels.csv", val_sub, N_SMOKE)

    cfg = build_config("v6", 42)
    cfg["training"]["num_epochs"] = NUM_EPOCHS
    cfg["data"]["train_label_file"] = str(train_sub)
    cfg["data"]["val_label_file"] = str(val_sub)
    cfg["data"]["max_patches"] = MAX_PATCHES
    cfg["data"]["sample_seed"] = 42
    cfg["environment"]["num_workers"] = 0
    cfg["environment"]["seed"] = 42
    out_dir = SMOKE_ROOT / "run"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")

    from train import Trainer
    logger = logging.getLogger("smoke_v6")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    trainer = Trainer(cfg, logger, "smoke_v6")
    assert getattr(trainer, "is_v6", False) is True, "stage2_type must set is_v6"

    model, best = trainer.train()

    # v6 param groups are built inside train() via create_optimizer_scheduler
    assert trainer.v6_group_a_params is not None and trainer.v6_group_b_params is not None, \
        "optimizer must build v6 param groups"
    assert len(trainer.v6_group_a_params) > 0 and len(trainer.v6_group_b_params) > 0, \
        "v6 param groups must be non-empty"

    ckpt = out_dir / "ckpt" / "best_model.pt"
    assert ckpt.exists(), "best_model.pt missing"
    assert best is not None and np.isfinite(best), f"best metric non-finite: {best}"
    assert np.isfinite(trainer.v6_avg_loss_he), "loss_he non-finite"
    assert np.isfinite(trainer.v6_avg_loss_fused), "loss_fused non-finite"
    assert trainer.v6_avg_loss_he > 0 and trainer.v6_avg_loss_fused > 0, \
        f"losses must be positive: he={trainer.v6_avg_loss_he:.4f} " \
        f"fused={trainer.v6_avg_loss_fused:.4f}"

    # ── 直接 forward_v6 一次，确认返回结构 + fused 带梯度 ──
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
        logits = out[0]
    assert logits.shape[-1] == 2, "eval forward logits must be [1,2]"

    model.train()
    out6 = model.forward_v6([smp["features"]["HE"].to(device).unsqueeze(0),
                             smp["features"]["PR"].to(device).unsqueeze(0)])
    for k in ("he_logits", "fused_logits", "fused_features"):
        assert k in out6, f"forward_v6 missing {k}"
    assert out6["he_logits"].shape[-1] == 2
    assert out6["fused_logits"].shape[-1] == 2
    assert out6["fused_features"].requires_grad, "fused_features must require grad"

    return {
        "best_val_auc": float(best),
        "stage2_type": CONDITIONS["v6"],
        "loss_he": float(trainer.v6_avg_loss_he),
        "loss_fused": float(trainer.v6_avg_loss_fused),
        "fused_features_requires_grad": bool(out6["fused_features"].requires_grad),
        "n_group_a": len(trainer.v6_group_a_params),
        "n_group_b": len(trainer.v6_group_b_params),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"[smoke] gpu={args.gpu} device={device}", flush=True)

    r = run(device)
    print(f"[smoke] best_val_auc={r['best_val_auc']:.4f} "
          f"loss_he={r['loss_he']:.4f} loss_fused={r['loss_fused']:.4f} "
          f"groupA={r['n_group_a']} groupB={r['n_group_b']} "
          f"fused_grad={r['fused_features_requires_grad']}", flush=True)

    (SMOKE_ROOT / "result.json").write_text(json.dumps(r, indent=2) + "\n")
    print("\n[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
