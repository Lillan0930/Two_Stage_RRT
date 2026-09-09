#!/usr/bin/env python3
"""v7 训练路径冒烟测试 —— 真实 Trainer 端到端跑 A/B 两个小子集（§5 集成层）。

验证：
  1. is_v7 生效，训练走 _train_epoch_v7 双 loss 分支（CE_HE + CE_fused）；
  2. forward_v7 返回 he_logits / fused_logits / fused_features；
  3. 三个参数组非空且覆盖全部可训参数（v7_other 为空）；
  4. best_model.pt 保存、best_val_auc 有限、两个 loss 有限；
  5. init_hash 已记录（A/B 同 seed 应一致——在主 run 中核对）；
  6. 每 epoch 的 fused AUC 与 HE 路径 AUC（logits_he）都被记录。

用 270/129 协议文件的前 8 个 slide 做迷你集，max_patches=64，num_epochs=2，
A/B 各跑一次，单卡约 3–4 分钟。不作为任何实验结论。

用法:
  python scripts/smoke_v7_train.py --gpu 7
"""
import os, sys, json, logging, argparse, csv
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
os.chdir(str(PROJECT))

SMOKE_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v7" / "_smoke"
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


def run(device, variant: str):
    from run_stage2_v7 import build_config, VARIANTS

    train_sub = SMOKE_ROOT / f"{variant}_train.csv"
    val_sub = SMOKE_ROOT / f"{variant}_val.csv"
    _subset_label_file(PROJECT / "data/C16_labels/c16_train_labels.csv", train_sub, N_SMOKE)
    _subset_label_file(PROJECT / "data/C16_labels/c16_test_labels.csv", val_sub, N_SMOKE)

    cfg = build_config(variant, 42)
    cfg["training"]["num_epochs"] = NUM_EPOCHS
    cfg["data"]["train_label_file"] = str(train_sub)
    cfg["data"]["val_label_file"] = str(val_sub)
    cfg["data"]["max_patches"] = MAX_PATCHES
    cfg["data"]["sample_seed"] = 42
    cfg["environment"]["num_workers"] = 0
    cfg["environment"]["seed"] = 42
    out_dir = SMOKE_ROOT / f"run_{variant}"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")

    from train import Trainer
    logger = logging.getLogger(f"smoke_v7{variant}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    trainer = Trainer(cfg, logger, f"smoke_v7{variant}")
    assert getattr(trainer, "is_v7", False) is True, "stage2_type must set is_v7"
    assert trainer.v7_fused_he_detach == VARIANTS[variant], "variant flag mismatch"

    model, best = trainer.train()

    assert trainer.v7_group_he_params and trainer.v7_group_mil_params \
        and trainer.v7_group_pr2_params, "three v7 param groups must be non-empty"
    assert not trainer.v7_other_params, \
        f"uncovered params: {len(trainer.v7_other_params)}"
    assert trainer.init_hash is not None, "init_hash must be recorded"

    ckpt = out_dir / "ckpt" / "best_model.pt"
    assert ckpt.exists(), "best_model.pt missing"
    assert best is not None and np.isfinite(best), f"best metric non-finite: {best}"
    assert np.isfinite(trainer.v7_avg_loss_he) and trainer.v7_avg_loss_he > 0
    assert np.isfinite(trainer.v7_avg_loss_fused) and trainer.v7_avg_loss_fused > 0

    # 每 epoch 历史：fused AUC + HE AUC
    assert len(trainer.v7_epoch_history) == NUM_EPOCHS, "epoch history length mismatch"
    for h in trainer.v7_epoch_history:
        assert np.isfinite(h["val_auc"]), f"val_auc non-finite: {h}"
        assert h["val_auc_he"] is None or np.isfinite(h["val_auc_he"]), \
            f"val_auc_he non-finite: {h}"

    # forward_v7 结构 + eval 前向正常
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(data_cfg["feature_base_dir"],
                                      data_cfg["modalities"],
                                      data_cfg.get("dir_mapping", None))
    ds = C16MultimodalDataset(feature_dirs=feature_dirs, label_file=str(val_sub),
                              max_patches=MAX_PATCHES, preload=False, verbose=False,
                              sampling="random", sample_seed=42, per_epoch=False)
    smp = ds[0]
    he = smp["features"]["HE"].to(device).unsqueeze(0)
    pr = smp["features"]["PR"].to(device).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model([he, pr])[0]
    assert logits.shape[-1] == 2
    model.train()
    out7 = model.forward_v7([smp["features"]["HE"].to(device).unsqueeze(0),
                             smp["features"]["PR"].to(device).unsqueeze(0)])
    for k in ("he_logits", "fused_logits", "fused_features"):
        assert k in out7, f"forward_v7 missing {k}"
    assert out7["he_logits"].shape[-1] == 2
    assert out7["fused_logits"].shape[-1] == 2

    return {
        "variant": variant,
        "best_val_auc": float(best),
        "loss_he": float(trainer.v7_avg_loss_he),
        "loss_fused": float(trainer.v7_avg_loss_fused),
        "init_hash": trainer.init_hash,
        "n_he": len(trainer.v7_group_he_params),
        "n_mil": len(trainer.v7_group_mil_params),
        "n_pr2": len(trainer.v7_group_pr2_params),
        "epoch_history": trainer.v7_epoch_history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"[smoke] gpu={args.gpu} device={device}", flush=True)

    results = {}
    for variant in ("A", "B"):
        print(f"\n[smoke] v7{variant}", flush=True)
        r = run(device, variant)
        results[variant] = {k: v for k, v in r.items() if k != "epoch_history"}
        results[variant]["epoch_history"] = r["epoch_history"]
        print(f"[smoke] v7{variant} best_val_auc={r['best_val_auc']:.4f} "
              f"loss_he={r['loss_he']:.4f} loss_fused={r['loss_fused']:.4f} "
              f"groups he/mil/pr2={r['n_he']}/{r['n_mil']}/{r['n_pr2']} "
              f"hash={r['init_hash'][:12]}…", flush=True)

    (SMOKE_ROOT / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
