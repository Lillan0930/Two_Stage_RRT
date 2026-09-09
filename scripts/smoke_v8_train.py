#!/usr/bin/env python3
"""v8 训练路径冒烟测试 —— 真实 Trainer 端到端跑 routed/patch 两个小子集（§6 集成层）。

验证：
  1. is_v8 生效且训练走默认 v3 路径（单 fused CE、全局裁剪；is_v6/is_v7 均为 False）；
  2. init_hash 已记录（routed/patch 同 seed 应一致——在主 run 中核对）；
  3. best_model.pt 保存、best_val_auc 有限；
  4. patch 模式在真实数据上前向正常（K = 实际 PR patch 数）；
  5. eval 前向（normal dispatch）正常。

用 270/129 协议文件的前 8 个 slide 做迷你集，max_patches=64，num_epochs=2，
routed/patch 各跑一次，单卡约 3–4 分钟。不作为任何实验结论。

用法:
  python scripts/smoke_v8_train.py --gpu 7
"""
import os, sys, json, logging, argparse, csv
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
os.chdir(str(PROJECT))

SMOKE_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v8" / "_smoke"
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


def run(device, mode: str):
    from run_stage2_v8 import build_config, MODES

    train_sub = SMOKE_ROOT / f"{mode}_train.csv"
    val_sub = SMOKE_ROOT / f"{mode}_val.csv"
    _subset_label_file(PROJECT / "data/C16_labels/c16_train_labels.csv", train_sub, N_SMOKE)
    _subset_label_file(PROJECT / "data/C16_labels/c16_test_labels.csv", val_sub, N_SMOKE)

    cfg = build_config(mode, 42)
    cfg["training"]["num_epochs"] = NUM_EPOCHS
    cfg["data"]["train_label_file"] = str(train_sub)
    cfg["data"]["val_label_file"] = str(val_sub)
    cfg["data"]["max_patches"] = MAX_PATCHES
    cfg["data"]["sample_seed"] = 42
    cfg["environment"]["num_workers"] = 0
    cfg["environment"]["seed"] = 42
    out_dir = SMOKE_ROOT / f"run_{mode}"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")

    from train import Trainer
    logger = logging.getLogger(f"smoke_v8_{mode}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False

    trainer = Trainer(cfg, logger, f"smoke_v8_{mode}")
    assert getattr(trainer, "is_v8", False) is True, "stage2_type must set is_v8"
    assert getattr(trainer, "is_v6", False) is False and \
        getattr(trainer, "is_v7", False) is False, \
        "v8 must use the default v3 training path (no v6/v7 special branches)"
    assert trainer.v8_pr_memory_mode == MODES[mode], "mode flag mismatch"

    model, best = trainer.train()

    assert trainer.init_hash is not None, "init_hash must be recorded"
    ckpt = out_dir / "ckpt" / "best_model.pt"
    assert ckpt.exists(), "best_model.pt missing"
    assert best is not None and np.isfinite(best), f"best metric non-finite: {best}"

    # 真实数据上前向：patch 模式 K = 实际 PR patch 数
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
    n_pr = smp["features"]["PR"].shape[0]
    model.eval()
    with torch.no_grad():
        logits = model([smp["features"]["HE"].to(device).unsqueeze(0),
                        smp["features"]["PR"].to(device).unsqueeze(0)])[0]
    assert logits.shape[-1] == 2
    if mode == "patch":
        cm = model.cross_region_mod
        captured = {}

        def _hook(m, inp, out):
            captured['K'] = inp[0].shape[1]

        handle = cm.w_k.register_forward_hook(_hook)
        with torch.no_grad():
            model([smp["features"]["HE"].to(device).unsqueeze(0),
                   smp["features"]["PR"].to(device).unsqueeze(0)])
        handle.remove()
        assert captured['K'] == n_pr, \
            f"patch-mode K must equal actual PR patch count {n_pr}, got {captured['K']}"

    return {
        "mode": mode,
        "best_val_auc": float(best),
        "init_hash": trainer.init_hash,
        "n_pr_patches": n_pr,
        "patch_k_len": captured.get('K') if mode == 'patch' else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=7)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    print(f"[smoke] gpu={args.gpu} device={device}", flush=True)

    results = {}
    for mode in ("routed", "patch"):
        print(f"\n[smoke] v8-{mode}", flush=True)
        r = run(device, mode)
        results[mode] = r
        print(f"[smoke] v8-{mode} best_val_auc={r['best_val_auc']:.4f} "
              f"hash={r['init_hash'][:12]}… "
              f"n_pr={r['n_pr_patches']} k_len={r['patch_k_len']}", flush=True)

    (SMOKE_ROOT / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
