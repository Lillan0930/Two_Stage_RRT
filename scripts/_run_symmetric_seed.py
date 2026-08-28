#!/usr/bin/env python3
"""对称双模态 official-style CR-MSA 单 seed worker（训练 + 后处理）。

读取一个完整 config.json（含 encoder_cfg / stage2_cfg / seed / lr_stage1 / lr_stage2），
在指定 GPU 上训练，随后做后处理并写产物（保留 checkpoint，不删除）：

  - result.json             : auc / acc / f1 / sensitivity / specificity / best_epoch / checkpoint 路径
  - test_predictions.csv    : 129 test 的 slide_id,label,prob_pos,pred（best checkpoint 重估）
  - stage2_magnitude_seed42.json : 仅 seed==42 —— ||delta||/||z|| 幅度诊断（HE 与 PR 各自）

padding 统计（N/H/W/add_length/padding_ratio）由 driver 统一计算（与 seed 无关）。

用法:
  python scripts/_run_symmetric_seed.py --config <path/config.json> --gpu 3
"""
import os, sys, json, math, logging, argparse
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))


def write_test_predictions(cfg, trainer, out_dir, build_feature_dirs, C16MultimodalDataset):
    """用 trainer.best_val_probs 写 129 test 的 test_predictions.csv。

    val_loader 的 slide 顺序 = 数据集按 slide_id 排序；best_val_probs 行序与之对应。
    """
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


def write_magnitude(model, cfg, out_dir, build_feature_dirs, C16MultimodalDataset):
    """seed42 专用：Stage2 幅度诊断 ||delta||/||z||（HE/PR 各自）。"""
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
    mod = model.cross_region_mod
    dev = next(model.parameters()).device
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            he = item["features"]["HE"].to(dev).unsqueeze(0)
            pr = item["features"]["PR"].to(dev).unsqueeze(0)
            z_he = model.rrt_he(model.patch_to_emb[0](he))
            z_pr = model.rrt_ihc(model.patch_to_emb[1](pr))
            if z_he.dim() == 2:
                z_he = z_he.unsqueeze(0)
            if z_pr.dim() == 2:
                z_pr = z_pr.unsqueeze(0)
            z_he_n = mod.norm(z_he)
            z_pr_n = mod.norm(z_pr)
            r_he = mod._combine(z_he_n)
            r_pr = mod._combine(z_pr_n)
            routing = torch.cat([r_he[0], r_pr[0]], dim=1)
            routing = mod.attn(routing)
            n_he = r_he[0].shape[1]
            delta_he = mod._dispatch(routing[:, :n_he], r_he[1], r_he[2],
                                     r_he[3], r_he[4], r_he[5], r_he[6])
            delta_pr = mod._dispatch(routing[:, n_he:], r_pr[1], r_pr[2],
                                     r_pr[3], r_pr[4], r_pr[5], r_pr[6])
            zn_he = float(z_he.norm())
            zn_pr = float(z_pr.norm())
            dn_he = float(delta_he.norm())
            dn_pr = float(delta_pr.norm())
            rows.append({
                "slide": item["slide_id"],
                "norm_z_he": round(zn_he, 4), "norm_z_pr": round(zn_pr, 4),
                "norm_delta_he": round(dn_he, 4), "norm_delta_pr": round(dn_pr, 4),
                "ratio_he": round(dn_he / (zn_he + 1e-8), 4),
                "ratio_pr": round(dn_pr / (zn_pr + 1e-8), 4),
            })
    r_he = np.array([r["ratio_he"] for r in rows])
    r_pr = np.array([r["ratio_pr"] for r in rows])
    def summ(v):
        return {
            "mean": round(float(np.mean(v)), 4),
            "median": round(float(np.median(v)), 4),
            "min": round(float(np.min(v)), 4),
            "max": round(float(np.max(v)), 4),
        }
    diag = {
        "note": "||delta||/||z|| for Stage2 symmetric CR-MSA (seed42, test set)",
        "region_num": cfg["model"]["stage2_cfg"]["region_num"],
        "epeg": cfg["model"]["stage2_cfg"].get("epeg", False),
        "ratio_he": summ(r_he),
        "ratio_pr": summ(r_pr),
        "per_slide": rows,
    }
    (out_dir / "stage2_magnitude_seed42.json").write_text(json.dumps(diag, indent=2) + "\n")


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
    sc = cfg["model"]["stage2_cfg"]

    logger = logging.getLogger(f"sym_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"symmetric CR-MSA seed={seed} region_num={sc['region_num']} "
                f"epeg={sc.get('epeg')}")

    trainer = Trainer(cfg, logger, f"s{seed}")
    model, _best_metric = trainer.train()

    result = {
        "seed": seed,
        "val_auc": float(trainer.best_val_auc),
        "val_acc": float(trainer.best_val_acc),
        "val_f1": float(trainer.best_val_f1),
        "val_sensitivity": float(trainer.best_val_sensitivity),
        "val_specificity": float(trainer.best_val_specificity),
        "best_epoch": int(trainer.best_epoch),
        "checkpoint": str(out_dir / "ckpt" / "best_model.pt"),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    write_test_predictions(cfg, trainer, out_dir, build_feature_dirs, C16MultimodalDataset)

    if seed == 42:
        write_magnitude(model, cfg, out_dir, build_feature_dirs, C16MultimodalDataset)

    print(f"[seed {seed} gpu {args.gpu}] auc={result['val_auc']:.4f} "
          f"acc={result['val_acc']:.4f} f1={result['val_f1']:.4f} "
          f"epoch={result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
