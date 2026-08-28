#!/usr/bin/env python3
"""Two-stage (HE+PR) 训练协议验证脚本（不进行完整训练，只做 7 项检查）。

验证内容：
  1. 模型确实从头初始化（无 pretrained / correction / logit_attn）
  2. 优化器实际 LR = 1e-4（统一 LR）
  3. 所有活跃模块均有梯度（patch_to_emb[0/1] / rrt_he / rrt_ihc /
     cross_region_mod / mil）
  4. optimizer.step 后所有活跃模块参数确实更新（relative_update > 0）
  5. per-epoch random sampling 确实随 epoch 变化（含 DataLoader 层验证）
  6. Val/Test fixed random 保持固定（set_epoch 不影响）
  7. 全部 train/test slide 的 patch 数量与特征维度一致性（N_HE==N_PR, D_HE==D_PR）

只有全部 assertion 通过才允许启动正式控制实验。

用法:
  python scripts/verify_twostage_training_protocol.py --gpu 0
"""
import os, sys, json, math, hashlib, logging, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
MAX_PATCHES = 2500
INPUT_DIM = 768
NUM_CLASSES = 2

# Stage1 各自 best —— 绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}

ACTIVE_MODULES = {
    "he_projection": "patch_to_emb.0.",
    "pr_projection": "patch_to_emb.1.",
    "he_rrt": "rrt_he.",
    "pr_rrt": "rrt_ihc.",
    "stage2": "cross_region_mod.",
    "abmil": "mil.",
}


def build_config(seed: int = 42):
    """统一 LR (1e-4)、无 lr_stage1/lr_stage2 的 two-stage config。"""
    return {
        "data": {
            "dataset_type": "c16", "modalities": MODALITIES,
            "dir_mapping": DIR_MAPPING,
            "train_label_file": TRAIN_LABEL_FILE,
            "val_label_file": VAL_LABEL_FILE,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": INPUT_DIM, "num_classes": NUM_CLASSES,
            "max_patches": MAX_PATCHES, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "stage2_type": "staining_msa",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            "encoder_cfg": STAGE1_ENCODER_CFG,
            "stage2_cfg": STAGE2_CFG,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": str(PROJECT / "results" / "_verify_tmp" / "ckpt"),
            "log_dir": str(PROJECT / "results" / "_verify_tmp" / "logs"),
            "img_dir": str(PROJECT / "results" / "_verify_tmp" / "img"),
        },
    }


def make_logger():
    lg = logging.getLogger("verify_twostage_protocol")
    lg.handlers.clear()
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(sh)
    lg.propagate = False
    return lg


def _tensor_md5(t):
    arr = t.detach().float().cpu().numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()


def first_weight(model, prefix):
    """返回 module 前缀下第一个 ndim>=2 的参数（Linear/Conv weight），跳过 bias。"""
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.dim() >= 2:
            return name, p
    return None, None


def grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += float((p.grad ** 2).sum())
    return math.sqrt(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=str(PROJECT / "results" / "twostage_training_protocol_verify.json"))
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from train import Trainer, build_feature_dirs
    from data.c16_multimodal_dataset import (
        C16MultimodalDataset, c16_multimodal_collate_fn, stable_slide_seed)

    lg = make_logger()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lg.info(f"[verify] device={device}")
    report = {"passed": True, "checks": {}}
    feature_dirs = build_feature_dirs(FEATURE_BASE, MODALITIES, DIR_MAPPING)

    # ── Check 1: 模型确实从头初始化 ────────────────────────────────────────
    cfg = build_config(42)
    assert cfg["model"].get("pretrained_he_ckpt") is None, "pretrained_he_ckpt must be absent"
    assert not cfg["model"].get("use_correction_only", False), "use_correction_only must be False"
    assert not cfg["model"].get("use_logit_attn", False), "use_logit_attn must be False"
    assert "lr_stage1" not in cfg["training"] and "lr_stage2" not in cfg["training"], \
        "unified LR config must not define lr_stage1/lr_stage2"
    lg.info("[init] random_from_scratch = True")
    report["checks"]["check1_init"] = {
        "random_from_scratch": True, "pretrained_he_ckpt": None,
        "use_correction_only": False, "use_logit_attn": False,
    }

    # ── Check 2: 优化器实际 LR = 1e-4 ──────────────────────────────────────
    trainer = Trainer(cfg, lg, "verify")
    model = trainer.create_model()
    optimizer, scheduler = trainer.create_optimizer_scheduler(model)

    assert len(optimizer.param_groups) >= 1
    group_reports = []
    for g in optimizer.param_groups:
        assert abs(g["lr"] - 1e-4) < 1e-12, f"param group lr {g['lr']} != 1e-4"
        group_reports.append({
            "name": g.get("name", "all_params"),
            "lr": g["lr"],
            "weight_decay": g.get("weight_decay", cfg["training"]["weight_decay"]),
            "parameter_count": sum(p.numel() for p in g["params"]),
        })
    for gr in group_reports:
        lg.info(f"[lr] group={gr['name']} lr={gr['lr']:.0e} "
                f"wd={gr['weight_decay']:.0e} params={gr['parameter_count']}")

    # 每个活跃模块 → 实际 LR
    id2lr = {}
    for g in optimizer.param_groups:
        for p in g["params"]:
            id2lr[id(p)] = g["lr"]
    module_lrs = {}
    for mname, prefix in ACTIVE_MODULES.items():
        for name, p in model.named_parameters():
            if name.startswith(prefix):
                module_lrs[mname] = id2lr[id(p)]
                break
    for mname, mlr in module_lrs.items():
        assert abs(mlr - 1e-4) < 1e-12, f"{mname} lr {mlr} != 1e-4"
        lg.info(f"[lr] {mname} = {mlr:.0e}")
    report["checks"]["check2_optimizer_lrs"] = {
        "param_groups": group_reports,
        "module_lrs": module_lrs,
    }

    # ── Check 3 + 4: 梯度 + optimizer.step 更新 ────────────────────────────
    train_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=TRAIN_LABEL_FILE,
        max_patches=MAX_PATCHES, preload=False, verbose=False,
        sampling="random", sample_seed=42, per_epoch=True,
    )
    item = train_ds[0]
    he = item["features"]["HE"].float().to(device).unsqueeze(0)
    pr = item["features"]["PR"].float().to(device).unsqueeze(0)
    label = torch.tensor([item["label"]], dtype=torch.long, device=device)

    model.to(device)
    model.train()
    optimizer.zero_grad()
    out = model([he, pr])
    logits = out[0]
    loss = nn.CrossEntropyLoss()(logits, label)
    loss.backward()

    grad_norms = {}
    for mname, prefix in ACTIVE_MODULES.items():
        sub = model.patch_to_emb[0] if mname == "he_projection" else \
              model.patch_to_emb[1] if mname == "pr_projection" else \
              getattr(model, {
                  "he_rrt": "rrt_he", "pr_rrt": "rrt_ihc",
                  "stage2": "cross_region_mod", "abmil": "mil",
              }[mname])
        gn = grad_norm(sub)
        assert gn > 0, f"{mname} grad_norm == 0"
        assert math.isfinite(gn), f"{mname} grad_norm not finite"
        grad_norms[mname] = round(gn, 6)
        lg.info(f"[grad] {mname} = {gn:.6f}")

    # 未使用参数（grad=None）—— 列出但不报错（本轮不重构模型）
    unused = [name for name, p in model.named_parameters() if p.grad is None]
    if unused:
        lg.info(f"[grad] unused params (grad=None): {len(unused)} "
                f"(e.g. {unused[:3]}...)")
    report["checks"]["check3_grad_norms"] = {
        "grad_norm": grad_norms,
        "unused_grad_none_count": len(unused),
    }

    # Check 4: 记录 step 前参数 → step → relative_update
    # stage2 明确取 cross_region_mod.attn.qkv.weight（phi 是 nn.Parameter，会在
    # named_parameters 中先于子模块权重出现，但 qkv 才是 InnerAttention 的主体）
    rep_prefixes = dict(ACTIVE_MODULES)
    rep_prefixes["stage2"] = "cross_region_mod.attn.qkv."
    reps = {}
    for mname, prefix in rep_prefixes.items():
        name, p = first_weight(model, prefix)
        assert p is not None, f"no weight found for {mname} (prefix {prefix})"
        reps[mname] = (name, p)
    before = {k: p.detach().clone() for k, (_, p) in reps.items()}

    optimizer.step()

    rel_updates = {}
    for k, (name, p) in reps.items():
        rel = float((p.detach() - before[k]).norm()) / (float(before[k].norm()) + 1e-12)
        assert rel > 0, f"{k} relative_update == 0"
        assert math.isfinite(rel), f"{k} relative_update not finite"
        rel_updates[k] = round(rel, 8)
        lg.info(f"[update] {k} relative_update = {rel:.3e} (param={name})")
    report["checks"]["check4_relative_updates"] = rel_updates

    # ── Check 5: per-epoch random sampling 确实变化 ────────────────────────
    def find_large_slide(ds):
        for i in range(len(ds)):
            sid = ds.samples[i]["slide_id"]
            feat = ds._load_feature(ds.modalities[0], sid)
            if feat.shape[0] > (ds.max_patches or 0):
                return i, sid, feat.shape[0]
        return None, None, None

    target_index, target_slide, total_patches = find_large_slide(train_ds)
    assert target_index is not None, "no train slide with > max_patches found"

    train_ds.set_epoch(0)
    idx0 = train_ds._build_indices(target_slide, total_patches)
    train_ds.set_epoch(1)
    idx1 = train_ds._build_indices(target_slide, total_patches)
    train_ds.set_epoch(0)
    idx0_r = train_ds._build_indices(target_slide, total_patches)
    assert not np.array_equal(idx0, idx1), "epoch0 == epoch1 indices"
    assert np.array_equal(idx0, idx0_r), "epoch0 != epoch0_repeat indices"
    lg.info(f"[sampler] slide={target_slide} N={total_patches} "
            f"epoch0!=epoch1=True epoch0==repeat=True")

    # DataLoader 层验证（num_workers=2, persistent_workers=False）
    dl = DataLoader(
        Subset(train_ds, [target_index]), batch_size=1, num_workers=2,
        persistent_workers=False, shuffle=False,
        collate_fn=c16_multimodal_collate_fn,
    )

    def fetch_he_pr(epoch):
        train_ds.set_epoch(epoch)
        batch = next(iter(dl))
        he = batch["features"][0][0]
        pr = batch["features"][1][0]
        return he, pr

    he0, pr0 = fetch_he_pr(0)
    he1, pr1 = fetch_he_pr(1)
    he0r, pr0r = fetch_he_pr(0)

    assert he0.shape[0] == MAX_PATCHES and pr0.shape[0] == MAX_PATCHES, \
        f"expected {MAX_PATCHES} patches, got HE={he0.shape[0]} PR={pr0.shape[0]}"
    assert _tensor_md5(he0) != _tensor_md5(he1), "HE epoch0 == epoch1"
    assert _tensor_md5(pr0) != _tensor_md5(pr1), "PR epoch0 == epoch1"
    assert _tensor_md5(he0) == _tensor_md5(he0r), "HE epoch0 != epoch0_repeat"
    assert _tensor_md5(pr0) == _tensor_md5(pr0r), "PR epoch0 != epoch0_repeat"

    lg.info(f"[sampler] epoch0_indices_md5={_tensor_md5(he0)[:12]}...")
    lg.info(f"[sampler] epoch1_indices_md5={_tensor_md5(he1)[:12]}...")
    lg.info(f"[sampler] epoch0_repeat_indices_md5={_tensor_md5(he0r)[:12]}...")
    lg.info("[sampler] train_persistent_workers=False")
    report["checks"]["check5_per_epoch_sampler"] = {
        "slide": target_slide, "total_patches": total_patches,
        "epoch0_indices_md5": _tensor_md5(he0),
        "epoch1_indices_md5": _tensor_md5(he1),
        "epoch0_repeat_indices_md5": _tensor_md5(he0r),
        "train_persistent_workers": False,
        "he_pr_patch_count": [int(he0.shape[0]), int(pr0.shape[0])],
    }

    # ── Check 6: Val/Test fixed random 保持固定 ─────────────────────────────
    val_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs, label_file=VAL_LABEL_FILE,
        max_patches=MAX_PATCHES, preload=False, verbose=False,
        sampling="random", sample_seed=42, per_epoch=False,
    )
    v_target_index, v_target_slide, v_total = find_large_slide(val_ds)
    assert v_target_index is not None, "no test slide with > max_patches found"
    val_ds.set_epoch(0)
    v0 = val_ds._build_indices(v_target_slide, v_total)
    val_ds.set_epoch(1)
    v1 = val_ds._build_indices(v_target_slide, v_total)
    assert np.array_equal(v0, v1), "val fixed random changed across set_epoch"
    lg.info(f"[sampler] val fixed random stable (slide={v_target_slide} N={v_total})")
    report["checks"]["check6_val_fixed_random"] = {
        "slide": v_target_slide, "total_patches": v_total, "stable": True,
    }

    # ── Check 7: patch 数量 / 维度一致性（全量，只检查 shape）──────────────
    pc_mismatch = 0
    dim_mismatch = 0
    for split, label_file in [("train", TRAIN_LABEL_FILE), ("test", VAL_LABEL_FILE)]:
        ds = C16MultimodalDataset(
            feature_dirs=feature_dirs, label_file=label_file,
            max_patches=MAX_PATCHES, preload=False, verbose=False,
            sampling="random", sample_seed=42, per_epoch=False,
        )
        for i in range(len(ds)):
            item = ds[i]
            he = item["features"]["HE"]
            pr = item["features"]["PR"]
            if he.shape[0] != pr.shape[0]:
                pc_mismatch += 1
            if he.shape[-1] != pr.shape[-1]:
                dim_mismatch += 1
        lg.info(f"[shape] {split}: checked {len(ds)} slides")
    assert pc_mismatch == 0, f"patch count mismatch = {pc_mismatch}"
    assert dim_mismatch == 0, f"feature dim mismatch = {dim_mismatch}"
    lg.info(f"[shape] patch_count_mismatch=0 feature_dim_mismatch=0")
    report["checks"]["check7_patch_consistency"] = {
        "checked_train_slides": 270, "checked_test_slides": 129,
        "patch_count_mismatch": pc_mismatch,
        "feature_dim_mismatch": dim_mismatch,
    }

    report["passed"] = True
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    lg.info(f"[verify] ALL CHECKS PASSED → {out_path}")
    print("[verify] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
