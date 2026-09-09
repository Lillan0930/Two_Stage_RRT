#!/usr/bin/env python3
"""HE Residual Cross **v6** —— Train / Val-as-Test 协议 paired driver（§7）。

v6 = v3 前向结构 + 训练期「HE vs PR+cross」梯度分工：
    L = CE(M(H), y) + CE(M_detached(Stage2(H.detach(), P)), y)
    HE 投影/RRT + 主 ABMIL(M) 只由 HE CE 更新；
    PR 投影/RRT + 全部 Stage2 只由 fused CE 更新。
前向一次、两个 loss、两个参数组各自 clip_grad_norm_(1.0)。

本脚本在**同一个 270/129 协议**下配对运行两个条件：
    v3  (control, stage2_type='he_residual_cross_v3')   —— 单 loss，常规训练
    v6  (target,  stage2_type='he_residual_cross_v6')   —— 双 loss，梯度分工
两者公共模块（patch_to_emb / rrt_he / rrt_ihc / mil / cross_region_mod 均为同一
类 HEResidualCrossCRMSAv3）在**同 seed 下初始化逐位一致**；slide 顺序（DataLoader
shuffle 走 torch RNG = set_seed(model_seed)）、patch 采样（sample_seed=seed）一致。
因此 per-seed 的 v6−v3 AUC 是配对比较。

注意：此 v3 是**本目录内配对重跑的对照**，与历史 results/stage2_he_residual_cross_v3
可能因 cuDNN / 采样非确定性有微小差异（不应互相覆盖）。历史结果保持不动。

用法:
  python scripts/run_stage2_v6.py --gpus 0 1 4
  python scripts/run_stage2_v6.py --summary-only
"""
import os, sys, json, subprocess, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
SEEDS = [42, 123, 456]
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
MAX_PATCHES = 2500
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v6"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")

# Stage1 各自 best —— 与 v3/v5 完全一致，绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG；v6 与 v3 共用同一份（v6 仅改训练梯度分工，不改结构）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
}

# condition name → stage2_type
CONDITIONS = {
    "v3": "he_residual_cross_v3",   # control：v3 前向 + 单 loss
    "v6": "he_residual_cross_v6",   # target：v3 前向 + 双 loss 梯度分工
}
LABELS = {
    "v3": "HE Residual Cross v3 (paired control)",
    "v6": "HE Residual Cross v6 (gradient division)",
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(condition: str, seed: int):
    assert condition in CONDITIONS, f"unknown condition {condition}"
    training = {
        "batch_size": 1, "num_epochs": 80,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": {"type": "cosine"},
        "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
        "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
        "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
        "no_validation": False,
    }
    model = {
        "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
        "region_num": 4, "n_layers": 2, "n_heads": 4,
        "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
        "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
        "crmsa_heads": 8, "crmsa_mlp": False,
        "fusion_type": "two_stage_region", "fusion_stage": "middle",
        "use_gated_fusion": False, "abmil_hidden_dim": 256,
        "use_mclc": False, "aggregate_modalities": True,
        "stage2_type": CONDITIONS[condition],
        "encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": dict(STAGE2_CFG),
    }

    return {
        "data": {
            "dataset_type": "c16", "modalities": MODALITIES,
            "dir_mapping": DIR_MAPPING,
            "train_label_file": TRAIN_LABEL_FILE,
            "val_label_file": VAL_LABEL_FILE,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": MAX_PATCHES, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": model,
        "training": training,
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "seeds": {
            "model_seed": seed,
            "sampling_seed": seed,
            "shuffle_seed": "DataLoader shuffle=True via torch RNG (set_seed(model_seed))",
        },
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
        "protocol": {
            "evaluation_protocol": "Train / Val-as-Test",
            "train": 270, "val_as_test": 129,
            "note": "official test set 在训练中被用于 checkpoint selection / early stopping，"
                    "属既定 val-as-test 开发协议，非严格意义的独立 untouched test",
        },
    }


def is_complete(seed_dir: Path) -> bool:
    r = seed_dir / "result.json"
    p = seed_dir / "test_predictions.csv"
    if not (r.exists() and p.exists()):
        return False
    try:
        return "auc" in json.loads(r.read_text())
    except Exception:
        return False


def run_wave(pairs):
    procs = []
    for cond, seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / cond
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((cond, seed, p, log_f))
        print(f"[launch] {cond} seed={seed} gpu={gpu}", flush=True)
    for cond, seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / cond / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {cond} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        extra = ""
        if cond == "v6":
            extra = f" loss_he={r.get('v6_loss_he', float('nan')):.4f} " \
                    f"loss_fused={r.get('v6_loss_fused', float('nan')):.4f}"
        print(f"[done] {cond} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']}{extra}", flush=True)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "—"


def _load_auc(cond, seed):
    rpath = RESULTS_ROOT / cond / f"seed{seed}" / "result.json"
    if not rpath.exists():
        return None
    return json.loads(rpath.read_text())


def generate_summary():
    rec = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for s in SEEDS:
            r = _load_auc(cond, s)
            if r is None:
                continue
            rec[cond][s] = {
                "auc": r["auc"], "acc": r["accuracy"], "f1": r["f1"],
                "sensitivity_tumor": r["sensitivity_tumor"],
                "specificity_tumor": r["specificity_tumor"],
                "sensitivity_macro": r["sensitivity_macro"],
                "specificity_macro": r["specificity_macro"],
                "best_epoch": r["best_epoch"],
                "v6_loss_he": r.get("v6_loss_he"),
                "v6_loss_fused": r.get("v6_loss_fused"),
                "actual_optimizer_lrs": r.get("actual_optimizer_lrs"),
            }

    def _mean_std(cond, key):
        vals = [rec[cond][s][key] for s in SEEDS if s in rec[cond]]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    def _paired_delta():
        common = [s for s in SEEDS if s in rec["v3"] and s in rec["v6"]]
        deltas = [rec["v6"][s]["auc"] - rec["v3"][s]["auc"] for s in common]
        return {
            "common_seeds": common,
            "per_seed_delta": {str(s): round(rec["v6"][s]["auc"] - rec["v3"][s]["auc"], 6)
                               for s in common},
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        }

    per_seed = {}
    for cond in CONDITIONS:
        per_seed[cond] = {str(s): rec[cond][s]["auc"] for s in rec[cond]}

    auc_m6, auc_s6 = _mean_std("v6", "auc")
    auc_m3, auc_s3 = _mean_std("v3", "auc")

    summary = {
        "task": "stage2_he_residual_cross_v6",
        "evaluation_protocol": "Train / Val-as-Test",
        "protocol_note": ("official test set 在训练中被用于 checkpoint selection / early stopping，"
                          "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        "conditions": {
            "v3": {
                "label": LABELS["v3"], "stage2_type": CONDITIONS["v3"],
                "per_seed_auc": per_seed["v3"],
                "mean_auc": auc_m3, "std_auc": auc_s3,
                "best_epoch": {str(s): rec["v3"][s]["best_epoch"] for s in rec["v3"]},
                "actual_optimizer_lrs": rec["v3"][list(rec["v3"])[0]]["actual_optimizer_lrs"]
                if rec["v3"] else None,
            },
            "v6": {
                "label": LABELS["v6"], "stage2_type": CONDITIONS["v6"],
                "per_seed_auc": per_seed["v6"],
                "mean_auc": auc_m6, "std_auc": auc_s6,
                "best_epoch": {str(s): rec["v6"][s]["best_epoch"] for s in rec["v6"]},
                "loss_he": {str(s): rec["v6"][s]["v6_loss_he"] for s in rec["v6"]},
                "loss_fused": {str(s): rec["v6"][s]["v6_loss_fused"] for s in rec["v6"]},
                "actual_optimizer_lrs": rec["v6"][list(rec["v6"])[0]]["actual_optimizer_lrs"]
                if rec["v6"] else None,
            },
        },
        "paired_delta_auc": {
            "v6_minus_v3": _paired_delta(),
        },
        "protocol": {
            "train": 270, "val_as_test": 129, "monitor": "val_auc", "mode": "max",
            "patience": 10, "num_epochs": 80, "scheduler": "cosine",
            "sampling": "random", "train_per_epoch": True, "val_as_test_per_epoch": False,
            "batch_size": 1, "num_workers": 2, "max_patches": 2500,
            "lr": 1e-4, "unified_lr": True, "weight_decay": 1e-5,
            "dropout": 0.25, "abmil_hidden_dim": 256,
            "label_smoothing": 0.0, "aux_loss_weight": 0.0,
            "modality_dropout": 0.0, "kd_enabled": False,
            "model_seed": SEEDS, "sampling_seed": SEEDS,
            "shuffle_seed": "DataLoader shuffle=True via torch RNG (set_seed(model_seed))",
            "paired": ("v3 与 v6 同 seed 公共模块初始化逐位一致；slide 顺序 / patch 采样一致；"
                       "v3 为本目录内配对重跑对照"),
        },
        "v6_loss_objective": "CE(HE logits, y) + CE(fused logits, y)，两 loss 权重均 1",
        "v6_param_groups": {
            "A (HE CE 更新)": "patch_to_emb[0] / rrt_he / mil",
            "B (fused CE 更新)": "patch_to_emb[1] / rrt_ihc / cross_region_mod",
            "clip": "两组各自 clip_grad_norm_(max_norm=1.0)，无全模型联合裁剪",
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
        "note": "此 v3 为配对对照，非历史 results/stage2_he_residual_cross_v3（历史不动）。",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    # README
    L = []
    L.append("# HE Residual Cross v6 —— Train / Val-as-Test 协议（paired v3 vs v6）\n")
    L.append("**Evaluation protocol: Train / Val-as-Test**\n")
    L.append("> official test set 在训练中被用于 checkpoint selection / early stopping，"
             "因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。\n")
    L.append("v6 = v3 前向结构 + 训练期梯度分工：HE 投影/RRT + 主 ABMIL 只由 HE CE 更新；"
             "PR 投影/RRT + 全部 Stage2 只由 fused CE 更新。前向一次、两个 loss、两组各自裁剪。\n")

    L.append("## AUC（逐 seed；v3 为配对对照）\n")
    L.append("| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    L.append(f"| v3 (control) | {_fmt(per_seed['v3'].get('42'))} | {_fmt(per_seed['v3'].get('123'))} "
             f"| {_fmt(per_seed['v3'].get('456'))} | {_fmt(auc_m3)} ± {_fmt(auc_s3)} |")
    L.append(f"| v6 (target)  | {_fmt(per_seed['v6'].get('42'))} | {_fmt(per_seed['v6'].get('123'))} "
             f"| {_fmt(per_seed['v6'].get('456'))} | {_fmt(auc_m6)} ± {_fmt(auc_s6)} |")

    pd_ = summary["paired_delta_auc"]["v6_minus_v3"]
    L.append("\n## paired ΔAUC（v6 − v3）\n")
    L.append("| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    L.append(f"| v6 − v3 | {_fmt(pd_['per_seed_delta'].get('42'))} | "
             f"{_fmt(pd_['per_seed_delta'].get('123'))} | "
             f"{_fmt(pd_['per_seed_delta'].get('456'))} | "
             f"{_fmt(pd_['mean_delta'])} ± {_fmt(pd_['std_delta'])} |")

    if rec["v6"]:
        L.append("\n## v6 两个 loss（末 epoch 均值）\n")
        L.append("| Seed | CE_HE | CE_fused |")
        L.append("|---|---|---|")
        for s in SEEDS:
            if s in rec["v6"]:
                L.append(f"| {s} | {_fmt(rec['v6'][s]['v6_loss_he'])} "
                         f"| {_fmt(rec['v6'][s]['v6_loss_fused'])} |")

    L.append("\n## Stage2 config (v3 = v6)\n```\n" + json.dumps(STAGE2_CFG, indent=2) + "\n```\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] v3 mean AUC = {_fmt(auc_m3)} | v6 mean AUC = {_fmt(auc_m6)}", flush=True)
    print(f"[summary] v6 − v3 = {_fmt(pd_['mean_delta'])} ± {_fmt(pd_['std_delta'])}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 4])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        generate_summary()
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    todo = []
    for cond in CONDITIONS:
        (RESULTS_ROOT / cond).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = RESULTS_ROOT / cond / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] {cond} seed={seed} (complete)", flush=True)
                continue
            todo.append((cond, seed, seed_dir))

    print(f"to-run: {len(todo)} job(s) | gpus={args.gpus} | force={args.force}", flush=True)

    if todo:
        cfg_by_key = {}
        for cond, seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(cond, seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[(cond, seed)] = cfg_path

        flat = [(c, s, cfg_by_key[(c, s)]) for (c, s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(c, s, g, cfg_path)
                     for (c, s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    if any((RESULTS_ROOT / cond / f"seed{s}" / "result.json").exists()
           for cond in CONDITIONS for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
