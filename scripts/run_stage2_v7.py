#!/usr/bin/env python3
"""HE Residual Cross **v7**（恢复融合梯度）—— Train / Val-as-Test 协议 paired driver。

v7 = v3 前向结构 + 训练期双 loss，同一主 ABMIL 无任何参数 detach：
    L = CE(M(H), y) + CE(M(F), y)       （两项权重均 1，主预测始终 M(F)）
    A：F = Stage2(H.detach(), P)  —— HE encoder 只接受 HE CE；MIL 接受两项 CE；
        PR encoder + Stage2 接受 fused CE。
    B：F = Stage2(H, P)           —— HE encoder 与 MIL 都接受两项 CE；PR/Stage2
        接受 fused CE。无 v6 式 HE/MIL 梯度隔离。

三个不重叠参数组（HE encoder / 主 MIL / PR encoder+Stage2）各自
`clip_grad_norm_(1.0)`，避免 MIL 梯度经共同裁剪系数间接改变其他组更新（§3）。

本脚本在 270/129 协议下配对运行 A/B（各 3 seeds）。同 seed 下 A/B 公共模块初始化
逐位一致（保存 init hash 供核对）；slide 顺序 / patch 采样一致。**已有同期 v3 结果
（results/stage2_he_residual_cross_v6/v3）作为性能基线，不重训、不覆盖（§6/§7）。**

用法:
  python scripts/run_stage2_v7.py --gpus 0 1 5
  python scripts/run_stage2_v7.py --summary-only
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
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v7"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")

# 同期 v3 基线（v6 轮的配对对照，直接复用，不重训）
V3_BASELINE_DIR = PROJECT / "results" / "stage2_he_residual_cross_v6" / "v3"

# Stage1 各自 best —— 与 v3/v6 完全一致，绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG；v7 与 v3/v6 共用同一份（仅训练期双 loss 不同）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
}

# variant name → v7_fused_he_detach
VARIANTS = {
    "A": True,    # F = Stage2(H.detach(), P)
    "B": False,   # F = Stage2(H, P)   —— 主候选
}
LABELS = {
    "A": "v7A: F=Stage2(H.detach(),P), HE only HE-CE",
    "B": "v7B: F=Stage2(H,P), HE+MIL both CEs",
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(variant: str, seed: int):
    assert variant in VARIANTS, f"unknown variant {variant}"
    stage2_cfg = dict(STAGE2_CFG)
    stage2_cfg["v7_fused_he_detach"] = VARIANTS[variant]
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
        "stage2_type": "he_residual_cross_v7",
        "encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": stage2_cfg,
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
    for variant, seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / variant
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((variant, seed, p, log_f))
        print(f"[launch] v7{variant} seed={seed} gpu={gpu}", flush=True)
    for variant, seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / variant / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] v7{variant} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] v7{variant} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']} "
              f"loss_he={r.get('v7_loss_he', float('nan')):.4f} "
              f"loss_fused={r.get('v7_loss_fused', float('nan')):.4f}",
              flush=True)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "—"


def _load_result(root, cond, seed):
    rpath = root / cond / f"seed{seed}" / "result.json"
    if not rpath.exists():
        return None
    return json.loads(rpath.read_text())


def generate_summary():
    rec = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for s in SEEDS:
            r = _load_result(RESULTS_ROOT, v, s)
            if r is None:
                continue
            rec[v][s] = {
                "auc": r["auc"], "acc": r["accuracy"], "f1": r["f1"],
                "sensitivity_tumor": r["sensitivity_tumor"],
                "specificity_tumor": r["specificity_tumor"],
                "best_epoch": r["best_epoch"],
                "v7_loss_he": r.get("v7_loss_he"),
                "v7_loss_fused": r.get("v7_loss_fused"),
                "init_hash": r.get("init_hash"),
                "epoch_history": r.get("v7_epoch_history", []),
            }
    # 同期 v3 基线（v6 轮的配对对照）
    v3 = {}
    for s in SEEDS:
        r = _load_result(V3_BASELINE_DIR, "", s)
        if r is None:
            r = _load_result(V3_BASELINE_DIR, "seed", s)  # fallback naming
        if r is not None:
            v3[s] = {"auc": r["auc"], "best_epoch": r["best_epoch"]}

    def _mean_std(v, key):
        vals = [rec[v][s][key] for s in SEEDS if s in rec[v]]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    def _paired_delta(base_auc: dict, target_auc: dict):
        common = sorted(set(target_auc) & set(base_auc))
        deltas = [target_auc[s] - base_auc[s] for s in common]
        return {
            "common_seeds": common,
            "per_seed_delta": {str(s): round(target_auc[s] - base_auc[s], 6)
                               for s in common},
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        }

    per_seed = {v: {str(s): rec[v][s]["auc"] for s in rec[v]} for v in VARIANTS}
    v3_auc = {str(s): v3[s]["auc"] for s in v3}

    auc_m = {v: _mean_std(v, "auc")[0] for v in VARIANTS}
    auc_s = {v: _mean_std(v, "auc")[1] for v in VARIANTS}

    # init hash cross-check：同 seed 的 A/B 应逐位一致
    hash_check = {}
    for s in SEEDS:
        if s in rec["A"] and s in rec["B"]:
            hash_check[str(s)] = (rec["A"][s]["init_hash"] == rec["B"][s]["init_hash"]
                                  and rec["A"][s]["init_hash"] is not None)

    summary = {
        "task": "stage2_he_residual_cross_v7",
        "evaluation_protocol": "Train / Val-as-Test",
        "protocol_note": ("official test set 在训练中被用于 checkpoint selection / early stopping，"
                          "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        "variants": {
            v: {
                "label": LABELS[v],
                "v7_fused_he_detach": VARIANTS[v],
                "per_seed_auc": per_seed[v],
                "mean_auc": auc_m[v], "std_auc": auc_s[v],
                "best_epoch": {str(s): rec[v][s]["best_epoch"] for s in rec[v]},
                "loss_he": {str(s): rec[v][s]["v7_loss_he"] for s in rec[v]},
                "loss_fused": {str(s): rec[v][s]["v7_loss_fused"] for s in rec[v]},
                "init_hash": {str(s): rec[v][s]["init_hash"] for s in rec[v]},
                "epoch_history": {str(s): rec[v][s]["epoch_history"] for s in rec[v]},
            }
            for v in VARIANTS
        },
        "v3_baseline": {
            "source": str(V3_BASELINE_DIR),
            "per_seed_auc": v3_auc,
            "mean_auc": float(np.mean(list(v3_auc.values()))) if v3_auc else None,
            "note": "同期 v3（v6 轮配对对照），复用不重训",
        },
        "paired_delta_auc": {
            "v7A_minus_v3": _paired_delta(v3_auc, per_seed["A"]),
            "v7B_minus_v3": _paired_delta(v3_auc, per_seed["B"]),
            "v7B_minus_v7A": _paired_delta(per_seed["A"], per_seed["B"]),
        },
        "init_hash_a_equals_b_per_seed": hash_check,
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
            "checkpoint_rule": "fused val AUC 选 checkpoint（与 v3 原规则一致）",
        },
        "v7_loss_objective": "CE(M(H), y) + CE(M(F), y)，两 loss 权重均 1；主预测 M(F)",
        "v7_param_groups": {
            "HE": "patch_to_emb[0] / rrt_he",
            "MIL": "mil",
            "PR2": "patch_to_emb[1] / rrt_ihc / cross_region_mod",
            "clip": "三组各自 clip_grad_norm_(max_norm=1.0)，无全模型联合裁剪",
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
        "acceptance": "A/B 必须超过同期 v3（不能只以超过较弱 v6 判成功）",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    # README
    L = []
    L.append("# HE Residual Cross v7（恢复融合梯度）—— Train / Val-as-Test 协议\n")
    L.append("**Evaluation protocol: Train / Val-as-Test**\n")
    L.append("> official test set 在训练中被用于 checkpoint selection / early stopping，"
             "因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。\n")
    L.append("v7 = v3 前向 + 训练期双 loss（同一 MIL，无 detach）："
             "`L = CE(M(H),y) + CE(M(F),y)`。A: `F=Stage2(H.detach(),P)`；"
             "B: `F=Stage2(H,P)`（主候选）。三参数组各自 clip。\n")

    L.append("## AUC（逐 seed；v3 为同期基线，复用）\n")
    L.append("| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    L.append(f"| v3 (baseline) | {_fmt(v3_auc.get('42'))} | {_fmt(v3_auc.get('123'))} "
             f"| {_fmt(v3_auc.get('456'))} | {_fmt(summary['v3_baseline']['mean_auc'])} |")
    for v in VARIANTS:
        L.append(f"| v7{v} | {_fmt(per_seed[v].get('42'))} | {_fmt(per_seed[v].get('123'))} "
                 f"| {_fmt(per_seed[v].get('456'))} | {_fmt(auc_m[v])} ± {_fmt(auc_s[v])} |")

    L.append("\n## paired ΔAUC\n")
    L.append("| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    for name, key in [("v7A − v3", "v7A_minus_v3"), ("v7B − v3", "v7B_minus_v3"),
                      ("v7B − v7A", "v7B_minus_v7A")]:
        pd_ = summary["paired_delta_auc"][key]
        L.append(f"| {name} | {_fmt(pd_['per_seed_delta'].get('42'))} | "
                 f"{_fmt(pd_['per_seed_delta'].get('123'))} | "
                 f"{_fmt(pd_['per_seed_delta'].get('456'))} | "
                 f"{_fmt(pd_['mean_delta'])} ± {_fmt(pd_['std_delta'])} |")

    L.append(f"\n## 初始化核对\n")
    L.append(f"A/B 同 seed init hash 一致: {hash_check}\n")

    L.append("\n## Stage2 config (v3 = v7)\n```\n" + json.dumps(STAGE2_CFG, indent=2) + "\n```\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] v3={summary['v3_baseline']['mean_auc']:.4f} | "
          f"A={auc_m['A']:.4f} | B={auc_m['B']:.4f}", flush=True)
    print(f"[summary] A−v3={summary['paired_delta_auc']['v7A_minus_v3']['mean_delta']:+.4f} | "
          f"B−v3={summary['paired_delta_auc']['v7B_minus_v3']['mean_delta']:+.4f} | "
          f"B−A={summary['paired_delta_auc']['v7B_minus_v7A']['mean_delta']:+.4f}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 5])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        generate_summary()
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    todo = []
    for v in VARIANTS:
        (RESULTS_ROOT / v).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = RESULTS_ROOT / v / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] v7{v} seed={seed} (complete)", flush=True)
                continue
            todo.append((v, seed, seed_dir))

    print(f"to-run: {len(todo)} job(s) | gpus={args.gpus} | force={args.force}", flush=True)

    if todo:
        cfg_by_key = {}
        for v, seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(v, seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[(v, seed)] = cfg_path

        flat = [(v, s, cfg_by_key[(v, s)]) for (v, s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(v, s, g, cfg_path)
                     for (v, s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    if any((RESULTS_ROOT / v / f"seed{s}" / "result.json").exists()
           for v in VARIANTS for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
