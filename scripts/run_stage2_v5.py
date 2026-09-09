#!/usr/bin/env python3
"""HE Residual Cross **v5** —— 固定 split（216/54）内部验证配对对照 driver。

两个条件 × 3 seeds（42/123/456），都用 `stage2_type='he_residual_cross_v5'`：

  Condition "v5_beta0"    pr_value_aux_weight = 0.0   （= v3 β=0 对照组）
  Condition "v5_beta01"   pr_value_aux_weight = 0.1   （= v5 β=0.1 实验组）

二者唯一的差异是训练期辅助分类监督的权重 `pr_value_aux_weight`。因为两个条件
共用同一个 v5 模块、同一份 seed，公共模块（两支 RRT、cross QKV、prototype、ABMIL）
在随机初始化时**逐位一致**，满足 §5「公共模块使用同一份随机初始权重」的要求。
aux head（`aux_attn`/`aux_classifier`）在两个条件下都注册进 optimizer，但 β=0 时
aux CE 项为 0，不产生反向梯度 ⇒ 主路径训练动态与 v3 严格一致（Test 2 已验证
β=0 时 v5 fused output 与 v3 逐位相同）。

监督协议（§5）：
  - 训练集  = fixed_split/train.csv（216）
  - 验证集  = fixed_split/val.csv（54）——用于 checkpoint selection / early stopping
  - 测试集  = c16_test_labels.csv（129）——仅作为开发背景，**不能**称独立测试收益
    （此前的 val-as-test 协议已污染该 129 集；本轮评估重点为内部验证集配对对照）。

训练超参统一：LR=1e-4、weight_decay=1e-5、cosine、patience=10、num_epochs=80、
batch=1、dropout=0.25、abmil_hidden_dim=256、aux_loss_weight=0.0（旧路径禁用）、
modality_dropout=0、label_smoothing=0。两支 RRT 从头端到端训练，无预训练、无冻结。

断点续跑：result.json + test_predictions.csv 完整即跳过；--force 覆盖。

用法:
  python scripts/run_stage2_v5.py --gpus 0 1 2
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
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/fixed_split/train.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/fixed_split/val.csv")
MAX_PATCHES = 2500
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v5"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_stage2_he_residual_seed.py")

# Stage1 各自 best —— 与 HE-only / v1 / v2 / v3 / v4 完全一致，绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG；v5 = v3 全部配置 + aux head（不新增 main 路径 knob）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
    "aux_hidden_dim": 64, "aux_dropout": 0.1,
}

STAGE2_TYPE = "he_residual_cross_v5"

# 条件：目录名 → pr_value_aux_weight（β）
CONDITIONS = {
    "v5_beta0": 0.0,     # v3 β=0 对照（aux head 存在但无监督）
    "v5_beta01": 0.1,    # v5 β=0.1 实验（aux CE 监督 PR value memory）
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(condition: str, seed: int):
    beta = CONDITIONS[condition]

    training = {
        "batch_size": 1, "num_epochs": 80,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": {"type": "cosine"},
        "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
        "kd_enabled": False, "modality_dropout": 0.0,
        "aux_loss_weight": 0.0,          # 旧 aux path 禁用（v5 走 pr_value_aux_weight）
        "pr_value_aux_weight": beta,     # v5 辅助分类监督权重
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
        "stage2_type": STAGE2_TYPE,
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
            "evaluation_protocol": "fixed_split_internal_val",
            "train": 216, "val": 54, "dev_test": 129,
            "note": "checkpoint selection / early stopping 在内部 val(54) 上进行；"
                    "129 集为开发背景（曾作为 val-as-test 被早期版本污染），"
                    "不构成独立测试收益。",
        },
    }


def is_complete(seed_dir: Path) -> bool:
    r = seed_dir / "result.json"
    p = seed_dir / "test_predictions.csv"
    if not (r.exists() and p.exists()):
        return False
    try:
        return "test_auc" in json.loads(r.read_text())
    except Exception:
        return False


def run_wave(pairs):
    procs = []
    for condition, seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / condition
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((condition, seed, p, log_f))
        print(f"[launch] {condition} seed={seed} gpu={gpu}", flush=True)
    for condition, seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / condition / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {condition} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] {condition} seed={seed} val={r['best_val_auc']:.4f} "
              f"test={r['test_auc']:.4f} acc={r['test_acc']:.4f} "
              f"f1={r['test_f1']:.4f} epoch={r['best_epoch']}", flush=True)


def _fmt(v):
    return f"{v:.4f}"


def _mean_std(vals):
    vals = list(vals)
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _paired_delta(target: dict, base: dict):
    common = sorted(set(target) & set(base))
    deltas = [target[s] - base[s] for s in common]
    return {
        "common_seeds": common,
        "per_seed_delta": {str(s): round(target[s] - base[s], 6) for s in common},
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "std_delta": float(np.std(deltas)) if deltas else None,
    }


def generate_summary():
    cond_aucs, cond_val, cond_epoch = {}, {}, {}
    for condition in CONDITIONS:
        aucs, val_aucs, epochs = [], [], []
        out_root = RESULTS_ROOT / condition
        for s in SEEDS:
            rpath = out_root / f"seed{s}" / "result.json"
            if not rpath.exists():
                continue
            r = json.loads(rpath.read_text())
            aucs.append(r["test_auc"])
            val_aucs.append(r["best_val_auc"])
            epochs.append(r["best_epoch"])
        cond_aucs[condition] = {s: a for s, a in zip(SEEDS, aucs)}
        cond_val[condition] = {s: a for s, a in zip(SEEDS, val_aucs)}
        cond_epoch[condition] = {s: e for s, e in zip(SEEDS, epochs)}

    def _meta(name):
        auc_m, auc_s = _mean_std(cond_aucs[name].values())
        val_m, val_s = _mean_std(cond_val[name].values())
        return {
            "pr_value_aux_weight": CONDITIONS[name],
            "per_seed_dev_test_auc": cond_aucs[name],
            "mean_dev_test_auc": auc_m, "std_dev_test_auc": auc_s,
            "per_seed_val_auc": cond_val[name],
            "mean_val_auc": val_m, "std_val_auc": val_s,
            "best_epoch": cond_epoch[name],
        }

    summary = {
        "task": "stage2_he_residual_cross_v5",
        "stage2_type": STAGE2_TYPE,
        "conditions": {c: _meta(c) for c in CONDITIONS},
        "primary_paired_delta": {
            "v5_beta01_minus_v5_beta0": _paired_delta(cond_aucs["v5_beta01"],
                                                      cond_aucs["v5_beta0"]),
            "val_v5_beta01_minus_v5_beta0": _paired_delta(cond_val["v5_beta01"],
                                                          cond_val["v5_beta0"]),
        },
        "protocol": {
            "evaluation_protocol": "fixed_split_internal_val",
            "train": 216, "val": 54, "dev_test": 129,
            "monitor": "val_auc", "mode": "max", "patience": 10,
            "num_epochs": 80, "scheduler": "cosine",
            "lr": 1e-4, "unified_lr": True, "weight_decay": 1e-5,
            "sampling": "random", "train_per_epoch": True, "val_per_epoch": False,
            "batch_size": 1, "num_workers": 2, "max_patches": 2500,
            "dropout": 0.25, "abmil_hidden_dim": 256,
            "label_smoothing": 0.0, "aux_loss_weight": 0.0,
            "modality_dropout": 0.0, "kd_enabled": False,
            "model_seed": SEEDS, "sampling_seed": SEEDS,
            "note": "129 集为开发背景（早期 val-as-test 已污染），"
                    "primary 结论以内部 val(54) 配对对照为准。",
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "v5_change": ("在 v3 的 merged PR value memory（V_tilde）上接轻量辅助 ABMIL，"
                      "训练期叠加 β·CE(pr_value_logits, y)；主路径推理不变。"),
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    L = []
    L.append("# HE Residual Cross v5 —— 固定 split 内部验证配对对照\n")
    L.append("**协议：fixed_split（216/54）+ dev test(129，仅背景）**\n")
    L.append("> checkpoint selection / early stopping 在内部 val(54) 上进行；"
             "129 集仅作开发背景（早期 val-as-test 已污染），不能称独立测试收益。\n")
    L.append("v5 = v3（cosine τ=0.2 + bias-free QKV + dataset-prototype PR value centering）"
             "+ 训练期对 merged PR value memory 的辅助 ABMIL 分类监督（β=0.1）。"
             "对照组 v5_beta0 用同一模块、β=0，与 v3 主路径逐位一致。\n")

    L.append("## dev test AUC（129，仅背景）\n")
    L.append("| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |")
    L.append("|---|---|---|---|")
    for s in SEEDS:
        a = cond_aucs["v5_beta0"].get(s)
        b = cond_aucs["v5_beta01"].get(s)
        d = b - a if (a is not None and b is not None) else None
        L.append(f"| {s} | {_fmt(a) if a is not None else '—'} | "
                 f"{_fmt(b) if b is not None else '—'} | "
                 f"{_fmt(d) if d is not None else '—'} |")
    m0, s0 = _mean_std(cond_aucs["v5_beta0"].values())
    m1, s1 = _mean_std(cond_aucs["v5_beta01"].values())
    L.append(f"| **mean** | {_fmt(m0) if m0 is not None else '—'} | "
             f"{_fmt(m1) if m1 is not None else '—'} | "
             f"{_fmt(m1 - m0) if (m0 is not None and m1 is not None) else '—'} |")

    L.append("\n## 内部 val AUC（54，primary）\n")
    L.append("| seed | v5_beta0 (β=0) | v5_beta01 (β=0.1) | Δ |")
    L.append("|---|---|---|---|")
    for s in SEEDS:
        a = cond_val["v5_beta0"].get(s)
        b = cond_val["v5_beta01"].get(s)
        d = b - a if (a is not None and b is not None) else None
        L.append(f"| {s} | {_fmt(a) if a is not None else '—'} | "
                 f"{_fmt(b) if b is not None else '—'} | "
                 f"{_fmt(d) if d is not None else '—'} |")
    vm0, vs0 = _mean_std(cond_val["v5_beta0"].values())
    vm1, vs1 = _mean_std(cond_val["v5_beta01"].values())
    L.append(f"| **mean** | {_fmt(vm0) if vm0 is not None else '—'} | "
             f"{_fmt(vm1) if vm1 is not None else '—'} | "
             f"{_fmt(vm1 - vm0) if (vm0 is not None and vm1 is not None) else '—'} |")

    L.append("\n## Stage2 config (v5)\n```\n" + json.dumps(STAGE2_CFG, indent=2) + "\n```\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] dev_test: beta0={_fmt(m0) if m0 is not None else '—'} "
          f"beta01={_fmt(m1) if m1 is not None else '—'} Δ={_fmt(m1 - m0) if (m0 is not None and m1 is not None) else '—'}",
          flush=True)
    print(f"[summary] val: beta0={_fmt(vm0) if vm0 is not None else '—'} "
          f"beta01={_fmt(vm1) if vm1 is not None else '—'} Δ={_fmt(vm1 - vm0) if (vm0 is not None and vm1 is not None) else '—'}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        generate_summary()
        return

    todo = []
    for condition in CONDITIONS:
        out_root = RESULTS_ROOT / condition
        out_root.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = out_root / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] {condition} seed={seed} (complete)", flush=True)
                continue
            todo.append((condition, seed, seed_dir))

    print(f"to-run: {len(todo)} seed(s) | gpus={args.gpus} | force={args.force}",
          flush=True)

    if todo:
        cfg_by_key = {}
        for condition, seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(condition, seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[(condition, seed)] = cfg_path

        flat = [(c, s, cfg_by_key[(c, s)]) for (c, s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(c, s, g, cfg_path)
                     for (c, s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    if any((RESULTS_ROOT / c / f"seed{s}" / "result.json").exists()
           for c in CONDITIONS for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
