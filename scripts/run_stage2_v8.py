#!/usr/bin/env python3
"""HE Residual Cross **v8**（HE region queries 读取 PR patch memory）—— Train /
Val-as-Test 协议 paired driver。

恢复 v3 训练方式（§1）：仅 CE(fused_logits, y)，完整端到端反传，原全局梯度裁剪，
无 HE/PR 辅助 CE、无 detach、无分组裁剪。唯一核心变化在 cross 模块的 PR memory
mode（§2/§3，同 stage2_type='he_residual_cross_v8'、同一训练代码）：

    routed — 对照：PR 走原 Stage2 routing 压缩（G_PR·k=48 个 slot），≡ v3。
    patch  — 消融：route_norm_pr 与 attn_norm_pr 之间的 routing 加权 pooling
             替换为 identity，K/V 直接读同次 forward 的 Z_PR 全部有效 patch
             token（K = 有效 PR patch 数，物理压缩非 mask）。

保留：route_norm_pr→attn_norm_pr→Wk/Wv、HE routing/Wq、cosine τ=0.2、Wout、HE
dispatch、alpha=0.1、ABMIL、EMA centering（β=0.99，统计对象为有效 PR patch value
token 均值，train-only 每 forward 一次）。旧 PR phi 在新模式 forward 中不使用
（保留用于初始化/checkpoint 兼容，已记录）。

两个 mode 公共参数同 seed 初始化逐位一致（保存 init hash 核对）；slide 顺序 /
patch 采样固定配对；seeds=42/123/456。**基线 = 同期 v3（results/stage2_he_
residual_cross_v6/v3，复用不重训）；验收只看主预测是否超过同设置 v3（§7）。**

用法:
  python scripts/run_stage2_v8.py --gpus 0 1 5
  python scripts/run_stage2_v8.py --summary-only
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
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v8"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")

# 同期 v3 基线（v6 轮的配对对照，直接复用，不重训）
V3_BASELINE_DIR = PROJECT / "results" / "stage2_he_residual_cross_v6" / "v3"

# Stage1 各自 best —— 与 v3/v6/v7 完全一致，绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG；与 v3/v6/v7 同一份（v8 仅改 PR memory 读取方式）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
}

# mode name → pr_memory_mode
MODES = {
    "routed": "routed",   # control：≡ v3
    "patch": "patch",     # ablation：K/V = 全部有效 PR patch tokens
}
LABELS = {
    "routed": "v8-routed: PR routing pooling (≡v3)",
    "patch": "v8-patch: K/V = all valid Z_PR patch tokens",
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(mode: str, seed: int):
    assert mode in MODES, f"unknown mode {mode}"
    stage2_cfg = dict(STAGE2_CFG)
    stage2_cfg["pr_memory_mode"] = MODES[mode]
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
        "stage2_type": "he_residual_cross_v8",
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
    for mode, seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / mode
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((mode, seed, p, log_f))
        print(f"[launch] {mode} seed={seed} gpu={gpu}", flush=True)
    for mode, seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / mode / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {mode} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] {mode} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "—"


def _load_result(root, cond, seed):
    rpath = root / cond / f"seed{seed}" / "result.json"
    if not rpath.exists():
        return None
    return json.loads(rpath.read_text())


def generate_summary():
    rec = {m: {} for m in MODES}
    for m in MODES:
        for s in SEEDS:
            r = _load_result(RESULTS_ROOT, m, s)
            if r is None:
                continue
            rec[m][s] = {
                "auc": r["auc"], "acc": r["accuracy"], "f1": r["f1"],
                "sensitivity_tumor": r["sensitivity_tumor"],
                "specificity_tumor": r["specificity_tumor"],
                "best_epoch": r["best_epoch"],
                "init_hash": r.get("init_hash"),
                "pr_memory_mode": r.get("pr_memory_mode"),
            }
    v3 = {}
    for s in SEEDS:
        r = _load_result(V3_BASELINE_DIR, "", s)
        if r is not None:
            v3[s] = {"auc": r["auc"], "best_epoch": r["best_epoch"]}

    def _mean_std(m, key):
        vals = [rec[m][s][key] for s in SEEDS if s in rec[m]]
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

    per_seed = {m: {str(s): rec[m][s]["auc"] for s in rec[m]} for m in MODES}
    v3_auc = {str(s): v3[s]["auc"] for s in v3}
    auc_m = {m: _mean_std(m, "auc")[0] for m in MODES}
    auc_s = {m: _mean_std(m, "auc")[1] for m in MODES}

    hash_check = {}
    for s in SEEDS:
        if s in rec["routed"] and s in rec["patch"]:
            hash_check[str(s)] = (rec["routed"][s]["init_hash"]
                                  == rec["patch"][s]["init_hash"]
                                  and rec["routed"][s]["init_hash"] is not None)

    summary = {
        "task": "stage2_he_residual_cross_v8",
        "evaluation_protocol": "Train / Val-as-Test",
        "protocol_note": ("official test set 在训练中被用于 checkpoint selection / early stopping，"
                          "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        "modes": {
            m: {
                "label": LABELS[m],
                "pr_memory_mode": MODES[m],
                "per_seed_auc": per_seed[m],
                "mean_auc": auc_m[m], "std_auc": auc_s[m],
                "best_epoch": {str(s): rec[m][s]["best_epoch"] for s in rec[m]},
                "init_hash": {str(s): rec[m][s]["init_hash"] for s in rec[m]},
            }
            for m in MODES
        },
        "v3_baseline": {
            "source": str(V3_BASELINE_DIR),
            "per_seed_auc": v3_auc,
            "mean_auc": float(np.mean(list(v3_auc.values()))) if v3_auc else None,
            "note": "同期 v3（v6 轮配对对照），复用不重训",
        },
        "paired_delta_auc": {
            "routed_minus_v3": _paired_delta(v3_auc, per_seed["routed"]),
            "patch_minus_v3": _paired_delta(v3_auc, per_seed["patch"]),
            "patch_minus_routed": _paired_delta(per_seed["routed"], per_seed["patch"]),
        },
        "init_hash_routed_equals_patch_per_seed": hash_check,
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
            "training": "v3 恢复：仅 CE(fused)，全局梯度裁剪，无 detach/分组/辅助 CE",
        },
        "structural_change": ("patch 模式：route_norm_pr→attn_norm_pr 之间的 routing "
                              "加权 pooling 替换为 identity，K/V = Z_PR 全部有效 patch "
                              "token（K=有效数，物理压缩）；phi_pr 保留但不参与 forward"),
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
        "acceptance": "patch 必须超过同设置 v3 的主预测（不能以超过弱 v7 判成功）",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    L = []
    L.append("# HE Residual Cross v8（HE region queries 读取 PR patch memory）—— Train / Val-as-Test 协议\n")
    L.append("**Evaluation protocol: Train / Val-as-Test**\n")
    L.append("> official test set 在训练中被用于 checkpoint selection / early stopping，"
             "因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。\n")
    L.append("v8 = 恢复 v3 训练（单 fused CE、全局裁剪）+ PR memory 结构对照："
             "routed（PR routing 压缩，≡v3）vs patch（跳过 PR routing pooling，"
             "K/V = 全部有效 Z_PR patch token）。\n")

    L.append("## AUC（逐 seed；v3 为同期基线，复用）\n")
    L.append("| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    L.append(f"| v3 (baseline) | {_fmt(v3_auc.get('42'))} | {_fmt(v3_auc.get('123'))} "
             f"| {_fmt(v3_auc.get('456'))} | {_fmt(summary['v3_baseline']['mean_auc'])} |")
    for m in MODES:
        L.append(f"| v8-{m} | {_fmt(per_seed[m].get('42'))} | {_fmt(per_seed[m].get('123'))} "
                 f"| {_fmt(per_seed[m].get('456'))} | {_fmt(auc_m[m])} ± {_fmt(auc_s[m])} |")

    L.append("\n## paired ΔAUC\n")
    L.append("| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    for name, key in [("routed − v3", "routed_minus_v3"),
                      ("patch − v3", "patch_minus_v3"),
                      ("patch − routed", "patch_minus_routed")]:
        pd_ = summary["paired_delta_auc"][key]
        L.append(f"| {name} | {_fmt(pd_['per_seed_delta'].get('42'))} | "
                 f"{_fmt(pd_['per_seed_delta'].get('123'))} | "
                 f"{_fmt(pd_['per_seed_delta'].get('456'))} | "
                 f"{_fmt(pd_['mean_delta'])} ± {_fmt(pd_['std_delta'])} |")

    L.append(f"\n## 初始化核对\n")
    L.append(f"routed/patch 同 seed init hash 一致: {hash_check}\n")

    L.append("\n## Stage2 config (v3 = v8)\n```\n" + json.dumps(STAGE2_CFG, indent=2) + "\n```\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] v3={summary['v3_baseline']['mean_auc']:.4f} | "
          f"routed={auc_m['routed']:.4f} | patch={auc_m['patch']:.4f}", flush=True)
    print(f"[summary] routed−v3={summary['paired_delta_auc']['routed_minus_v3']['mean_delta']:+.4f} | "
          f"patch−v3={summary['paired_delta_auc']['patch_minus_v3']['mean_delta']:+.4f} | "
          f"patch−routed={summary['paired_delta_auc']['patch_minus_routed']['mean_delta']:+.4f}",
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
    for m in MODES:
        (RESULTS_ROOT / m).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = RESULTS_ROOT / m / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] {m} seed={seed} (complete)", flush=True)
                continue
            todo.append((m, seed, seed_dir))

    print(f"to-run: {len(todo)} job(s) | gpus={args.gpus} | force={args.force}", flush=True)

    if todo:
        cfg_by_key = {}
        for m, seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(m, seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[(m, seed)] = cfg_path

        flat = [(m, s, cfg_by_key[(m, s)]) for (m, s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(m, s, g, cfg_path)
                     for (m, s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    if any((RESULTS_ROOT / m / f"seed{s}" / "result.json").exists()
           for m in MODES for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
