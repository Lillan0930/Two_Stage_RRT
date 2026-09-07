#!/usr/bin/env python3
"""HE Residual Cross **v2** —— Train / Val-as-Test 协议 driver（3 Model seeds）。

只训练 v2 的 3 个 model seed（42/123/456）。HE-only 与 v1 he_residual_cross 的
结果**复用**上一轮 `results/stage2_he_residual_cross_val_as_test/summary.json`
（HE-only 0.8344/0.8615/0.8143；v1 0.8406/0.8908/0.8551），本轮不重训。

v2 与 v1 唯一差异（模块层面）：
    cosine cross-attention (τ=0.2) + PR value centering + 全部 cross 投影 bias=False。
其余（两个独立 RRT、独立 HE/PR routing、Q=HE、K/V=PR、HE identity 残差、
residual_scale=0.1、ABMIL、CE-only、Train/Val-as-Test 协议、LR=1e-4、随机初始化
端到端）完全一致。

用法:
  python scripts/run_stage2_v2.py --gpus 0 1 4
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
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_v2"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")

# Stage1 各自 best —— 与 v1 / HE-only 完全一致，绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG；v2 额外 temperature=0.2 / value_centering=True /
# qkv_bias=False（cosine + centering + bias-free 为 v2 三改动）。
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "value_centering": True, "residual_scale": 0.1,
    "disable_cross": False,
}

CONDITION = "he_residual_cross_v2"
LABEL = "HE Residual Cross v2"

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(seed: int):
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
        "stage2_type": CONDITION,
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
    for seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / CONDITION
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((seed, p, log_f))
        print(f"[launch] {CONDITION} seed={seed} gpu={gpu}", flush=True)
    for seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / CONDITION / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {CONDITION} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] {CONDITION} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def _fmt(v):
    return f"{v:.4f}"


def generate_summary():
    rec = {"auc": {}, "acc": {}, "f1": {}, "sens_t": {}, "spec_t": {},
           "sens_m": {}, "spec_m": {}, "epoch": {}, "lrs": None}
    for s in SEEDS:
        rpath = RESULTS_ROOT / CONDITION / f"seed{s}" / "result.json"
        if not rpath.exists():
            continue
        r = json.loads(rpath.read_text())
        rec["auc"][s] = r["auc"]
        rec["acc"][s] = r["accuracy"]
        rec["f1"][s] = r["f1"]
        rec["sens_t"][s] = r["sensitivity_tumor"]
        rec["spec_t"][s] = r["specificity_tumor"]
        rec["sens_m"][s] = r["sensitivity_macro"]
        rec["spec_m"][s] = r["specificity_macro"]
        rec["epoch"][s] = r["best_epoch"]
        if rec["lrs"] is None:
            rec["lrs"] = r.get("actual_optimizer_lrs")

    def _mean_std(d):
        vals = list(d.values())
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    auc_m, auc_s = _mean_std(rec["auc"])
    meta = {
        "label": LABEL,
        "stage2_type": CONDITION,
        "per_seed_auc": {str(s): v for s, v in rec["auc"].items()},
        "mean_auc": auc_m, "std_auc": auc_s,
        "mean_acc": _mean_std(rec["acc"])[0], "std_acc": _mean_std(rec["acc"])[1],
        "mean_f1": _mean_std(rec["f1"])[0], "std_f1": _mean_std(rec["f1"])[1],
        "mean_sensitivity_tumor": _mean_std(rec["sens_t"])[0],
        "mean_specificity_tumor": _mean_std(rec["spec_t"])[0],
        "mean_sensitivity_macro": _mean_std(rec["sens_m"])[0],
        "mean_specificity_macro": _mean_std(rec["spec_m"])[0],
        "per_seed_acc": {str(s): v for s, v in rec["acc"].items()},
        "per_seed_f1": {str(s): v for s, v in rec["f1"].items()},
        "per_seed_sensitivity_tumor": {str(s): v for s, v in rec["sens_t"].items()},
        "per_seed_specificity_tumor": {str(s): v for s, v in rec["spec_t"].items()},
        "best_epoch": {str(s): v for s, v in rec["epoch"].items()},
        "actual_optimizer_lrs": rec["lrs"],
    }

    # reuse baselines from the previous round (do NOT retrain)
    prev_summary = json.loads(
        (PROJECT / "results" / "stage2_he_residual_cross_val_as_test" / "summary.json")
        .read_text())
    he_only_auc = prev_summary["conditions"]["he_only"]["per_seed_auc"]
    v1_auc = prev_summary["conditions"]["he_residual_cross"]["per_seed_auc"]

    def paired_delta(target_auc, base_auc):
        common = sorted(set(target_auc) & set(base_auc))
        deltas = [target_auc[s] - base_auc[s] for s in common]
        return {
            "common_seeds": common,
            "per_seed_delta": {str(s): round(target_auc[s] - base_auc[s], 6) for s in common},
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        }

    summary = {
        "task": "stage2_he_residual_cross_v2",
        "evaluation_protocol": "Train / Val-as-Test",
        "protocol_note": ("official test set 在训练中被用于 checkpoint selection / early stopping，"
                          "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        "condition": meta,
        "paired_delta_auc": {
            "v2_minus_he": paired_delta(meta["per_seed_auc"], he_only_auc),
            "v2_minus_v1": paired_delta(meta["per_seed_auc"], v1_auc),
        },
        "reused_baselines": {
            "he_only": {"per_seed_auc": he_only_auc},
            "he_residual_cross_v1": {"per_seed_auc": v1_auc},
            "source": "results/stage2_he_residual_cross_val_as_test/summary.json",
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
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    L = []
    L.append("# HE Residual Cross v2 —— Train / Val-as-Test 协议\n")
    L.append("**Evaluation protocol: Train / Val-as-Test**\n")
    L.append("> official test set 在训练中被用于 checkpoint selection / early stopping，"
             "因此属既定 val-as-test 开发协议，非严格意义的独立 untouched test。\n")
    L.append("v2 = cosine cross-attention (τ=0.2) + PR value centering + bias-free QKV。"
             "HE-only 与 v1 结果复用上一轮，不重训。\n")

    L.append("## AUC（v2 逐 seed；HE-only / v1 复用）\n")
    L.append("| Model | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    L.append(f"| HE-only (reuse) | {_fmt(he_only_auc['42'])} | {_fmt(he_only_auc['123'])} "
             f"| {_fmt(he_only_auc['456'])} | 0.8367 ± 0.0193 |")
    L.append(f"| HE Residual Cross v1 (reuse) | {_fmt(v1_auc['42'])} | {_fmt(v1_auc['123'])} "
             f"| {_fmt(v1_auc['456'])} | 0.8622 ± 0.0211 |")
    L.append(f"| {LABEL} | {_fmt(meta['per_seed_auc'].get('42'))} | "
             f"{_fmt(meta['per_seed_auc'].get('123'))} | {_fmt(meta['per_seed_auc'].get('456'))} | "
             f"{_fmt(auc_m)} ± {_fmt(auc_s)} |")

    L.append("\n## paired ΔAUC\n")
    L.append("| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    for name, key in [("v2 − HE", "v2_minus_he"), ("v2 − v1", "v2_minus_v1")]:
        pd_ = summary["paired_delta_auc"][key]
        p42 = pd_["per_seed_delta"].get("42")
        p123 = pd_["per_seed_delta"].get("123")
        p456 = pd_["per_seed_delta"].get("456")
        m, s = pd_["mean_delta"], pd_["std_delta"]
        L.append(f"| {name} | {_fmt(p42) if p42 is not None else '—'} | "
                 f"{_fmt(p123) if p123 is not None else '—'} | "
                 f"{_fmt(p456) if p456 is not None else '—'} | "
                 f"{_fmt(m) if m is not None else '—'} ± {_fmt(s) if s is not None else '—'} |")

    L.append("\n## Stage2 config (v2)\n```\n" + json.dumps(STAGE2_CFG, indent=2) + "\n```\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] v2 mean AUC = {_fmt(auc_m)}", flush=True)
    print(f"[summary] v2 − HE = {_fmt(summary['paired_delta_auc']['v2_minus_he']['mean_delta'])} | "
          f"v2 − v1 = {_fmt(summary['paired_delta_auc']['v2_minus_v1']['mean_delta'])}", flush=True)
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

    out_root = RESULTS_ROOT / CONDITION
    out_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for seed in SEEDS:
        seed_dir = out_root / f"seed{seed}"
        if is_complete(seed_dir) and not args.force:
            print(f"[skip] {CONDITION} seed={seed} (complete)", flush=True)
            continue
        todo.append((seed, seed_dir))

    print(f"to-run: {len(todo)} seed(s) | gpus={args.gpus} | force={args.force}",
          flush=True)

    if todo:
        cfg_by_key = {}
        for seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[seed] = cfg_path

        flat = [(s, cfg_by_key[s]) for (s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(s, g, cfg_path)
                     for (s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    if any((out_root / f"seed{s}" / "result.json").exists() for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
