#!/usr/bin/env python3
"""Stage2 改造 —— Train / Val-as-Test 协议 driver（3 条件 × 3 Model seeds）。

把最新的 HE-only / Joint CR-MSA / HE Residual Cross 放回同一个
**Train / Val-as-Test** 协议下重跑，得到与旧实验可横向比较的结果。

数据划分（与旧 val-as-test 协议一致）：
    train = C16 official train（270 WSIs，c16_train_labels.csv）
    val-as-test = C16 official test（129 WSIs，c16_test_labels.csv）
    不再从 train 中额外划分 validation。

val-as-test 同时用于每个 epoch 验证 / early stopping / best checkpoint 选择 /
最终 AUC·ACC·F1·Sens·Spec 报告。

三条件（各 Model seed = 42 / 123 / 456）：
    1. he_only          HE-only baseline（官方 RRTEncoder → ABMIL，无 Stage2）
    2. staining_msa     Joint CR-MSA（stage2_type = staining_msa，对称双边）
    3. he_residual_cross  HE Residual Cross（stage2_type = he_residual_cross）

统一协议：LR=1e-4、weight_decay=1e-5、batch=1、max_patches=2500、epochs=80、
cosine、patience=10、aux_loss_weight=0、kd=False、modality_dropout=0。
HE/PR RRT + Stage2 + ABMIL 全部随机初始化、从头端到端训练、不冻结、不加载旧 ckpt。
沿用已验证的 sampler（train per_epoch=True / val per_epoch=False）与 epoch 更新逻辑。

Seed 语义（明确标注，避免混淆）：
    model_seed     = environment.seed   → set_seed()：torch/cuda/numpy + 模型随机初始化
    sampling_seed  = data.sample_seed    → C16MultimodalDataset 的 patch 采样 RNG
    shuffle_seed   → DataLoader shuffle=True 使用 set_seed(model_seed) 后的 torch RNG

断点续跑：result.json + test_predictions.csv 完整即跳过；--force 覆盖。

用法:
  python scripts/run_stage2_val_as_test.py --gpus 0 1 4 5
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
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross_val_as_test"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")

# Stage1 各自 best —— 绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG（he_residual_cross 复用同一结构参数 + residual_scale=0.1）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}

# 三条件：目录名 → (stage2_type 或 None=HE-only, 是否 HE-only, 报告标签)
CONDITIONS = {
    "he_only": (None, True, "HE-only"),
    "staining_msa": ("staining_msa", False, "Joint CR-MSA"),
    "he_residual_cross": ("he_residual_cross", False, "HE Residual Cross"),
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(condition: str, seed: int):
    stage2_type, he_only, _label = CONDITIONS[condition]
    modalities = ["HE"] if he_only else MODALITIES
    dir_mapping = {"HE": "C16_HE_features"} if he_only else DIR_MAPPING

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
    # 统一 LR —— 不写 lr_stage1 / lr_stage2（create_optimizer_scheduler 走 unified 分支）

    model = {
        "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
        "region_num": 4, "n_layers": 2, "n_heads": 4,
        "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
        "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
        "crmsa_heads": 8, "crmsa_mlp": False,
        "fusion_type": "two_stage_region", "fusion_stage": "middle",
        "use_gated_fusion": False, "abmil_hidden_dim": 256,
        "use_mclc": False, "aggregate_modalities": True,
    }
    if not he_only:
        model["stage2_type"] = stage2_type
        model["encoder_cfg"] = STAGE1_ENCODER_CFG
        s2 = dict(STAGE2_CFG)
        if stage2_type == "he_residual_cross":
            s2["residual_scale"] = 0.1
            s2["disable_cross"] = False
        model["stage2_cfg"] = s2

    return {
        "data": {
            "dataset_type": "c16", "modalities": modalities,
            "dir_mapping": dir_mapping,
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
        print(f"[done] {condition} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def _fmt(v):
    return f"{v:.4f}"


def _collect(condition):
    out_root = RESULTS_ROOT / condition
    rec = {"auc": {}, "acc": {}, "f1": {}, "sens_t": {}, "spec_t": {},
           "sens_m": {}, "spec_m": {}, "epoch": {}, "lrs": None}
    for s in SEEDS:
        rpath = out_root / f"seed{s}" / "result.json"
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
    return rec


def _mean_std(d):
    vals = list(d.values())
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def generate_summary():
    rec = {c: _collect(c) for c in CONDITIONS}
    meta = {}
    for c in CONDITIONS:
        auc_m, auc_s = _mean_std(rec[c]["auc"])
        acc_m, acc_s = _mean_std(rec[c]["acc"])
        f1_m, f1_s = _mean_std(rec[c]["f1"])
        sens_t_m, _ = _mean_std(rec[c]["sens_t"])
        spec_t_m, _ = _mean_std(rec[c]["spec_t"])
        sens_m_m, _ = _mean_std(rec[c]["sens_m"])
        spec_m_m, _ = _mean_std(rec[c]["spec_m"])
        meta[c] = {
            "label": CONDITIONS[c][2],
            "stage2_type": CONDITIONS[c][0],
            "per_seed_auc": {str(s): v for s, v in rec[c]["auc"].items()},
            "mean_auc": auc_m, "std_auc": auc_s,
            "mean_acc": acc_m, "std_acc": acc_s,
            "mean_f1": f1_m, "std_f1": f1_s,
            "mean_sensitivity_tumor": sens_t_m, "mean_specificity_tumor": spec_t_m,
            "mean_sensitivity_macro": sens_m_m, "mean_specificity_macro": spec_m_m,
            "per_seed_acc": {str(s): v for s, v in rec[c]["acc"].items()},
            "per_seed_f1": {str(s): v for s, v in rec[c]["f1"].items()},
            "per_seed_sensitivity_tumor": {str(s): v for s, v in rec[c]["sens_t"].items()},
            "per_seed_specificity_tumor": {str(s): v for s, v in rec[c]["spec_t"].items()},
            "best_epoch": {str(s): v for s, v in rec[c]["epoch"].items()},
            "actual_optimizer_lrs": rec[c]["lrs"],
        }

    he, msa, rcx = "he_only", "staining_msa", "he_residual_cross"

    def paired_delta(a, b):
        da, db = rec[a]["auc"], rec[b]["auc"]
        common = sorted(set(da) & set(db))
        deltas = [da[s] - db[s] for s in common]
        return {
            "common_seeds": common,
            "per_seed_delta": {str(s): round(da[s] - db[s], 6) for s in common},
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        }

    summary = {
        "task": "stage2_he_residual_cross_val_as_test",
        "evaluation_protocol": "Train / Val-as-Test",
        "protocol_note": ("official test set 在训练中被用于 checkpoint selection / early stopping，"
                          "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        "conditions": {c: meta[c] for c in CONDITIONS},
        "paired_delta_auc": {
            "joint_minus_he": paired_delta(msa, he),
            "residual_cross_minus_he": paired_delta(rcx, he),
            "residual_cross_minus_joint": paired_delta(rcx, msa),
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

    # ---- README.md ----
    L = []
    L.append("# Stage2 改造 —— Train / Val-as-Test 协议\n")
    L.append("**Evaluation protocol: Train / Val-as-Test**\n")
    L.append("> official test set 在训练过程中被用于 checkpoint selection / early stopping，"
             "因此这里的结果属于既定 val-as-test 开发协议，不作为严格意义上的独立 untouched test。\n")
    L.append("3 条件 × 3 Model seeds（42/123/456），train=270 / val-as-test=129，"
             "统一 LR=1e-4，全部从头随机初始化，无 pretrained。\n")

    L.append("## AUC 表格\n")
    L.append("| Model | Seed42 AUC | Seed123 AUC | Seed456 AUC | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    for c in CONDITIONS:
        label = CONDITIONS[c][2]
        a42 = rec[c]["auc"].get(42)
        a123 = rec[c]["auc"].get(123)
        a456 = rec[c]["auc"].get(456)
        m, s = meta[c]["mean_auc"], meta[c]["std_auc"]
        L.append(f"| {label} | {_fmt(a42) if a42 is not None else '—'} | "
                 f"{_fmt(a123) if a123 is not None else '—'} | "
                 f"{_fmt(a456) if a456 is not None else '—'} | "
                 f"{_fmt(m) if m is not None else '—'} ± {_fmt(s) if s is not None else '—'} |")

    L.append("\n## paired ΔAUC（逐 seed + 均值）\n")
    L.append("| Δ | Seed42 | Seed123 | Seed456 | Mean ± Std |")
    L.append("|---|---|---|---|---|")
    for name, (a, b) in {
        "Joint − HE": (msa, he),
        "ResidualCross − HE": (rcx, he),
        "ResidualCross − Joint": (rcx, msa),
    }.items():
        pd_ = paired_delta(a, b)
        p42 = pd_["per_seed_delta"].get("42")
        p123 = pd_["per_seed_delta"].get("123")
        p456 = pd_["per_seed_delta"].get("456")
        m, s = pd_["mean_delta"], pd_["std_delta"]
        L.append(f"| {name} | {_fmt(p42) if p42 is not None else '—'} | "
                 f"{_fmt(p123) if p123 is not None else '—'} | "
                 f"{_fmt(p456) if p456 is not None else '—'} | "
                 f"{_fmt(m) if m is not None else '—'} ± {_fmt(s) if s is not None else '—'} |")

    L.append("\n## ACC / F1 / Sensitivity / Specificity（均值，val-as-test）\n")
    L.append("| Model | ACC | F1 | Sens(tumor) | Spec(tumor) | Sens(macro) | Spec(macro) |")
    L.append("|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        label = CONDITIONS[c][2]
        L.append(f"| {label} | {_fmt(meta[c]['mean_acc']) if meta[c]['mean_acc'] is not None else '—'} | "
                 f"{_fmt(meta[c]['mean_f1']) if meta[c]['mean_f1'] is not None else '—'} | "
                 f"{_fmt(meta[c]['mean_sensitivity_tumor']) if meta[c]['mean_sensitivity_tumor'] is not None else '—'} | "
                 f"{_fmt(meta[c]['mean_specificity_tumor']) if meta[c]['mean_specificity_tumor'] is not None else '—'} | "
                 f"{_fmt(meta[c]['mean_sensitivity_macro']) if meta[c]['mean_sensitivity_macro'] is not None else '—'} | "
                 f"{_fmt(meta[c]['mean_specificity_macro']) if meta[c]['mean_specificity_macro'] is not None else '—'} |")

    L.append("\n## 逐 seed 详细指标\n")
    L.append("```")
    L.append(json.dumps({c: {k: meta[c][k] for k in
                             ("per_seed_auc", "per_seed_acc", "per_seed_f1",
                              "per_seed_sensitivity_tumor", "per_seed_specificity_tumor",
                              "best_epoch", "actual_optimizer_lrs")}
                         for c in CONDITIONS}, indent=2))
    L.append("```\n")

    L.append("## 实际实例化配置 / 实际 optimizer LR\n")
    L.append("- Stage1 encoder（HE/PR 各自 best，绝对未改）：")
    L.append("```")
    L.append(json.dumps(STAGE1_ENCODER_CFG, indent=2))
    L.append("```")
    L.append("- Stage2（region_num=4, crmsa_k=3, heads=8, dropout=0.1, epeg=False, ffn=False；"
             "he_residual_cross 另加 residual_scale=0.1, disable_cross=False）：")
    L.append("```")
    L.append(json.dumps(STAGE2_CFG, indent=2))
    L.append("```")
    L.append("- 实际 optimizer LR（每活跃模块，从 result.json 读取）：")
    L.append("```")
    L.append(json.dumps({c: meta[c]["actual_optimizer_lrs"] for c in CONDITIONS}, indent=2))
    L.append("```\n")

    L.append("## 附加工件\n")
    L.append("- 每个 seed 目录含 `config.json` / `logs/run.log` / `ckpt/best_model.pt` / "
             "`result.json` / `test_predictions.csv` / `protocol_check.json`\n")
    README_OUT.write_text("\n".join(L) + "\n")

    print(f"[summary] written {SUMMARY_OUT.name} + {README_OUT.name}", flush=True)
    print(f"[summary] HE={_fmt(meta[he]['mean_auc']) if meta[he]['mean_auc'] is not None else '—'} "
          f"MSA={_fmt(meta[msa]['mean_auc']) if meta[msa]['mean_auc'] is not None else '—'} "
          f"RCX={_fmt(meta[rcx]['mean_auc']) if meta[rcx]['mean_auc'] is not None else '—'}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 4, 5])
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
