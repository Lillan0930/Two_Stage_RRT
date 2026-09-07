#!/usr/bin/env python3
"""Stage2 改造实验 driver —— 3 条件 × 3 seeds（固定 split 216/54，Test 129）。

三个条件（各 seeds=[42,123,456]）：

  Condition 1  he_rrt_he_only
      HE-only baseline（官方 RRTEncoder → ABMIL，无 Stage2）。
  Condition 2  staining_msa
      对称双模态 CR-MSA（既有默认 Stage2，concat 后 ABMIL）。
  Condition 3  he_residual_cross
      新 Stage2：有向 HE→PR cross-attn + HE 残差写回（输出仅 HE）。

所有条件统一 LR=1e-4（无 lr_stage1/2），全部从头随机初始化，无 pretrained。
Stage1 encoder：HE=4/9/3/n_heads4/dp0，PR=8/15/5/n_heads8/dp0.1155（各 best）。
Stage2（staining_msa / he_residual_cross）：region_num=4, crmsa_k=3,
crmsa_heads=8, drop_out=0.1, drop_path=0, epeg=False, ffn=False；
he_residual_cross 另加 residual_scale=0.1, disable_cross=False。

数据：train=fixed_split/train.csv(216)，val=fixed_split/val.csv(54)（早停），
test=c16_test_labels.csv(129)（最终评估）。sampling random、max_patches 2500、
train per_epoch=True、val/test per_epoch=False、batch 1、num_workers 2、
num_epochs 80、cosine、patience 10、dropout 0.25、abmil_hidden_dim 256、
weight_decay 1e-5、label_smoothing 0、aux_loss_weight 0、modality_dropout 0、
kd_enabled False。

断点续跑：result.json + test_predictions.csv 完整即跳过；--force 覆盖。

用法:
  python scripts/run_stage2_he_residual_experiments.py --gpus 0 1 2
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
RESULTS_ROOT = PROJECT / "results" / "stage2_he_residual_cross"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_stage2_he_residual_seed.py")

# Stage1 各自 best —— 绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG（he_residual_cross 复用同一结构参数 + residual_scale）
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}

# 三条件：目录名 → (stage2_type 或 None=HE-only, 是否 HE-only)
CONDITIONS = {
    "he_rrt_he_only": ("__he_only__", True),
    "staining_msa": ("staining_msa", False),
    "he_residual_cross": ("he_residual_cross", False),
}

SUMMARY_OUT = RESULTS_ROOT / "summary.json"
README_OUT = RESULTS_ROOT / "README.md"


def build_config(condition: str, seed: int):
    stage2_type, he_only = CONDITIONS[condition]
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
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
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


def generate_summary():
    cond_aucs, cond_meta = {}, {}
    for condition in CONDITIONS:
        aucs, val_aucs, seeds_done, epochs = [], [], [], []
        out_root = RESULTS_ROOT / condition
        for s in SEEDS:
            rpath = out_root / f"seed{s}" / "result.json"
            if not rpath.exists():
                continue
            r = json.loads(rpath.read_text())
            seeds_done.append(s)
            aucs.append(r["test_auc"])
            val_aucs.append(r["best_val_auc"])
            epochs.append(r["best_epoch"])
        cond_aucs[condition] = {s: a for s, a in zip(seeds_done, aucs)}
        cond_meta[condition] = {
            "per_seed_test_auc": {s: a for s, a in zip(seeds_done, aucs)},
            "per_seed_val_auc": {s: a for s, a in zip(seeds_done, val_aucs)},
            "best_epoch": {s: e for s, e in zip(seeds_done, epochs)},
            "mean_test_auc": float(np.mean(aucs)) if aucs else None,
            "std_test_auc": float(np.std(aucs)) if aucs else None,
            "mean_val_auc": float(np.mean(val_aucs)) if val_aucs else None,
        }

    summary = {
        "task": "stage2_he_residual_cross",
        "conditions": {c: cond_meta[c] for c in CONDITIONS},
        "protocol": {
            "train": 216, "val": 54, "test": 129,
            "monitor": "val_auc", "mode": "max", "patience": 10,
            "num_epochs": 80, "scheduler": "cosine",
            "lr": 1e-4, "unified_lr": True,
            "sampling": "random", "max_patches": 2500,
            "batch_size": 1, "num_workers": 2,
            "dropout": 0.25, "abmil_hidden_dim": 256,
            "label_smoothing": 0.0, "aux_loss_weight": 0.0,
            "modality_dropout": 0.0, "kd_enabled": False,
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    lines = []
    lines.append("# Stage2 改造实验 — he_residual_cross vs staining_msa vs HE-only\n")
    lines.append("3 条件 × 3 seeds（42/123/456），固定 split 216/54，Test 129，"
                 "统一 LR=1e-4，全部从头随机初始化。\n")
    lines.append("## Test AUC 表格\n")
    lines.append("| seed | HE-only | staining_msa | he_residual_cross |")
    lines.append("|---|---|---|---|")
    for s in SEEDS:
        a = cond_aucs["he_rrt_he_only"].get(s)
        b = cond_aucs["staining_msa"].get(s)
        c = cond_aucs["he_residual_cross"].get(s)
        lines.append(f"| {s} | {_fmt(a) if a is not None else '—'} | "
                     f"{_fmt(b) if b is not None else '—'} | "
                     f"{_fmt(c) if c is not None else '—'} |")
    mh = cond_meta["he_rrt_he_only"]["mean_test_auc"]
    ms = cond_meta["staining_msa"]["mean_test_auc"]
    mr = cond_meta["he_residual_cross"]["mean_test_auc"]
    sh = cond_meta["he_rrt_he_only"]["std_test_auc"]
    ss_ = cond_meta["staining_msa"]["std_test_auc"]
    sr = cond_meta["he_residual_cross"]["std_test_auc"]
    lines.append(f"| **mean** | {_fmt(mh) if mh is not None else '—'} | "
                 f"{_fmt(ms) if ms is not None else '—'} | "
                 f"{_fmt(mr) if mr is not None else '—'} |")
    lines.append(f"| **std** | {_fmt(sh) if sh is not None else '—'} | "
                 f"{_fmt(ss_) if ss_ is not None else '—'} | "
                 f"{_fmt(sr) if sr is not None else '—'} |")
    lines.append("")
    lines.append("## best_epoch\n")
    lines.append("```")
    lines.append(json.dumps({c: cond_meta[c]["best_epoch"] for c in CONDITIONS}, indent=2))
    lines.append("```\n")
    lines.append("## 附加工件\n")
    lines.append("- 每个 seed 目录含 `result.json` / `test_predictions.csv` / "
                 "`config.json` / `ckpt/best_model.pt` / `logs/run.log`\n")
    README_OUT.write_text("\n".join(lines) + "\n")

    print(f"[summary] written {SUMMARY_OUT.name} + {README_OUT.name}", flush=True)
    print(f"[summary] HE={_fmt(mh) if mh is not None else '—'} "
          f"msa={_fmt(ms) if ms is not None else '—'} "
          f"resid={_fmt(mr) if mr is not None else '—'}", flush=True)


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
