#!/usr/bin/env python3
"""
PR+RRT+ABMIL 调参驱动 —— 参考 Base_mil PR recipe（focal loss 等），
在官方 270/129 协议（fixed 25 epochs / cosine / last ckpt / no val）下扫描。

每个 config 一个命名条目，覆盖 data/model/training 中若干 key（其余沿用 PR 基线）。
每 seed 复用 scripts/_run_official_train_test_exp.py（读 config.json → 训练 → 评估 129 test）。

用法（从 TwoStageRRT/ 目录）：
    python scripts/run_pr_rrt_tuning.py \
        --configs base_recipe,base_recipe_reg,focal_hidrop \
        --seeds 42,456,789 --gpus 6,7,6
"""
import os, sys, json, subprocess, argparse
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
RUNNER = PROJECT / "scripts" / "_run_official_train_test_exp.py"
BASE_OUT = PROJECT / "results" / "c16_official_train_test"
TRAIN_LABEL = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"

# ── PR 单模态基线（region 8/15/5，PR 调参结果）──
BASE_MODEL = {
    "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
    "region_num": 8, "n_layers": 2, "n_heads": 4,
    "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 15,
    "crmsa_k": 5, "cr_msa": True, "all_shortcut": True,
    "crmsa_heads": 8, "crmsa_mlp": False,
    "fusion_type": "two_stage_region", "fusion_stage": "middle",
    "use_gated_fusion": False, "abmil_hidden_dim": 256,
    "use_mclc": False, "aggregate_modalities": True,
}
BASE_TRAINING = {
    "batch_size": 1, "num_epochs": 25,
    "learning_rate": 1e-4, "weight_decay": 1e-5,
    "scheduler": {"type": "cosine"},
    "use_amp": False, "focal_loss": False, "focal_gamma": 2.0,
    "label_smoothing": 0.0,
    "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
    "no_validation": True,
}

# Base_mil config_c16_pr_best_acc.yaml 的关键值
RECIPE_LR = 1.8797e-4
RECIPE_WD = 3.829e-6
RECIPE_GAMMA = 2.403

CONFIGS = {
    # A. Base_mil recipe 原样映射到 R²T（低 dropout 0.10 + hidden 384）
    "base_recipe": {
        "model": {"dropout": 0.101, "abmil_hidden_dim": 384},
        "training": {"focal_loss": True, "focal_gamma": RECIPE_GAMMA,
                     "learning_rate": RECIPE_LR, "weight_decay": RECIPE_WD},
    },
    # B. Base_mil recipe + 保持 R²T dropout 0.25（R²T 参数多，多留一点正则）
    "base_recipe_reg": {
        "model": {"dropout": 0.25, "abmil_hidden_dim": 384, "drop_path": 0.1},
        "training": {"focal_loss": True, "focal_gamma": RECIPE_GAMMA,
                     "learning_rate": RECIPE_LR, "weight_decay": RECIPE_WD},
    },
    # C. focal + 高 dropout 0.40 + drop_path 0.1 + trans_dropout 0.2（R²T 强正则）
    "focal_hidrop": {
        "model": {"dropout": 0.40, "abmil_hidden_dim": 384, "drop_path": 0.1,
                  "trans_dropout": 0.2},
        "training": {"focal_loss": True, "focal_gamma": RECIPE_GAMMA,
                     "learning_rate": RECIPE_LR, "weight_decay": 1e-5},
    },
    # D. focal + 低 dropout 0.15 + drop_path 0.1（折中，lr 略高）
    "focal_lowdrop": {
        "model": {"dropout": 0.15, "abmil_hidden_dim": 384, "drop_path": 0.1},
        "training": {"focal_loss": True, "focal_gamma": 2.0,
                     "learning_rate": 2.5e-4, "weight_decay": 1e-5},
    },
    # ── 结构调参（回到 plain CE / lr 1e-4 / wd 1e-5 基线 recipe，只动结构）──
    "region4":   {"model": {"region_num": 4}},
    "region16":  {"model": {"region_num": 16}},
    "nlayers1":  {"model": {"n_layers": 1}},
    "dropout35_dp": {"model": {"dropout": 0.35, "drop_path": 0.1}},
    "dropout15": {"model": {"dropout": 0.15}},
}


def build_config(overrides, seed, out_dir):
    cfg = {
        "data": {
            "dataset_type": "c16",
            "modalities": ["PR"],
            "dir_mapping": {"PR": "C16_PR_features"},
            "train_label_file": TRAIN_LABEL,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": 42,
            "no_validation": True,
        },
        "model": dict(BASE_MODEL),
        "training": dict(BASE_TRAINING),
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }
    for section, kv in overrides.items():
        for k, v in kv.items():
            cfg[section][k] = v
    return cfg


def launch(config_name, seed, gpu):
    out_dir = BASE_OUT / config_name / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)
    overrides = CONFIGS[config_name]
    with open(out_dir / "config.json", "w") as f:
        json.dump(build_config(overrides, seed, out_dir), f, indent=2)
    env = dict(os.environ)
    env["EXP_GPU"] = str(gpu)
    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen(
        [PYTHON, str(RUNNER), str(out_dir)],
        stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT), env=env,
    )
    return proc, log_f, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', type=str, default='base_recipe')
    ap.add_argument('--seeds', type=str, default='42,456,789')
    ap.add_argument('--gpus', type=str, default='6')
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) == 1:
        gpus = gpus * (len(configs) * len(seeds))
    assert len(gpus) == len(configs) * len(seeds), \
        f"gpus 数量({len(gpus)})须等于 configs×seeds({len(configs)*len(seeds)})"

    print("=" * 72)
    print("PR+RRT+ABMIL 调参（270/129, 25ep, cosine, last ckpt, focal 等）")
    print(f"Configs: {configs}")
    print(f"Seeds:   {seeds}")
    print("=" * 72)

    jobs = []
    k = 0
    for cfg_name in configs:
        for seed in seeds:
            gpu = gpus[k]; k += 1
            proc, log_f, out_dir = launch(cfg_name, seed, gpu)
            jobs.append((cfg_name, seed, gpu, proc, log_f, out_dir))
            print(f"  {cfg_name:>18}  seed={seed:>4}  GPU={gpu}")

    print(f"\n▶ {len(jobs)} jobs launched. Waiting...\n")
    for cfg_name, seed, gpu, proc, log_f, out_dir in jobs:
        ret = proc.wait()
        log_f.close()
        mf = out_dir / "metrics.json"
        if mf.exists():
            m = json.load(open(mf))
            print(f"  [{'✓' if ret == 0 else '✗'}] {cfg_name:>18} seed={seed:>4}  "
                  f"AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")
        else:
            print(f"  [✗] {cfg_name:>18} seed={seed:>4}  NO RESULT (rc={ret})")

    # 按 config 汇总
    print("\n" + "=" * 72)
    print("SUMMARY（按 config，样本标准差 ddof=1）")
    print("=" * 72)
    for cfg_name in configs:
        aucs, accs = [], []
        for seed in seeds:
            mf = BASE_OUT / cfg_name / f"seed{seed}" / "metrics.json"
            if not mf.exists():
                continue
            m = json.load(open(mf))
            aucs.append(m["auc"]); accs.append(m["accuracy"])
        if aucs:
            print(f"  {cfg_name:>18}: AUC {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}  "
                  f"Acc {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}  (n={len(aucs)})")
    print("\nDone!")


if __name__ == "__main__":
    main()
