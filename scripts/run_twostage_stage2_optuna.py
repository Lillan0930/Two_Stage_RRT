#!/usr/bin/env python3
"""Two-stage (HE+PR) Stage-2 小规模 Optuna —— 防过拟合，10 trials × 3 seeds。

固定 Stage-1 两个 encoder（各自 best，不再调）：
  HE: region_num=4 / epeg_k=9 / crmsa_k=3 / n_heads=4 / drop_path=0.0
  PR: region_num=8 / epeg_k=15 / crmsa_k=5 / n_heads=8 / drop_path=0.1155

只搜 Stage-2 三个最关键参数（其余固定）：
  crmsa_k   ∈ {1, 3, 5}
  drop_path ∈ {0, 0.1, 0.2}
  stage2_lr ∈ {1e-5, 2e-5, 5e-5}

objective = 每 config 跑 3 seeds(42/123/456)，score = mean(AUC) − 0.5·std(AUC)
（选稳定，不选单 seed 峰值，避免 129 样本 test-as-val 过拟合）。

协议：270 train / 129 test-as-val，early-stop val_auc/max/patience 10，num_epochs=80，cosine。
只保留最终最优结果 + summary/best_params，中间缓存清理。

用法:
  python scripts/run_twostage_stage2_optuna.py --n-trials 10 --gpu42 3 --gpu123 6 --gpu456 7
"""
import os, sys, json, shutil, subprocess, argparse
from pathlib import Path

import numpy as np
import optuna

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
RESULTS_ROOT = PROJECT / "results" / "HE_PR_twostage_rrt"
SCRATCH = RESULTS_ROOT / "_scratch"

SEEDS = [42, 123, 456]
N_TRIALS = 10

SEED_RUNNER = str(PROJECT / "scripts" / "_run_twostage_seed.py")
PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"

STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4, "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}
STAGE2_FIXED = {"region_num": 4, "crmsa_heads": 8, "drop_out": 0.1,
                "epeg": True, "epeg_k": 9, "crmsa_mlp": False}
FIXED_TRAINING = {"lr_stage1": 1e-5, "weight_decay": 1e-5,
                  "dropout": 0.25, "label_smoothing": 0.0, "abmil_hidden_dim": 256}

GPU_BY_SEED = {42: 3, 123: 6, 456: 7}


def base_config(crmsa_k, drop_path, stage2_lr, seed):
    return {
        "data": {
            "dataset_type": "c16", "modalities": ["HE", "PR"],
            "dir_mapping": {"HE": "C16_HE_features", "PR": "C16_PR_features"},
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "val_label_file": str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": FIXED_TRAINING["dropout"],
            "use_gated": False,
            # 顶层共享默认 = HE 值（fallback）
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "stage2_type": "staining_msa",
            "use_gated_fusion": False, "abmil_hidden_dim": FIXED_TRAINING["abmil_hidden_dim"],
            "use_mclc": False, "aggregate_modalities": True,
            "encoder_cfg": STAGE1_ENCODER_CFG,
            "stage2_cfg": dict(STAGE2_FIXED, crmsa_k=crmsa_k, drop_path=drop_path),
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4,  # placeholder，差分 lr 生效时不用
            "lr_stage1": FIXED_TRAINING["lr_stage1"], "lr_stage2": stage2_lr,
            "weight_decay": FIXED_TRAINING["weight_decay"],
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False,
            "label_smoothing": FIXED_TRAINING["label_smoothing"],
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }


def objective(trial):
    crmsa_k = trial.suggest_categorical("crmsa_k", [1, 3, 5])
    drop_path = trial.suggest_categorical("drop_path", [0.0, 0.1, 0.2])
    stage2_lr = trial.suggest_categorical("stage2_lr", [1e-5, 2e-5, 5e-5])

    procs = []
    for seed in SEEDS:
        cfg = base_config(crmsa_k, drop_path, stage2_lr, seed)
        seed_dir = SCRATCH / f"trial_{trial.number}" / f"seed{seed}"
        for d in ["ckpt", "logs", "img"]:
            (seed_dir / d).mkdir(parents=True, exist_ok=True)
        cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
        cfg["output"]["log_dir"] = str(seed_dir / "logs")
        cfg["output"]["img_dir"] = str(seed_dir / "img")
        cfg_path = seed_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        cmd = [PY, SEED_RUNNER, "--config", str(cfg_path), "--gpu", str(GPU_BY_SEED[seed])]
        log_f = open(seed_dir / "logs" / "stdout.log", "w")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((seed, seed_dir, p, log_f))

    for _, _, p, log_f in procs:
        p.wait()
        log_f.close()

    aucs = []
    for seed, seed_dir, p, _ in procs:
        rpath = seed_dir / "result.json"
        if p.returncode != 0 or not rpath.exists():
            raise optuna.exceptions.TrialPruned(f"seed {seed} failed rc={p.returncode}")
        r = json.loads(rpath.read_text())
        aucs.append(float(r["val_auc"]))

    aucs = np.array(aucs)
    mean = float(aucs.mean())
    std = float(aucs.std())
    score = mean - 0.5 * std
    trial.set_user_attr("mean_auc", mean)
    trial.set_user_attr("std_auc", std)
    trial.set_user_attr("per_seed_auc", [float(a) for a in aucs])
    print(f"[trial {trial.number}] crmsa_k={crmsa_k} drop_path={drop_path} "
          f"stage2_lr={stage2_lr:.1e} | aucs={[round(a,4) for a in aucs]} "
          f"mean={mean:.4f} std={std:.4f} score={score:.4f}", flush=True)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--gpu42", type=int, default=3)
    ap.add_argument("--gpu123", type=int, default=6)
    ap.add_argument("--gpu456", type=int, default=7)
    args = ap.parse_args()

    global GPU_BY_SEED
    GPU_BY_SEED = {42: args.gpu42, 123: args.gpu123, 456: args.gpu456}

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"\nBEST score={best.value:.4f} | mean={best.user_attrs['mean_auc']:.4f} "
          f"std={best.user_attrs['std_auc']:.4f} | {best.params}", flush=True)

    trials_summary = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        trials_summary.append({
            "trial": t.number, "params": t.params, "score": t.value,
            "mean_auc": t.user_attrs.get("mean_auc"),
            "std_auc": t.user_attrs.get("std_auc"),
            "per_seed_auc": t.user_attrs.get("per_seed_auc"),
        })

    # 拷贝 best trial 的 3 seed config + result 到最终目录
    best_scratch = SCRATCH / f"trial_{best.number}"
    final_dir = RESULTS_ROOT / f"best_trial{best.number}"
    if best_scratch.exists():
        shutil.copytree(best_scratch, final_dir, dirs_exist_ok=True)

    summary = {
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc", "mode": "max",
                     "patience": 10, "num_epochs": 80, "scheduler": "cosine",
                     "seeds": SEEDS, "score_formula": "mean(AUC) - 0.5*std(AUC)",
                     "objective": "maximize stability-adjusted test-as-val AUC"},
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_fixed": STAGE2_FIXED,
        "fixed_training": FIXED_TRAINING,
        "best": {"trial": best.number, "params": best.params, "score": best.value,
                 "mean_auc": best.user_attrs.get("mean_auc"),
                 "std_auc": best.user_attrs.get("std_auc"),
                 "per_seed_auc": best.user_attrs.get("per_seed_auc")},
        "all_trials": trials_summary,
    }
    with open(RESULTS_ROOT / "best_params.json", "w") as f:
        json.dump(best.params, f, indent=2)
    with open(RESULTS_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 清理中间缓存（保留 best 最终目录 + summary/best_params）
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"\nSaved {RESULTS_ROOT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
