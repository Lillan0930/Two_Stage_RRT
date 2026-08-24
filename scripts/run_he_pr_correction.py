#!/usr/bin/env python3
"""实验A：PR→HE correction（Z_final = Z_HE + Δ_HE，只返回 HE，PR 不进 ABMIL）。

验证：之前 two-stage 掉到 ~0.73 是因为「PR 直接进入 ABMIL 稀释 HE」，
还是「跨染色交互本身不适合」。这里 PR 仅作为 CR-MSA 的 conditioning context，
只更新 HE routing token 并 dispatch 回 HE，ABMIL 只看到 N_HE 个 token。

配置与 Stage2 optuna best（trial 0）一致：
  Stage1 固定：HE(4/9/3, n_heads=4, dp=0)、PR(8/15/5, n_heads=8, dp=0.1155)
  Stage2：crmsa_k=3 / drop_path=0.0 / stage2_lr=2e-5（region_num=4, heads=8 固定）

协议：270 train / 129 test-as-val，early-stop val_auc/max/patience 10，80 epochs，cosine。
跑 5 个 seed（42/123/456/789/1024），对比 HE-only 0.8052±0.0233。

用法:
  python scripts/run_he_pr_correction.py --gpus 2 3 6 7
"""
import os, sys, json, shutil, subprocess, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
RESULTS_ROOT = PROJECT / "results" / "HE_PR_correction"
SCRATCH = RESULTS_ROOT / "_scratch"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_twostage_seed.py")
PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"

SEEDS = [42, 123, 456, 789, 1024]

STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4, "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}
STAGE2_CFG = {"region_num": 4, "crmsa_heads": 8, "drop_out": 0.1,
              "epeg": True, "epeg_k": 9, "crmsa_mlp": False,
              "crmsa_k": 3, "drop_path": 0.0}
STAGE2_LR = 2e-5
LR_STAGE1 = 1e-5


def build_config(seed: int):
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
            "learning_rate": 1e-4,  # placeholder，差分 lr 生效时不用
            "lr_stage1": LR_STAGE1, "lr_stage2": STAGE2_LR,
            "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }


def run_wave(seed_gpu_pairs):
    """并行跑一组 (seed, gpu)，返回 {seed: result dict}。"""
    procs = []
    for seed, gpu in seed_gpu_pairs:
        cfg = build_config(seed)
        seed_dir = SCRATCH / f"seed{seed}"
        for d in ["ckpt", "logs", "img"]:
            (seed_dir / d).mkdir(parents=True, exist_ok=True)
        cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
        cfg["output"]["log_dir"] = str(seed_dir / "logs")
        cfg["output"]["img_dir"] = str(seed_dir / "img")
        cfg_path = seed_dir / "config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log_f = open(seed_dir / "logs" / "stdout.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)], stdout=log_f,
                             stderr=subprocess.STDOUT)
        procs.append((seed, seed_dir, p, log_f))
        print(f"[launch] seed={seed} gpu={gpu}", flush=True)

    results = {}
    for seed, seed_dir, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = seed_dir / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] seed={seed} rc={p.returncode}", flush=True)
            results[seed] = None
            continue
        r = json.loads(rpath.read_text())
        results[seed] = r
        print(f"[done] seed={seed} val_auc={r['val_auc']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 6, 7])
    args = ap.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"PR→HE correction 实验A | seeds={SEEDS} | gpus={args.gpus}", flush=True)

    # 按可用 GPU 数分批跑（4 GPU 跑 5 seed → 先 4 后 1）
    all_results = {}
    for i in range(0, len(SEEDS), len(args.gpus)):
        chunk = SEEDS[i:i + len(args.gpus)]
        pairs = list(zip(chunk, args.gpus[:len(chunk)]))
        all_results.update(run_wave(pairs))

    aucs = [r["val_auc"] for r in all_results.values() if r is not None]
    accs = [r["val_acc"] for r in all_results.values() if r is not None]

    print("\n" + "=" * 64, flush=True)
    print("SUMMARY PR→HE correction (270/129 test-as-val)", flush=True)
    for s in SEEDS:
        r = all_results.get(s)
        if r:
            print(f"  seed={s:>4}: AUC={r['val_auc']:.4f} acc={r['val_acc']:.4f} "
                  f"epoch={r['best_epoch']}", flush=True)
        else:
            print(f"  seed={s:>4}: FAILED", flush=True)
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"  vs HE-only 0.8052 ± 0.0233", flush=True)

    summary = {
        "task": "he_pr_correction",
        "description": "PR→HE correction: Z_final = Z_HE + Δ_HE, PR only conditions HE",
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc",
                     "mode": "max", "patience": 10, "num_epochs": 80,
                     "scheduler": "cosine", "sampling": "random"},
        "seeds": SEEDS,
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "lr_stage1": LR_STAGE1, "lr_stage2": STAGE2_LR,
        "per_seed": {str(s): all_results.get(s) for s in SEEDS},
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
    }
    with open(RESULTS_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {RESULTS_ROOT / 'summary.json'}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
