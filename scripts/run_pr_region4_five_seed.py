#!/usr/bin/env python3
"""PR+RRT+ABMIL only，region_num=4（其余与 PR best 完全一致），5 seeds。

对比 region_num=8 的 PR-only（0.8368±0.0217）。**只改 region_num 8→4**，
best_params（n_heads=8 / abmil_hidden_dim=384 / drop_path / lr / wd / label_smoothing）
与固定结构参数（epeg_k=15 / crmsa_k=5 / n_layers=2 / mlp_dim=512 等）全部保持。

协议：270 train / 129 test-as-val，early-stop val_auc/max/patience 10，80 epochs，cosine。
5 seeds（42/123/456/789/1024）。

用法（driver，并行）:
  python scripts/run_pr_region4_five_seed.py --gpus 2 3 6 7
（worker，由 driver 内部调用）:
  python scripts/run_pr_region4_five_seed.py --seed 42 --gpu 2
"""
import os, sys, json, shutil, subprocess, argparse, logging
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
SEEDS = [42, 123, 456, 789, 1024]
BEST_PATH = PROJECT / "tune/best_results/pr_rrt_abmil/best_params.json"
OUT_ROOT = PROJECT / "results" / "pr_region4_five_seed"
REGION_NUM = 4


def build_config(seed: int):
    # 内联 _base_config("pr")，避免 import optuna（该文件顶层 import optuna，
    # 在本 shell 会触发 libstdc++/libicu CXXABI_1.3.15 冲突）。
    best = json.loads(BEST_PATH.read_text())
    p = best["best_params"]
    cfg = {
        "data": {
            "dataset_type": "c16", "modalities": ["PR"],
            "dir_mapping": {"PR": "C16_PR_features"},
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "val_label_file": str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
            "feature_base_dir": "/home/Public/lillan/features_result/C16_features",
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": REGION_NUM, "n_layers": 2, "n_heads": p["n_heads"],
            "drop_path": p["drop_path"], "trans_dropout": 0.1, "epeg": True, "epeg_k": 15,
            "crmsa_k": 5, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": p["abmil_hidden_dim"],
            "use_mclc": False, "aggregate_modalities": True,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": p["lr"], "weight_decay": p["wd"],
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False,
            "label_smoothing": p["label_smoothing"],
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }
    return cfg


def worker(seed: int, gpu: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from train import Trainer

    out_dir = OUT_ROOT / f"seed{seed}"
    for d in ["ckpt", "logs", "img"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    cfg = build_config(seed)
    cfg["output"]["save_dir"] = str(out_dir / "ckpt")
    cfg["output"]["log_dir"] = str(out_dir / "logs")
    cfg["output"]["img_dir"] = str(out_dir / "img")
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger = logging.getLogger(f"pr_r4_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"PR region_num={REGION_NUM} 5-seed | seed={seed}")

    trainer = Trainer(cfg, logger, f"seed{seed}")
    trainer.train()

    result = {
        "seed": seed,
        "val_auc": float(trainer.best_val_auc),
        "val_acc": float(trainer.best_val_acc),
        "val_sensitivity": float(trainer.best_val_sensitivity),
        "val_specificity": float(trainer.best_val_specificity),
        "best_epoch": int(trainer.best_epoch),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[seed {seed} gpu {gpu}] val_auc={result['val_auc']:.4f} "
          f"epoch={result['best_epoch']}", flush=True)

    shutil.rmtree(out_dir / "ckpt", ignore_errors=True)
    shutil.rmtree(out_dir / "img", ignore_errors=True)


def run_wave(pairs):
    procs = []
    for seed, gpu in pairs:
        log_f = open(OUT_ROOT / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, __file__, "--seed", str(seed), "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((seed, p, log_f))
        print(f"[launch] seed={seed} gpu={gpu}", flush=True)
    for seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = OUT_ROOT / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] seed={seed} val_auc={r['val_auc']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 6, 7])
    args = ap.parse_args()

    if args.seed is not None:
        worker(args.seed, args.gpu)
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    best = json.loads(BEST_PATH.read_text())
    print(f"PR region_num={REGION_NUM} 5-seed | seeds={SEEDS} | gpus={args.gpus}",
          flush=True)
    print(f"best_params={best['best_params']}", flush=True)

    for i in range(0, len(SEEDS), len(args.gpus)):
        chunk = SEEDS[i:i + len(args.gpus)]
        run_wave(list(zip(chunk, args.gpus[:len(chunk)])))

    aucs, accs, results = [], [], []
    for s in SEEDS:
        rpath = OUT_ROOT / f"seed{s}" / "result.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            results.append(r)
            aucs.append(r["val_auc"])
            accs.append(r["val_acc"])

    print("\n" + "=" * 64, flush=True)
    print(f"SUMMARY PR region_num={REGION_NUM} (270/129 test-as-val)", flush=True)
    for r in results:
        print(f"  seed={r['seed']:>4}: AUC={r['val_auc']:.4f} "
              f"acc={r['val_acc']:.4f} epoch={r['best_epoch']}", flush=True)
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"  vs PR region_num=8: 0.8368 ± 0.0217", flush=True)

    summary = {
        "task": "pr_region4_five_seed",
        "region_num": REGION_NUM,
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc",
                     "mode": "max", "patience": 10, "num_epochs": 80,
                     "scheduler": "cosine", "sampling": "random"},
        "seeds": SEEDS,
        "best_params": best["best_params"],
        "per_seed": results,
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
    }
    with open(OUT_ROOT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_ROOT / 'summary.json'}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
