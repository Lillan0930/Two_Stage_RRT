#!/usr/bin/env python3
"""
C16 单模态 best config 5-seed 复评（test-as-val），每个任务单独 GPU。

加载 tune/best_results/<task>_rrt_abmil/best_params.json 的最优参数，跑 5 个 seed，
汇报 test-as-val AUC 的 mean±std，与 Base_mil 0.8001±0.0468（同为 test-as-val 泄漏协议）对比。

协议与 optuna 完全一致（复用 _run_c16_rrt_optuna_worker 的 _base_config）：
  train=270 / val=129 test（test-as-val）→ early stopping val_auc/max/patience 10
  num_epochs=80 / cosine / batch 1 / sampling random。

仅变 seed：environment.seed + data.sample_seed 同时设为该 seed。

用法:
  python scripts/run_c16_best_five_seed.py --task er   --gpu 0
  python scripts/run_c16_best_five_seed.py --task her2 --gpu 1
  python scripts/run_c16_best_five_seed.py --task ki67 --gpu 2
"""
import os, sys, json, logging, shutil, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
os.chdir(str(PROJECT))

from _run_c16_rrt_optuna_worker import _base_config, TASKS  # noqa: E402

SEEDS = [42, 123, 456, 789, 1024]
RESULTS_ROOT = PROJECT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from train import Trainer  # noqa: E402  (env 设好后再 import，避免 CUDA 提前初始化)

    task = args.task
    modality = TASKS[task]["modality"]
    best_path = PROJECT / "tune/best_results" / f"{task}_rrt_abmil" / "best_params.json"
    out_root = RESULTS_ROOT / f"{modality}_rrt_abmil"

    best = json.loads(best_path.read_text())
    p = best["best_params"]

    def build_config(seed):
        cfg = _base_config(task)
        cfg["model"]["n_heads"] = p["n_heads"]
        cfg["model"]["abmil_hidden_dim"] = p["abmil_hidden_dim"]
        cfg["model"]["drop_path"] = p["drop_path"]
        cfg["training"]["learning_rate"] = p["lr"]
        cfg["training"]["weight_decay"] = p["wd"]
        cfg["training"]["label_smoothing"] = p["label_smoothing"]
        cfg["environment"]["seed"] = seed
        cfg["data"]["sample_seed"] = seed
        return cfg

    def run_seed(seed):
        out_dir = out_root / f"seed{seed}"
        for d in ["ckpt", "logs", "img"]:
            (out_dir / d).mkdir(parents=True, exist_ok=True)
        cfg = build_config(seed)
        cfg["output"]["save_dir"] = str(out_dir / "ckpt")
        cfg["output"]["log_dir"] = str(out_dir / "logs")
        cfg["output"]["img_dir"] = str(out_dir / "img")
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        logger = logging.getLogger(f"{modality}_seed{seed}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(fh)
        logger.propagate = False

        logger.info(f"{modality} best config 5-seed re-eval | seed={seed}")

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
        print(f"[{modality} seed {seed}] val_auc={result['val_auc']:.4f} "
              f"acc={result['val_acc']:.4f} sens={result['val_sensitivity']:.4f} "
              f"spec={result['val_specificity']:.4f} epoch={result['best_epoch']}",
              flush=True)

        shutil.rmtree(out_dir / "ckpt", ignore_errors=True)
        shutil.rmtree(out_dir / "img", ignore_errors=True)
        return result

    print(f"{modality} best config 5-seed re-eval (test-as-val, GPU {args.gpu})",
          flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Best params: {p}", flush=True)

    results = [run_seed(s) for s in SEEDS]

    aucs = [r["val_auc"] for r in results]
    accs = [r["val_acc"] for r in results]
    sens = [r["val_sensitivity"] for r in results]
    specs = [r["val_specificity"] for r in results]

    print("\n" + "=" * 64, flush=True)
    print(f"SUMMARY {modality} (test-as-val, 270 train / 129 val-test)", flush=True)
    for r in results:
        print(f"  seed={r['seed']:>4}: AUC={r['val_auc']:.4f} "
              f"acc={r['val_acc']:.4f} sens={r['val_sensitivity']:.4f} "
              f"spec={r['val_specificity']:.4f} epoch={r['best_epoch']}",
              flush=True)
    print(f"  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}", flush=True)
    print(f"  Sens: {np.mean(sens):.4f} ± {np.std(sens):.4f}", flush=True)
    print(f"  vs Base_mil 0.8001 ± 0.0468", flush=True)

    summary = {
        "task": f"{task}_rrt_abmil",
        "modality": modality,
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc",
                     "mode": "max", "patience": 10, "num_epochs": 80,
                     "scheduler": "cosine", "sampling": "random"},
        "seeds": SEEDS,
        "best_params": p,
        "per_seed": results,
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_sensitivity": float(np.mean(sens)),
        "std_sensitivity": float(np.std(sens)),
        "mean_specificity": float(np.mean(specs)),
        "std_specificity": float(np.std(specs)),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_root / 'summary.json'}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
