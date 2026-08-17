#!/usr/bin/env python3
"""
C16 单模态 R²T+ABMIL Optuna 驱动（通用，按 --task）。

  train=270 / val=129 test（test-as-val）→ objective=best_val_auc
  TPESampler(seed=42) + MedianPruner(n_startup=5, warmup=10, interval=5)
  patience=10, num_epochs=80, cosine, seed 42。

产物：
  tune/best_results/<task_name>/best_params.json   （只保留最优参数 + best AUC/指标）
  搜索期间的 study.db / trial_* / worker log 放在 results/c16_rrt_optuna/<task>/，跑完自动清理。

用法:
  python scripts/run_c16_rrt_optuna.py --task er   --gpus 0,1 --n-trials 30
  python scripts/run_c16_rrt_optuna.py --task her2 --gpus 2,3 --n-trials 30
  python scripts/run_c16_rrt_optuna.py --task ki67 --gpus 6,7 --n-trials 30
"""
import os, sys, json, time, argparse, subprocess, shutil
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
WORKER = PROJECT / "scripts" / "_run_c16_rrt_optuna_worker.py"
SCRATCH_ROOT = PROJECT / "results" / "c16_rrt_optuna"
BEST_ROOT = PROJECT / "tune" / "best_results"
WALLCLOCK_CAP_S = 80 * 60  # 每个任务 optuna 硬性上限 80 分钟

TASKS = {
    "pr":   {"modality": "PR",   "feature_dir": "C16_PR_features",   "name": "pr_rrt_abmil"},
    "er":   {"modality": "ER",   "feature_dir": "C16_ER_features",   "name": "er_rrt_abmil"},
    "her2": {"modality": "HER2", "feature_dir": "C16_HER2_features", "name": "her2_rrt_abmil"},
    "ki67": {"modality": "Ki67", "feature_dir": "C16_Ki67_features", "name": "ki67_rrt_abmil"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--gpus", type=str, required=True)
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    task = TASKS[args.task]
    task_name = task["name"]
    gpus = [int(x) for x in args.gpus.split(",")]
    n_gpus = len(gpus)
    per_worker = args.n_trials // n_gpus
    leftover = args.n_trials - per_worker * n_gpus

    scratch = SCRATCH_ROOT / args.task
    scratch.mkdir(parents=True, exist_ok=True)
    study_name = f"{task_name}_optuna"
    storage_url = f"sqlite:///{scratch}/{study_name}.db"
    # 长 timeout：多 worker 并发写 SQLite 时等待而非直接 database is locked
    storage = RDBStorage(storage_url, engine_kwargs={"connect_args": {"timeout": 60}})

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"Resumed study={study_name} ({len(study.trials)} trials)")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name, storage=storage, direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10,
                                interval_steps=5),
        )
        print(f"Created study={study_name}")

    print(f"[{task_name}] modality={task['modality']} GPUs={gpus} "
          f"trials={args.n_trials} (per_worker={per_worker})")

    procs = {}
    for i, gpu in enumerate(gpus):
        n = per_worker + (1 if i < leftover else 0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_f = open(str(scratch / f"_worker_gpu{gpu}.log"), "w")
        p = subprocess.Popen(
            [PYTHON, "-u", str(WORKER), "--task", args.task,
             "--study", study_name, "--storage", storage_url,
             "--n-trials", str(n)],
            env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs[gpu] = (p, log_f)
        print(f"  GPU {gpu} → worker pid={p.pid} ({n} trials)")

    t0 = time.time()
    print(f"\n[{task_name}] monitoring...\n")
    try:
        while True:
            time.sleep(20)
            alive = [g for g, (p, _) in procs.items() if p.poll() is None]
            if not alive:
                break
            try:
                s = optuna.load_study(study_name=study_name, storage=storage)
                done = sum(1 for t in s.trials
                           if t.state == optuna.trial.TrialState.COMPLETE)
                pruned = sum(1 for t in s.trials
                             if t.state == optuna.trial.TrialState.PRUNED)
                best = f"{s.best_value:.4f}" if done > 0 else "N/A"
                el = time.time() - t0
                print(f"[{task_name}][{el/60:5.1f}m] done={done} pruned={pruned} "
                      f"best_auc={best} alive={alive}", flush=True)
            except Exception as e:
                print(f"[{task_name}] monitor query skipped "
                      f"({type(e).__name__}: {e})", flush=True)
            if el > WALLCLOCK_CAP_S:
                print("Wall-clock cap reached, stopping workers.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for g, (p, log_f) in procs.items():
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            log_f.close()

    # ── 最终报告 + 落盘 best_params + 清理 ──
    s = optuna.load_study(study_name=study_name, storage=storage)
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("\n" + "=" * 60)
    print(f"[{task_name}] DONE  completed={len(done)}")
    if done:
        best = s.best_trial
        print(f"Best AUC (test-as-val): {best.value:.4f}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")

        best_dir = BEST_ROOT / task_name
        best_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "task": task_name,
            "modality": task["modality"],
            "feature_dir": task["feature_dir"],
            "protocol": {
                "train": 270, "val_test": 129, "early_stopping_monitor": "val_auc",
                "patience": 10, "num_epochs": 80, "scheduler": "cosine", "seed": 42,
            },
            "best_auc": best.value,
            "best_params": best.params,
            "best_trial_metrics": {
                "val_acc": best.user_attrs.get("val_acc"),
                "val_auc": best.user_attrs.get("val_auc"),
                "val_sensitivity": best.user_attrs.get("val_sensitivity"),
                "val_specificity": best.user_attrs.get("val_specificity"),
                "best_epoch": best.user_attrs.get("best_epoch"),
            },
            "fixed_model_params": {
                "region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_layers": 2,
                "dropout": 0.25, "trans_dropout": 0.1, "focal_loss": False,
                "mlp_dim": 512, "crmsa_heads": 8, "crmsa_mlp": False,
            },
            "n_completed": len(done),
            "n_pruned": sum(1 for t in s.trials
                            if t.state == optuna.trial.TrialState.PRUNED),
        }
        with open(best_dir / "best_params.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved {best_dir / 'best_params.json'}")
    print("=" * 60)

    # 清理中间过程与缓存
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"Cleaned scratch {scratch}")


if __name__ == "__main__":
    main()
