#!/usr/bin/env python3
"""
PR+RRT+ABMIL Optuna 驱动 —— 多 GPU 并行 Bayesian 搜索（对齐 Base_mil tune.py）。

  train=270 / val=129 test（test-as-val）→ objective = best_val_auc
  采样 TPESampler(seed=42) + MedianPruner(n_startup=5, warmup=10, interval=5)
  patience=10，num_epochs=80，cosine，单 seed 42。

用法:
  python scripts/run_pr_rrt_optuna.py --study pr_rrt_abmil_optuna --gpus 6,7 --n-trials 30
"""
import os, sys, json, time, argparse, subprocess, signal
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
STORAGE_DIR = PROJECT / "results" / "c16_pr_rrt_optuna"
WORKER = PROJECT / "scripts" / "_run_pr_rrt_optuna_worker.py"
WALLCLOCK_CAP_S = 80 * 60  # optuna 硬性上限 80 分钟（用户要求 1.5h 内）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=str, default="pr_rrt_abmil_optuna")
    ap.add_argument("--gpus", type=str, default="6,7")
    ap.add_argument("--n-trials", type=int, default=30)
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    n_gpus = len(gpus)
    per_worker = args.n_trials // n_gpus
    leftover = args.n_trials - per_worker * n_gpus
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{STORAGE_DIR}/{args.study}.db"

    try:
        study = optuna.load_study(study_name=args.study, storage=storage_url)
        print(f"Resumed study={args.study} ({len(study.trials)} trials)")
    except KeyError:
        study = optuna.create_study(
            study_name=args.study, storage=storage_url, direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10,
                                interval_steps=5),
        )
        print(f"Created study={args.study}")

    print(f"GPUs: {gpus}  total_trials={args.n_trials}  per_worker={per_worker}"
          f"  storage={storage_url}")
    print(f"Search: n_heads{{2,4,8}} abmil_hidden{{128,256,384}} drop_path[0,.3] "
          f"lr[3e-5,3e-4] wd[1e-6,1e-4] label_smoothing[0,.2] | patience=10")

    procs = {}
    for i, gpu in enumerate(gpus):
        n = per_worker + (1 if i < leftover else 0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_f = open(str(STORAGE_DIR / f"_worker_gpu{gpu}.log"), "w")
        p = subprocess.Popen(
            [PYTHON, "-u", str(WORKER), "--study", args.study,
             "--storage", storage_url, "--n-trials", str(n)],
            env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
        procs[gpu] = (p, log_f)
        print(f"  GPU {gpu} → worker pid={p.pid} ({n} trials)")

    t0 = time.time()
    print("\nMonitoring... (Ctrl+C to stop, best-so-far reported)\n")
    try:
        while True:
            time.sleep(20)
            alive = [g for g, (p, _) in procs.items() if p.poll() is None]
            if not alive:
                break
            s = optuna.load_study(study_name=args.study, storage=storage_url)
            done = sum(1 for t in s.trials
                       if t.state == optuna.trial.TrialState.COMPLETE)
            pruned = sum(1 for t in s.trials
                         if t.state == optuna.trial.TrialState.PRUNED)
            best = f"{s.best_value:.4f}" if done > 0 else "N/A"
            el = time.time() - t0
            print(f"[{el/60:5.1f}m] done={done} pruned={pruned} "
                  f"best_auc={best}  alive_gpus={alive}", flush=True)
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

    # ── 最终报告 ──
    s = optuna.load_study(study_name=args.study, storage=storage_url)
    done = [t for t in s.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("\n" + "=" * 64)
    print(f"TUNING DONE  study={args.study}  completed={len(done)}")
    if done:
        best = s.best_trial
        print(f"Best AUC (test-as-val): {best.value:.4f}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print("Best-trial metrics:")
        for k in ["val_acc", "val_auc", "val_sensitivity",
                  "val_specificity", "best_epoch"]:
            print(f"  {k}: {best.user_attrs.get(k, float('nan'))}")
        # 写 best_config.json 供 5-seed 复评
        best_cfg = {
            "params": best.params,
            "best_auc": best.value,
            "user_attrs": dict(best.user_attrs),
        }
        with open(STORAGE_DIR / "best_config.json", "w") as f:
            json.dump(best_cfg, f, indent=2)
        print(f"\nSaved {STORAGE_DIR / 'best_config.json'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
