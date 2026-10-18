#!/usr/bin/env python
"""C17 v3 Optuna 进度巡检 —— 只读，不改任何东西。

    python tune/c17_v3/status.py            # 全部组合
    python tune/c17_v3/status.py HE+Ki67    # 只看一个

给出每个组合的：
  * 有效 trial 数 / 12（COMPLETE/PRUNED 且配置互不重复，见 optuna_v3.n_valid）
  * 逐条 trial 的 number / 状态 / value / best_epoch
  * **Trial 0 与历史 seed42 的 AUC 对照**（验收点 ①）
  * startup 段（Trial 1–3）配置是否互不重复（验收点 ②）
  * hold-out 验证（3611/6430/39998）是否已跑、结果如何
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent.parent / "scripts"))

import optuna                                                  # noqa: E402
import optuna_v3 as V                                          # noqa: E402

COL = {"COMPLETE": "OK  ", "PRUNED": "PRUNE", "RUNNING": "RUN ",
       "WAITING": "WAIT", "FAIL": "FAIL"}


def historical_auc(combo, seed):
    try:
        return json.loads((V.O.seed_dir(combo, seed) / "metrics.json")
                          .read_text())["AUC"]
    except Exception:                                          # noqa: BLE001
        return None


def report(combo):
    slug, seed = V.COMBOS[combo]
    combo_dir = V.TUNE_ROOT / slug
    db = combo_dir / "study.db"
    hist = historical_auc(combo, seed)

    print("=" * 78)
    print(f"{combo}   (gpu={V.GPU_OF.get(combo, '?')}, slug={slug}, "
          f"search_seed={seed})")
    print("=" * 78)

    if not db.is_file():
        done = (combo_dir / "best_result.json").is_file()
        print("  study.db 不在" + ("（该组合已收尾，见 best_result.json）"
                                   if done else "（尚未开始）"))
        for f in ("best_params.json", "best_result.json", "holdout_result.json"):
            if (combo_dir / f).is_file():
                print(f"  产物: {f}")
        return

    study = optuna.load_study(study_name=f"c17_v3_{slug}",
                              storage=f"sqlite:///{db}")
    valid = V.valid_trials(study)
    n_val = len(valid)
    print(f"  有效 trial {n_val}/{V.N_TRIALS}"
          f"   库内记录 {len(study.trials)} 条"
          f"   还差 {max(0, V.N_TRIALS - n_val)} 条")

    # 逐条 trial
    print(f"  {'#':>3}  {'state':<7} {'value':>10}  {'epoch':>5}  配置")
    for t in sorted(study.trials, key=lambda x: x.number):
        v = "—" if t.value is None else f"{float(t.value):.7f}"
        ep = t.user_attrs.get("best_epoch", "")
        p = V.trial_params_of(t)
        ps = (f"lr={p['learning_rate']:.4e} wd={p['weight_decay']:.3e} "
              f"do={p['dropout']:<5} rs={p['residual_scale']:<6}"
              f"tau={p['tau']:<5} mom={p['prototype_momentum']}") if p else ""
        print(f"  {t.number:>3}  {COL.get(t.state.name, t.state.name):<7} "
              f"{v:>10}  {str(ep):>5}  {ps}")

    # ① Trial 0 vs 历史 seed42
    t0 = next((t for t in study.trials
               if t.number == 0 and t.state == optuna.trial.TrialState.COMPLETE),
              None)
    if t0 is not None and hist is not None:
        d = float(t0.value) - hist
        ok = abs(d) < 1e-9
        print(f"\n  ① Trial 0 vs 历史 seed{seed}: "
              f"{float(t0.value):.10f} vs {hist:.10f}  "
              f"Δ={d:+.2e}  {'逐位一致 OK' if ok else '**不一致，需查**'}")

    # ② startup 段互不重复
    st = [t for t in valid if t.number < V.N_STARTUP]
    if len(st) >= 2:
        keys = [V.config_key(V.trial_params_of(t)) for t in st]
        print(f"  ② startup 段有效 {len(st)} 条，不同配置 {len(set(keys))} 个"
              f"  {'OK' if len(set(keys)) == len(keys) else '**有重复，需查**'}")
        f = [t for t in study.trials if t.number < V.N_STARTUP
             and t.state.name in ("FAIL", "RUNNING")]
        if f:
            print(f"     （startup 段还有 {len(f)} 条非有效: "
                  f"{[(t.number, t.state.name) for t in f]}）")

    # hold-out 验证
    ho = combo_dir / "holdout_result.json"
    if ho.is_file():
        j = json.loads(ho.read_text())
        print(f"\n  hold-out: {json.dumps(j, ensure_ascii=False)[:400]}")
    else:
        print(f"\n  hold-out: 未跑（需 best params 确定后由 GPU7 worker 领）")


V.GPU_OF = {c: g for c, g in zip(V.COMBOS, (2, 3, 4, 5, 6))}

if __name__ == "__main__":
    want = sys.argv[1:] or list(V.COMBOS)
    for c in want:
        report(c)
        print()
