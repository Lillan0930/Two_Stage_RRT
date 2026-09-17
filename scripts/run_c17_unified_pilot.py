#!/usr/bin/env python
"""Phase B：**统一流程**下的 HE 对照 与 HE+PR Pilot-A（各 3 seeds）。

为什么需要 HE 对照
------------------
Pilot-A 的目的不是"和历史 baseline 比"，而是回答"**加 PR 到底有没有用**"。
历史 baseline 走的是 ``RRT_ABMIL``（单模态专用模型），而 HE+PR 走的是
``HEAuxUnifiedModel``（统一模型 + v3 cross fusion）——两者模型不同，直接相减
分不清差异来自 PR 还是来自模型本身。所以必须再跑一组**同一模型、同一流程、
同一 split、只把 modality_list 缩成 ["HE"]** 的对照：

    HE      : HEAuxUnifiedModel(modalities=["HE"])           ← 对照
    HE+PR   : HEAuxUnifiedModel(modalities=["HE","PR"])      ← Pilot-A

两组除 ``data.modalities`` 外没有任何字段不同（都由
``ours_main_c17_abmil.build_config`` 生成），因此 HE+PR − HE 的差值才是 PR
的贡献。历史 Phase A 只用于确认"配置可复现"。

产物::

    _pilot/unified/HE/seed{42,123,456}/{config.yaml,train.log,best_model.pt,
                                     metrics.json,history.csv,status.json}
    _pilot/unified/HE+PR/seed{42,123,456}/...
    _pilot/unified/summary.txt

用法::

    python scripts/run_c17_unified_pilot.py --seeds 42,123,456 --gpus 6,7
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ours_main_c17_abmil import OUT_ROOT, PRIMARY_METRICS   # noqa: E402

PILOT_ROOT = OUT_ROOT / "_pilot" / "unified"

#: (组合名, modalities) —— HE 是对照，HE+PR 是 Pilot-A
PILOT_COMBOS = [
    ("HE",    ["HE"]),
    ("HE+PR", ["HE", "PR"]),
]

#: Phase A 的结果（同一批 seed 的历史 baseline 复现），用于交叉参考
PHASE_A_DIR = OUT_ROOT / "_pilot" / "HE_baseline"


def run_dir(combo, seed):
    return PILOT_ROOT / combo / f"seed{seed}"


def launch(combo, seed, gpu):
    sd = run_dir(combo, seed)
    sd.mkdir(parents=True, exist_ok=True)
    logf = open(sd / "driver.log", "a")
    logf.write(f"\n===== unified pilot start "
               f"{datetime.now().isoformat(timespec='seconds')} "
               f"combo={combo} seed={seed} physical_gpu={gpu} =====\n")
    logf.flush()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "4"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_one_c17_seed.py"),
           "--combo", combo, "--seed", str(seed), "--gpu", str(gpu),
           "--out-root", str(PILOT_ROOT)]
    p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=str(REPO_ROOT))
    return {"combo": combo, "seed": seed, "proc": p, "t0": time.time(), "logf": logf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,123,456")
    ap.add_argument("--gpus", default="6,7")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)

    # 展开成 run 列表：先按 seed 再按组合，保证同一 seed 的 HE / HE+PR 尽量同时起
    queue = [(c, s) for s in seeds for c, _ in PILOT_COMBOS]
    print(f"[pilot] {len(queue)} runs: {[f'{c}/s{s}' for c, s in queue]}")
    print(f"[pilot] gpus={gpus} out={PILOT_ROOT}", flush=True)

    slots, failures = {}, []
    t0 = time.time()

    while queue or slots:
        for gpu in gpus:
            if gpu in slots or not queue:
                continue
            combo, seed = queue.pop(0)
            slots[gpu] = launch(combo, seed, gpu)
            print(f"[launch] gpu{gpu} <- {combo} seed{seed} "
                  f"(pid={slots[gpu]['proc'].pid})", flush=True)

        time.sleep(10)
        for gpu in list(slots):
            s = slots[gpu]
            rc = s["proc"].poll()
            if rc is None:
                continue
            s["logf"].close()
            dt = (time.time() - s["t0"]) / 60
            m = run_dir(s["combo"], s["seed"]) / "metrics.json"
            if rc == 0 and m.is_file():
                d = json.loads(m.read_text())
                print(f"[done  ] gpu{gpu} {s['combo']} seed{s['seed']} {dt:.1f}min "
                      f"AUC={d['AUC']:.4f} best_epoch={d['best_epoch']}", flush=True)
            else:
                failures.append((s["combo"], s["seed"], gpu, rc))
                print(f"[FAILED] gpu{gpu} {s['combo']} seed{s['seed']} rc={rc} "
                      f"{dt:.1f}min", flush=True)
            del slots[gpu]

    write_summary(seeds)
    print(f"\n[pilot] 完成 {len(queue)} 排队 / 失败 {len(failures)}，"
          f"耗时 {(time.time()-t0)/60:.1f} min")
    for c, s, g, rc in failures:
        print(f"  FAILED {c} seed{s} gpu{g} rc={rc}")
    return 0 if not failures else 2


def _load(combo, seed):
    p = run_dir(combo, seed) / "metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def _phase_a(seed):
    p = PHASE_A_DIR / f"seed{seed}" / "metrics.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def write_summary(seeds):
    L = []
    A = L.append
    A("=" * 88)
    A("Phase B — C17 统一流程 Pilot：HE 对照 vs HE+PR (Pilot-A)")
    A("=" * 88)
    A(f"生成时间    : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A("模型        : HEAuxUnifiedModel + ABMIL（TwoStageRRT，v3-style HE-anchored "
      "residual cross fusion）")
    A("配置        : scripts/ours_main_c17_abmil.py::build_config —— 两组只差 "
      "data.modalities")
    A("split       : 患者级 patient_000..099 train / patient_100..199 test-as-val"
      "（沿用历史，未改）")
    A(f"seeds       : {seeds}")
    A("")
    A("⚠ 3 seeds 只用于确认工程可行性 / PR 是否有正贡献，不是论文的 10-seed 统计。")
    A("")

    he = {s: _load("HE", s) for s in seeds}
    hp = {s: _load("HE+PR", s) for s in seeds}
    pa = {s: _phase_a(s) for s in seeds}

    if not any(he.values()) and not any(hp.values()):
        A("没有任何完成的 run。")
        (PILOT_ROOT / "summary.txt").write_text("\n".join(L) + "\n")
        return

    A("── 逐 seed 明细 ──")
    hdr = (f"{'Seed':<8}{'HE AUC':>10}{'HE+PR AUC':>12}{'Δ(PR−HE)':>12}"
           f"{'PhaseA AUC':>12}{'HE vs A':>10}")
    A(hdr)
    A("-" * len(hdr))
    for s in seeds:
        h, p, a = he.get(s), hp.get(s), pa.get(s)
        hs = f"{h['AUC']:.4f}" if h else "—"
        ps = f"{p['AUC']:.4f}" if p else "—"
        ds = f"{(p['AUC']-h['AUC'])*100:+.2f}" if (h and p) else "—"
        as_ = f"{a['AUC']:.4f}" if a else "—"
        vs = f"{(h['AUC']-a['AUC'])*100:+.2f}" if (h and a) else "—"
        A(f"{s:<8}{hs:>10}{ps:>12}{ds:>12}{as_:>12}{vs:>10}")
    A("-" * len(hdr))
    A("")

    def block(name, rows):
        got = [r for r in rows.values() if r]
        if not got:
            A(f"{name}: 无完成的 run")
            return
        A(f"── {name}（n={len(got)}）──")
        for k in PRIMARY_METRICS:
            v = np.array([r[k] for r in got], dtype=float)
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            A(f"  {k:<12}: {v.mean():.4f} ± {sd:.4f}")
        A("")

    block("HE（对照）", he)
    block("HE+PR（Pilot-A）", hp)

    # ── 配对差值 ──
    pair = [(s, hp[s]['AUC'] - he[s]['AUC']) for s in seeds if he.get(s) and hp.get(s)]
    if pair:
        d = np.array([x[1] for x in pair], dtype=float) * 100
        A("── 配对差值 HE+PR − HE（AUC 点）──")
        for s, x in pair:
            A(f"  seed {s:<6}: {x*100:+.2f}")
        ds = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        A(f"  平均     : {d.mean():+.2f} ± {ds:.2f}")
        A(f"  方向     : {int((d > 0).sum())}/{len(d)} 个 seed 为正")
        A("")

    # ── 判定提示（只陈述事实，不自动选配置）──
    A("── 事实陈述（供人工判定，脚本不自动选配置）──")
    he_ok, hp_ok = [r for r in he.values() if r], [r for r in hp.values() if r]
    if he_ok and hp_ok:
        hm = float(np.mean([r['AUC'] for r in he_ok]))
        pm = float(np.mean([r['AUC'] for r in hp_ok]))
        A(f"  HE   3-seed 均值 AUC = {hm:.4f}")
        A(f"  HE+PR 3-seed 均值 AUC = {pm:.4f}")
        A(f"  均值差 = {(pm-hm)*100:+.2f} AUC 点")
        if pair:
            A(f"  配对差均值 = {d.mean():+.2f} ± {ds:.2f} AUC 点"
              f"（{'PR 有正贡献' if d.mean() > 0 else 'PR 无正贡献'}）")
        if pa and all(pa.get(s) for s in seeds):
            am = float(np.mean([pa[s]['AUC'] for s in seeds]))
            A(f"  历史 baseline Phase A 3-seed 均值 = {am:.4f}"
              f"（配置复现确认用，非 PR 的对照）")
    A("")
    A("指标口径: 6 个主指标见各 seed 的 metrics.json；AUC=roc_auc(P(class1))，"
      "Recall=sensitivity_class_1，Specificity=sensitivity_class_0。")
    A("=" * 88)
    (PILOT_ROOT / "summary.txt").write_text("\n".join(L) + "\n")
    print(f"[pilot] wrote {PILOT_ROOT/'summary.txt'}")


if __name__ == "__main__":
    sys.exit(main())
