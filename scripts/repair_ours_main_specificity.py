#!/usr/bin/env python
"""修正 Ours 主实验已产出结果里的 ``Specificity`` 口径 —— **不重训**。

背景
----
第一版 ``primary_metrics`` 把 Specificity 映射到了 ``specificity_class_0``。
``utils/metrics.py`` 的 per-class 循环把第 i 类当作正类，于是
``specificity_class_0`` = "以 normal 为正类时的 TN/(TN+FP)" = 肿瘤预测为肿瘤的
比例 = ``sensitivity_class_1``，也就是 **Recall 本身**。所以首版汇总表里
Recall 与 Specificity 每一行都相等 —— 那是键选错，不是模型行为。

真正的"正常类召回 / 标准 Specificity（肿瘤为正类）"是 ``sensitivity_class_0``
（二分类下与 ``specificity_class_1`` 恒等，140/140 已验证 max|Δ|=0）。

修正方式（两处，都是精确复原，不是近似）
--------------------------------------
1. ``metrics.json``：直接取同一次评估里已经存下来的
   ``secondary.specificity_class_1`` —— 这是 ``calculate_metrics`` 算出的原始
   指标，本来就是正确的特异度，只是首版没把它放进主指标。
2. ``history.csv``：逐 epoch 的特异度当时没有落盘，但可以从同一行的
   accuracy / recall 精确反解。验证集固定为 80 normal + 49 tumor = 129：

       accuracy = (TP + TN) / 129,  TP = recall × 49,  TN = spec × 80
       ⇒ spec = (accuracy × 129 − recall × 49) / 80

   这是恒等式，不含任何拟合。

只改 ``Specificity``（以及 history 的 ``val_specificity``）一列，其余指标、
checkpoint、训练日志一概不动。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ours_main_c16_abmil as O                               # noqa: E402

N_VAL_TOTAL = 129
N_VAL_NORMAL = 80      # label 0
N_VAL_TUMOR = 49       # label 1


def repair_metrics(path):
    d = json.loads(path.read_text())
    sec = d.get("secondary", {})
    correct = sec.get("specificity_class_1")
    if correct is None:
        return "no secondary.specificity_class_1"
    old = d.get("Specificity")
    if old is not None and abs(old - correct) < 1e-12:
        return "already correct"
    d["Specificity"] = float(correct)
    d["specificity_source"] = (
        "secondary.specificity_class_1 (== sensitivity_class_0, 二分类恒等)；"
        "首版误用 specificity_class_0，那个键等于肿瘤召回")
    d["metric_definitions"]["Specificity"] = (
        "sensitivity_class_0 = 正常类召回 = 标准特异度（肿瘤为正类）")
    path.write_text(json.dumps(d, indent=2) + "\n")
    return f"fixed {old:.4f} -> {correct:.4f}"


def repair_history(path):
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return "empty"
    fields = list(rows[0].keys())
    changed = 0
    for r in rows:
        acc = float(r["val_accuracy"])
        rec = float(r["val_recall"])
        spec = (acc * N_VAL_TOTAL - rec * N_VAL_TUMOR) / N_VAL_NORMAL
        spec = min(1.0, max(0.0, spec))          # 浮点噪声夹紧
        if abs(spec - float(r["val_specificity"])) > 1e-9:
            changed += 1
        r["val_specificity"] = f"{spec:.10f}"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return f"recomputed {changed}/{len(rows)} rows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(O.OUT_ROOT))
    ap.add_argument("--skip-history", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_root)

    n_ok = n_skip = 0
    for p in sorted(root.glob("*/seed*/metrics.json")):
        if "_smoke" in str(p):
            continue
        res = repair_metrics(p)
        if res.startswith("fixed"):
            n_ok += 1
        else:
            n_skip += 1
    print(f"metrics.json: 修正 {n_ok} 个, 跳过 {n_skip} 个")

    if not args.skip_history:
        h_ok = 0
        for p in sorted(root.glob("*/seed*/history.csv")):
            if "_smoke" in str(p):
                continue
            repair_history(p)
            h_ok += 1
        print(f"history.csv: 重算 {h_ok} 个")

    # 重新生成 14 个 summary.txt + all_results_summary.{txt,csv}
    seeds, _, _ = O.load_or_create_seeds(root / "seeds.json")
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import drive_ours_main_c16_abmil as D
    D.O.OUT_ROOT = root
    D.write_all_summaries(seeds)
    print(f"已重新生成 {root}/all_results_summary.txt 和 .csv")


if __name__ == "__main__":
    main()
