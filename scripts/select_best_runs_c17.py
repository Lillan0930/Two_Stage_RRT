#!/usr/bin/env python
"""为每个组合选出当前最优 run / seed / checkpoint —— 判据是 **Validation AUC**。

规格（用户口径）：
    主判据  : Validation AUC 最高
    并列时  : 依次看 Validation F1 → Validation Accuracy
    禁止    : 根据 Test AUC 选 seed / checkpoint
    Test AUC: 只记录，不参与选择

⚠ 本数据集（C17）跑的是 ``Train / Test-as-Val``：官方 test 集
  （patient_100..199，``data.val_start=100``）**就是**训练期的 val loader，
  没有独立的第三份 holdout。因此本表里

      Val AUC  ==  Test AUC

  两列是**同一次计算**的结果，"Test AUC 不参与选择"在本协议下无法成立。
  这不是本脚本的取舍，是数据划分决定的（见 config.yaml 的 protocol 段）。

  另注：``best_model.pt`` 本身就是 trainer 按 ``monitor=val_auc`` +
  EarlyStopping(patience=15, min_epochs=10) 存下来的，也就是「val_auc 最优
  checkpoint」。所以按本表判据选出来的 checkpoint，与 trainer 自己存的
  best 是同一个，**不存在另一套可选**。

输出：
    best_runs_by_val_auc.txt   人读汇总表
    best_runs_by_val_auc.csv   机器可读（含 Model Path）
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ours_main_c17_abmil as O                               # noqa: E402

#: 表里 HE-only 排最后（它是参照臂，不是 §3 的 14 个组合之一）
#: 两个完成判据签名不同（is_completed(name, seed) vs he_is_completed(seed)），
#: 这里统一成 donefn(seed)。
ROWS = [(n, (lambda s, _n=n: O.seed_dir(_n, s)),
         (lambda s, _n=n: O.is_completed(_n, s)))
        for n, _ in O.COMBINATIONS]
ROWS += [("HE", O.he_seed_dir, O.he_is_completed)]


def read_history(sd):
    p = sd / "history.csv"
    if not p.is_file():
        return []
    with open(p, newline="") as f:
        out = []
        for row in csv.DictReader(f):
            try:
                out.append({
                    "epoch": int(float(row["epoch"])),
                    "val_auc": float(row["val_auc"]),
                    # 并列判据用 history 里的 val_f1 / val_accuracy
                    "val_f1": float(row["val_f1"]),
                    "val_accuracy": float(row["val_accuracy"]),
                })
            except (KeyError, TypeError, ValueError):
                continue        # 缺列/空值的行直接跳过，不让它污染 argmax
        return out


def best_epoch_of(hist):
    """主判据 val_auc 降序；并列依次 val_f1、val_accuracy 降序；再并列取更早的 epoch
    （trainer 存 best 用的是严格改进，即并列时保留**先**出现的那个）。"""
    if not hist:
        return None
    return max(hist, key=lambda r: (r["val_auc"], r["val_f1"],
                                    r["val_accuracy"], -r["epoch"]))


def audit():
    """逐 run 审计两件事：
       1) history 的 argmax(val_auc) 是否就是 metrics.json 的 best_epoch
       2) 该 epoch 的 val_auc 是否等于 metrics.json 记录的 AUC
    返回审计结论，用于在报告里如实说明 Val/Test 同源。"""
    n = ok_epoch = ok_auc = 0
    mismatch = []
    for name, sdfn, donefn in ROWS:
        for s in O.load_or_create_seeds()[0]:
            if not donefn(s):
                continue
            sd = sdfn(s)
            hist = read_history(sd)
            m = json.loads((sd / "metrics.json").read_text())
            be = best_epoch_of(hist)
            if be is None:
                continue
            n += 1
            if be["epoch"] == m["best_epoch"]:
                ok_epoch += 1
            else:
                mismatch.append((name, s, "epoch", be["epoch"], m["best_epoch"]))
            if abs(be["val_auc"] - m["AUC"]) < 1e-9:
                ok_auc += 1
            else:
                mismatch.append((name, s, "auc", be["val_auc"], m["AUC"]))
    return n, ok_epoch, ok_auc, mismatch


def main():
    seeds = O.load_or_create_seeds()[0]

    print("=" * 78)
    print("Val AUC 与 Test AUC 的同源性审计")
    print("=" * 78)
    n, ok_epoch, ok_auc, mismatch = audit()
    print(f"审计 run 数                             : {n}")
    print(f"argmax(history.val_auc) == best_epoch   : {ok_epoch}/{n}")
    print(f"该 epoch 的 val_auc == metrics['AUC']   : {ok_auc}/{n}")
    if mismatch:
        print("不一致明细（最多 10 条）:")
        for row in mismatch[:10]:
            print("   ", row)
    print()

    table, csv_rows = [], []
    for name, sdfn, donefn in ROWS:
        cands = []
        for s in seeds:
            if not donefn(s):
                continue
            sd = sdfn(s)
            hist = read_history(sd)
            be = best_epoch_of(hist)
            if be is None:
                continue
            m = json.loads((sd / "metrics.json").read_text())
            model = sd / "best_model.pt"
            cands.append({
                "combination": name,
                "seed": s,
                "epoch": be["epoch"],
                "val_auc": be["val_auc"],
                "val_f1": be["val_f1"],
                "val_accuracy": be["val_accuracy"],
                "test_auc": m["AUC"],
                "model_path": str(model),
                "model_exists": model.is_file(),
                "metrics_auc": m["AUC"],
            })
        if not cands:
            table.append((name, None))
            continue
        # 组合内选一个 seed：同样按 val_auc → val_f1 → val_accuracy
        best = max(cands, key=lambda c: (c["val_auc"], c["val_f1"],
                                         c["val_accuracy"], -c["seed"]))
        table.append((name, best))
        csv_rows.append(best)

    hdr = (f"{'Combination':<24}{'Best Seed':>10}{'Best Epoch':>11}"
           f"{'Val AUC':>10}{'Val F1':>9}{'Test AUC':>10}  Model Path")
    lines = []
    lines.append("=" * len(hdr))
    lines.append("C17 + ABMIL —— 各组合当前最优 run（判据：Validation AUC）")
    lines.append("=" * len(hdr))
    lines.append(f"结果根目录 : {O.OUT_ROOT}")
    lines.append(f"Seeds      : {', '.join(str(s) for s in seeds)}")
    lines.append("")
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for name, best in table:
        is_base = name == "HE"
        if is_base:
            lines.append("-" * len(hdr))       # 参照臂与 14 组合分栏
        label = f"[baseline] {name}" if is_base else name
        if best is None:
            lines.append(f"{label:<24}{'—':>10}{'—':>11}{'—':>10}{'—':>9}{'—':>10}  —")
            continue
        lines.append(f"{label:<24}{best['seed']:>10}{best['epoch']:>11}"
                     f"{best['val_auc']:>10.4f}{best['val_f1']:>9.4f}"
                     f"{best['test_auc']:>10.4f}  {best['model_path']}")
    lines.append("-" * len(hdr))
    lines.append("")
    lines.append("口径说明")
    lines.append("  · 选择判据: Validation AUC 最高；并列依次看 Validation F1、Validation Accuracy；")
    lines.append("    仍并列则取更早的 epoch（trainer 存 best 用严格改进，并列保留先出现的）。")
    lines.append("  · Val AUC 取自该 run 的 history.csv 对应 epoch 行；Test AUC 取自 metrics.json 的 AUC，")
    lines.append("    仅作记录，未参与任何选择。")
    lines.append("  · 本数据集跑的是 Train / Test-as-Val：patient_100..199 既是 val 又是 test，")
    lines.append("    因此 Val AUC 与 Test AUC 是同一列数值，两列必然相等。")
    lines.append("  · best_model.pt 由 trainer 按 monitor=val_auc 存下，本来就是 val_auc 最优 checkpoint。")
    lines.append("  · 本表是**选优结果**（每组合取 10 个 seed 里最好的一个），只能用于可视化 / 案例展示，")
    lines.append("    不能替代 mean ± std 作为论文主结果。")
    lines.append("=" * len(hdr))

    txt = "\n".join(lines) + "\n"
    print(txt)

    (O.OUT_ROOT / "best_runs_by_val_auc.txt").write_text(txt)
    cols = ["combination", "seed", "epoch", "val_auc", "val_f1", "val_accuracy",
            "test_auc", "metrics_auc", "model_path", "model_exists"]
    with open(O.OUT_ROOT / "best_runs_by_val_auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"已写出: {O.OUT_ROOT / 'best_runs_by_val_auc.txt'}")
    print(f"已写出: {O.OUT_ROOT / 'best_runs_by_val_auc.csv'}")


if __name__ == "__main__":
    main()
