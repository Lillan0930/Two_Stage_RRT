#!/usr/bin/env python
"""C16 五染色特征完整性审计 —— Ours 主实验（C16 + ABMIL）正式运行前的准入检查。

检查项（对应正式运行规格 §2）：

    1. 每个染色目录存在，且 ``*.pt`` 文件名（stem）集合完整对应 label CSV
    2. 270 train / 129 val-as-test 的 slide 数在**五个染色上完全一致**
    3. feature dim = 768、patch 数 > 0、dtype 为浮点
    4. 同一 slide_id 在多个子目录重复出现（duplicate）
    5. NaN / Inf 全量或抽样扫描

**本脚本绝不取交集、绝不自动修复。** 任一染色缺失样本都会改变 270/129，
此时以非零码退出并在报告里逐条列出，交由人工决定。
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 固定模态顺序 —— 14 个组合与所有报告都依赖它
MODALITY_ORDER = ["HE", "PR", "ER", "HER2", "Ki67"]

#: 染色 → 特征目录名（与 configs/he_aux_unified/*.json 的 dir_mapping 一致）
DIR_MAPPING = {
    "HE": "C16_HE_features",
    "PR": "C16_PR_features",
    "ER": "C16_ER_features",
    "HER2": "C16_HER2_features",
    "Ki67": "C16_Ki67_features",
}

SUBDIRS = ("normal", "tumor", "test")


def read_label_ids(path):
    """返回 ``(id 列表, id → label)``；只读 CSV，不排序、不去重，便于发现重复。"""
    ids, mapping = [], {}
    with open(path) as f:
        header = f.readline().strip().split(",")
        assert header[:2] == ["slide_id", "label"], f"unexpected header in {path}: {header}"
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, lab = line.split(",")[:2]
            ids.append(sid)
            mapping[sid] = int(lab)
    return ids, mapping


def scan_one_file(args):
    """子进程：读一个 ``.pt``，返回 (路径, patch 数, dim, dtype, 是否有限, 错误)。"""
    path, do_values = args
    import torch  # 在子进程里导入，避免 fork 后 CUDA/线程状态问题
    try:
        t = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - 审计要把任何读取失败都报出来
        return (str(path), -1, -1, "?", False, f"load failed: {exc!r}")

    if not hasattr(t, "shape") or t.dim() != 2:
        return (str(path), -1, -1, type(t).__name__, False,
                f"expected 2-D tensor, got {type(t).__name__} shape="
                f"{tuple(t.shape) if hasattr(t, 'shape') else '?'}")
    n, d = int(t.shape[0]), int(t.shape[1])
    dtype = str(t.dtype)
    finite = True
    err = ""
    if do_values:
        finite = bool(torch.isfinite(t).all())
        if not finite:
            err = (f"non-finite values: nan={int(torch.isnan(t).sum())} "
                   f"inf={int(torch.isinf(t).sum())}")
    del t
    return (str(path), n, d, dtype, finite, err)


def collect_files(mod_dir):
    """→ (stem → [相对路径, ...])。同一 stem 出现在多个子目录即 duplicate。"""
    found = {}
    for sub in SUBDIRS:
        d = mod_dir / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.pt")):
            found.setdefault(f.stem, []).append(f"{sub}/{f.name}")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feature-root", default="/home/Public/lillan/features_result/C16_features")
    ap.add_argument("--train-labels",
                    default=str(REPO_ROOT / "data/C16_labels/c16_train_labels.csv"))
    ap.add_argument("--val-labels",
                    default=str(REPO_ROOT / "data/C16_labels/c16_test_labels.csv"))
    ap.add_argument("--out", required=True, help="审计报告输出路径 (data_audit.txt)")
    ap.add_argument("--value-scan", choices=["none", "sample", "all"], default="all",
                    help="NaN/Inf 扫描范围：全部文件 / 每染色抽样 / 不扫")
    ap.add_argument("--sample-n", type=int, default=40, help="--value-scan sample 时每染色抽样数")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    feature_root = Path(args.feature_root)
    train_ids, train_map = read_label_ids(args.train_labels)
    val_ids, val_map = read_label_ids(args.val_labels)

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s, flush=True)

    emit("=" * 78)
    emit("C16 数据审计 — Ours 主实验 (HEAuxUnifiedModel + ABMIL)")
    emit("=" * 78)
    emit(f"feature_root : {feature_root}")
    emit(f"train labels : {args.train_labels}")
    emit(f"val labels   : {args.val_labels}")
    emit(f"NaN/Inf scan : {args.value_scan}")
    emit()

    hard_fail = []          # 会改变 270/129 的问题 ⇒ 必须停工
    warnings = []           # 不改变样本数、但需要知情的问题

    # ── 0. label CSV 自身 ──────────────────────────────────────────────
    emit("── 0. label CSV ──")
    for name, ids, mapping in (("train", train_ids, train_map), ("val", val_ids, val_map)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        emit(f"  {name}: {len(ids)} 行, 唯一 {len(set(ids))}, "
             f"label 分布 {json.dumps({str(k): list(mapping.values()).count(k) for k in sorted(set(mapping.values()))})}")
        if dup:
            emit(f"    !! 重复 slide_id: {dup}")
            hard_fail.append(f"{name} label CSV 有重复 slide_id: {dup}")
    overlap = sorted(set(train_ids) & set(val_ids))
    emit(f"  train ∩ val 重叠: {len(overlap)}")
    if overlap:
        hard_fail.append(f"train/val slide_id 重叠 {len(overlap)} 个: {overlap[:10]}")
    emit(f"  合计唯一 slide: {len(set(train_ids) | set(val_ids))}  (期望 399 = 270 + 129)")
    if len(train_ids) != 270 or len(val_ids) != 129:
        hard_fail.append(f"label 行数不是 270/129，而是 {len(train_ids)}/{len(val_ids)}")
    emit()

    expected_ids = set(train_ids) | set(val_ids)

    # ── 1-4. 逐染色结构检查 ────────────────────────────────────────────
    emit("── 1-4. 逐染色结构 ──")
    per_mod_ids = {}
    for mod in MODALITY_ORDER:
        mod_dir = feature_root / DIR_MAPPING[mod]
        emit(f"\n  [{mod}]  {mod_dir}")
        if not mod_dir.is_dir():
            emit("    !! 目录不存在")
            hard_fail.append(f"{mod}: 特征目录不存在 {mod_dir}")
            continue

        found = collect_files(mod_dir)
        per_mod_ids[mod] = set(found)

        dupes = {s: v for s, v in found.items() if len(v) > 1}
        missing = sorted(expected_ids - set(found))
        extra = sorted(set(found) - expected_ids)
        n_train_found = len(set(found) & set(train_ids))
        n_val_found = len(set(found) & set(val_ids))

        emit(f"    .pt 文件 {sum(len(v) for v in found.values())} 个, 唯一 slide {len(found)}")
        emit(f"    覆盖 train {n_train_found}/270, val-as-test {n_val_found}/129")
        emit(f"    子目录分布: " + ", ".join(
            f"{sub}={len(list((mod_dir / sub).glob('*.pt'))) if (mod_dir / sub).is_dir() else 'NA'}"
            for sub in SUBDIRS))
        if dupes:
            emit(f"    !! duplicate ({len(dupes)}): " +
                 "; ".join(f"{s} → {v}" for s, v in list(dupes.items())[:5]))
            hard_fail.append(f"{mod}: {len(dupes)} 个 slide 在多个子目录重复")
        if missing:
            emit(f"    !! missing ({len(missing)}): {missing[:10]}")
            hard_fail.append(f"{mod}: 缺 {len(missing)} 个 label 中的 slide: {missing[:10]}")
        if extra:
            emit(f"    ?? extra ({len(extra)}): {extra[:10]}  (不在 label 中，会被数据集忽略)")
            warnings.append(f"{mod}: {len(extra)} 个特征文件不在 label 中: {extra[:10]}")
        if not missing and not dupes and n_train_found == 270 and n_val_found == 129:
            emit("    OK: 270/129 完整")

    emit()
    emit("── 1b. 染色间一致性 ──")
    if len(per_mod_ids) == len(MODALITY_ORDER):
        base = per_mod_ids["HE"]
        for mod in MODALITY_ORDER[1:]:
            diff = expected_ids - per_mod_ids[mod]
            only_he = base - per_mod_ids[mod]
            emit(f"  HE vs {mod:5s}: 缺失差集 {len(diff)}  "
                 f"(HE 有而 {mod} 没有: {len(only_he)})")
            if diff:
                hard_fail.append(f"{mod} 相对完整 label 缺 {len(diff)} 个 slide")
        allmod_common = set.intersection(*per_mod_ids.values())
        emit(f"  五染色共同覆盖: {len(allmod_common)}  (五染色都为 399 才说明无需取交集)")
        if len(allmod_common) != 399:
            hard_fail.append(f"五染色共同覆盖只有 {len(allmod_common)}，会改变 270/129 ⇒ 不自动取交集")
    emit()

    # ── 5. 逐个文件读 shape / dtype / NaN ──────────────────────────────
    if args.value_scan != "none":
        emit("── 5. 逐文件 shape / dtype / NaN / Inf ──")
        jobs = []
        for mod in MODALITY_ORDER:
            if mod not in per_mod_ids:
                continue
            files = sorted((feature_root / DIR_MAPPING[mod]).glob("*/*.pt"))
            if args.value_scan == "sample":
                rng = random.Random(0)
                files = rng.sample(files, min(args.sample_n, len(files)))
            jobs.extend((str(f), args.value_scan == "all") for f in files)

        emit(f"  待扫描 {len(jobs)} 个文件 (workers={args.workers}) …")
        bad_shape, bad_finite, bad_load = [], [], []
        dims, dtypes, patch_counts = set(), set(), []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (path, n, d, dt, finite, err) in enumerate(
                    ex.map(scan_one_file, jobs, chunksize=8), 1):
                if n < 0 and "load failed" in err:
                    bad_load.append((path, err))
                    continue
                if n < 0:
                    bad_shape.append((path, err))
                    continue
                dims.add(d)
                dtypes.add(dt)
                patch_counts.append(n)
                if d != 768:
                    bad_shape.append((path, f"dim={d}"))
                if n == 0:
                    bad_shape.append((path, "0 patches"))
                if not finite:
                    bad_finite.append((path, err))
                if i % 500 == 0:
                    emit(f"    … {i}/{len(jobs)}")

        emit(f"  dim 集合: {sorted(dims)}  (期望 {{768}})")
        emit(f"  dtype 集合: {sorted(dtypes)}  (期望 float32)")
        if patch_counts:
            emit(f"  patch 数: min={min(patch_counts)} max={max(patch_counts)} "
                 f"mean={sum(patch_counts) / len(patch_counts):.1f} "
                 f"中位数={sorted(patch_counts)[len(patch_counts) // 2]}")
        emit(f"  load 失败: {len(bad_load)}")
        for p, e in bad_load[:5]:
            emit(f"    !! {p}: {e}")
        emit(f"  shape/内容异常: {len(bad_shape)}")
        for p, e in bad_shape[:5]:
            emit(f"    !! {p}: {e}")
        emit(f"  非有限值: {len(bad_finite)}")
        for p, e in bad_finite[:5]:
            emit(f"    !! {p}: {e}")

        if bad_load:
            hard_fail.append(f"{len(bad_load)} 个特征文件无法读取")
        if bad_shape:
            hard_fail.append(f"{len(bad_shape)} 个特征文件 shape/dim 异常")
        if bad_finite:
            hard_fail.append(f"{len(bad_finite)} 个特征文件含 NaN/Inf")
        if dims and dims != {768}:
            hard_fail.append(f"feature dim 不是 768: {sorted(dims)}")
    else:
        emit("── 5. 逐文件扫描：已跳过 (--value-scan none) ──")

    # ── 结论 ───────────────────────────────────────────────────────────
    emit()
    emit("=" * 78)
    if hard_fail:
        emit(f"结论：FAIL — {len(hard_fail)} 项阻断问题（270/129 无法保证）")
        for i, m in enumerate(hard_fail, 1):
            emit(f"  [{i}] {m}")
        emit("  ⇒ 按规格 §2，停止运行，等待人工处理；不自动取交集。")
    else:
        emit("结论：PASS — 五染色 270/129 完整对应，feature dim 768，无 NaN/Inf/重复/缺失")
    if warnings:
        emit(f"提示：{len(warnings)} 项非阻断警告")
        for i, m in enumerate(warnings, 1):
            emit(f"  ({i}) {m}")
    emit("=" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\n报告已写入 {out}")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
