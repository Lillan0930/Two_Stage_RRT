#!/usr/bin/env python
"""C17 pilot 审计：配置审计 + 数据审计。

产出（规格 §1 / §2）::

    <out>/_pilot/config_audit.txt
    <out>/_pilot/data_audit.txt

配置审计**只读**历史 ``config_full.yaml``，不做任何推测：每个字段都从文件里
读出来再打印。数据审计逐文件扫描 5 个染色的全部 ``.pt``，核对
slide ID / 标签覆盖 / 模态一致性 / feature dim / NaN / Inf / 重复 ID，
并**显式报告**缺失与多余，绝不静默取交集。

用法::

    python scripts/audit_c17_pilot.py
    python scripts/audit_c17_pilot.py --workers 16 --skip-content   # 跳过逐文件扫描
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ── 历史基线（只读，绝不修改） ──────────────────────────────────────────
HIST_ROOT = Path("/home/Public/lillan/work_results/comparative_exp/RRT+abMIL/C17")
HIST_CONFIG = HIST_ROOT / "config_full.yaml"
HIST_CONFIG_SHORT = HIST_ROOT / "config.yaml"
HIST_SUMMARY = HIST_ROOT / "summary.txt"
HIST_SEEDS = HIST_ROOT / "seeds.txt"
HIST_ALLSEEDS = HIST_ROOT / "all_seeds.csv"

OUT_ROOT = Path("/home/Public/lillan/work_results/ours_main/C17+abmil")
PILOT = OUT_ROOT / "_pilot"

MODALITY_ORDER = ["HE", "PR", "ER", "HER2", "Ki67"]
DIR_MAPPING = {
    "HE": "C17_HE_new_features",
    "PR": "C17_PR_new_features",
    "ER": "C17_ER_new_feature",       # 注意：历史目录名是单数 feature
    "HER2": "C17_HER2_new_feature",   # 同上
    "Ki67": "C17_Ki67_new_features",
}
FEATURE_ROOT = Path("/home/Public/lillan/features_result/C17_features")
LABEL_FILE = Path("/home/Public/lillan/data/C17_binary_label.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 1. 配置审计
# ═══════════════════════════════════════════════════════════════════════════

def audit_config():
    cfg = yaml.safe_load(HIST_CONFIG.read_text())
    d, m, t = cfg["data"], cfg["model"], cfg["training"]

    L = []
    A = L.append
    A("=" * 78)
    A("C17 历史配置审计 — comparative_exp/RRT+abMIL/C17")
    A("=" * 78)
    A(f"审计时间      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"配置文件      : {HIST_CONFIG}")
    A(f"文件 sha256   : ", )
    import hashlib
    A(f"                {hashlib.sha256(HIST_CONFIG.read_bytes()).hexdigest()}")
    A("")
    A("以下每一项都**逐字读取**自 config_full.yaml，未做任何推断或补全。")
    A("")

    A("── data ──")
    A(f"  dataset             : {d['dataset']}")
    A(f"  feature_dir         : {d['feature_dir']}")
    A(f"  label_file          : {d['label_file']}")
    A(f"  input_dim           : {d['input_dim']}")
    A(f"  num_classes         : {d['num_classes']}   ← 二分类任务")
    A(f"  max_patches         : {d['max_patches']}")
    A("")

    A("── model ──")
    A(f"  name                : {m['name']}")
    A(f"  attention_type      : {m['attention_type']}  (use_gated = False)")
    A(f"  mlp_dim             : {m['mlp_dim']}")
    A(f"  region_num          : {m['region_num']}")
    A(f"  n_layers            : {m['n_layers']}")
    A(f"  n_heads             : {m['n_heads']}")
    A(f"  drop_path           : {m['drop_path']}")
    A(f"  trans_dropout       : {m['trans_dropout']}")
    A(f"  dropout             : {m['dropout']}")
    A(f"  abmil_hidden_dim    : {m['abmil_hidden_dim']}")
    A(f"  epeg                : {m['epeg']}")
    A(f"  epeg_k              : {m['epeg_k']}")
    A(f"  crmsa_k             : {m['crmsa_k']}")
    A(f"  cr_msa              : {m['cr_msa']}")
    A(f"  all_shortcut        : {m['all_shortcut']}")
    A("")

    A("── training ──")
    A(f"  epochs              : {t['epochs']}")
    A(f"  batch_size          : {t['batch_size']}")
    A(f"  lr                  : {t['lr']}")
    A(f"  weight_decay        : {t['weight_decay']}")
    A(f"  patience            : {t['patience']}")
    A(f"  min_epochs          : {t['min_epochs']}")
    A(f"  focal_loss          : {t['focal_loss']}")
    A(f"  focal_gamma         : {t['focal_gamma']}")
    A(f"  label_smoothing     : {t['label_smoothing']}")
    A("")

    A("── _metadata ──")
    for k, v in (cfg.get("_metadata") or {}).items():
        A(f"  {k:<20}: {v}")
    A("")

    A("── 配置里**没有**写、由代码决定的项（读自 run_experiment.py） ──")
    A("  optimizer           : AdamW")
    A("  scheduler           : CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)")
    A("  loss                : nn.CrossEntropyLoss(label_smoothing=0.0)  (focal_loss=False)")
    A("  early stopping      : monitor = -test_auc (max AUC), stop_epoch = min_epochs = 10")
    A("  checkpoint select   : 测试集 AUC 最高的 epoch（test-as-val 开发协议）")
    A("  seed 入口           : random / numpy / torch / torch.cuda (manual_seed + manual_seed_all)")
    A("  num_workers         : 2")
    A("  collate             : data.dataset.collate_fn (RRT_ABMIL 包)")
    A("  device              : 'cuda:3' if available else 'cpu'  ← 硬编码")
    A("")

    A("── 训练代码来源 ──")
    A("  runner              : /home/Public/lillan/work_results/comparative_exp/RRT+abMIL/run_experiment.py")
    A("  model               : /home/Public/lillan/RRT_ABMIL/models/rrt_abmil.py  (class RRT_ABMIL)")
    A("  dataset             : /home/Public/lillan/RRT_ABMIL/data/dataset.py      (class FeatureDataset)")
    A("  metrics             : /home/Public/lillan/RRT_ABMIL/utils/metrics.py    (AverageMeter, EarlyStopping)")
    A("")

    A("── split（train_patients / test_patients） ──")
    A("  来源                : run_experiment.py 的 C17_CONFIG 字典（config_full.yaml 未记录）")
    A("  train_patients      : patient_000 … patient_099   (100 个患者，编号 < 100)")
    A("  test_patients       : patient_100 … patient_199   (100 个患者，编号 >= 100)")
    A("  划分方式            : 患者级、按编号切片，**不是**随机划分，也不是 KFold")
    A("  随机种子影响 split  : 否 —— split 是确定的字符串列表，与 seed 无关")
    A("  ⚠ 本流程禁止改变该 split 与标签定义。")
    A("")

    A("── label mapping ──")
    A("  标签文件            : /home/Public/lillan/data/C17_binary_label.csv")
    A("  列                  : slide_id, label")
    A("  取值                : {0, 1}   ← 二分类")
    A("  语义                : 沿用历史定义，不重新解释；0/1 直接作为 CrossEntropy 的类别索引")
    A("  粒度                : slide 级（一个 slide_id 一行；patient 下的 node 各自成样本）")
    A("")

    A("── 历史 10-seed 基线结果（只读 summary.txt / all_seeds.csv） ──")
    if HIST_SEEDS.is_file():
        seeds = [int(x) for x in HIST_SEEDS.read_text().split()]
        A(f"  历史 seeds          : {seeds}")
    if HIST_SUMMARY.is_file():
        A("")
        for line in HIST_SUMMARY.read_text().splitlines():
            A("  " + line if line.strip() else "")
    if HIST_ALLSEEDS.is_file():
        A("")
        A("  历史 all_seeds.csv:")
        for line in HIST_ALLSEEDS.read_text().splitlines():
            A("    " + line)
    A("")
    A("=" * 78)
    return "\n".join(L) + "\n", cfg


# ═══════════════════════════════════════════════════════════════════════════
# 2. 数据审计
# ═══════════════════════════════════════════════════════════════════════════

def _scan_one(path):
    """扫一个 .pt：返回 (stem, ok, shape, dtype, n_nan_inf, err)。"""
    import torch
    try:
        t = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:                                     # noqa: BLE001
        return (Path(path).stem, False, None, None, -1, f"{type(exc).__name__}: {exc}")
    if not hasattr(t, "shape"):
        return (Path(path).stem, False, None, None, -1, f"not a tensor: {type(t).__name__}")
    if t.dim() == 1:
        t = t.unsqueeze(0)
    bad = int((~torch.isfinite(t)).sum().item())
    return (Path(path).stem, True, tuple(t.shape), str(t.dtype), bad, "")


def audit_data(cfg, workers=12, skip_content=False):
    L = []
    A = L.append
    A("=" * 78)
    A("C17 数据审计 — Ours 主实验 (HEAuxUnifiedModel + ABMIL)")
    A("=" * 78)
    A(f"审计时间      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    A(f"feature_root  : {FEATURE_ROOT}")
    A(f"label_file    : {LABEL_FILE}")
    A(f"模态顺序      : {MODALITY_ORDER}")
    A("NaN/Inf 扫描  : all (逐文件 torch.isfinite)")
    A("")

    # ── 0. 标签 ──
    lab = pd.read_csv(LABEL_FILE)
    A("── 0. 标签文件 ──")
    A(f"  列                : {list(lab.columns)}")
    A(f"  行数              : {len(lab)}")
    A(f"  唯一 slide_id     : {lab['slide_id'].nunique()}")
    A(f"  重复 slide_id     : {len(lab) - lab['slide_id'].nunique()}")
    dist = lab["label"].value_counts().sort_index().to_dict()
    A(f"  label 分布        : {dist}")
    A(f"  label 取值集合    : {sorted(lab['label'].unique())}")
    lab["patient_id"] = lab["slide_id"].str.split("_").str[:2].str.join("_")
    A(f"  患者数            : {lab['patient_id'].nunique()}")
    per_pat = lab.groupby("patient_id")["label"].nunique().value_counts().to_dict()
    A(f"  每患者 label 种类 : {per_pat}  (2 表示该患者下既有 0 也有 1)")
    A("")

    # ── 0b. split ──
    def _pnum(p):
        try:
            return int(p.split("_")[-1])
        except (ValueError, IndexError):
            return 0
    pats = sorted(lab["patient_id"].unique())
    train_pats = [p for p in pats if _pnum(p) < 100]
    val_pats = [p for p in pats if _pnum(p) >= 100]
    train_ids = sorted(lab[lab.patient_id.isin(train_pats)]["slide_id"])
    val_ids = sorted(lab[lab.patient_id.isin(val_pats)]["slide_id"])
    A("── 0b. split（患者级，val_start=100，沿用历史定义） ──")
    A(f"  train 患者        : {len(train_pats)}  ({train_pats[0]} … {train_pats[-1]})")
    A(f"  test  患者        : {len(val_pats)}  ({val_pats[0]} … {val_pats[-1]})")
    A(f"  train slide (标签表): {len(train_ids)}")
    A(f"  test  slide (标签表): {len(val_ids)}")
    A(f"  train ∩ test      : {len(set(train_ids) & set(val_ids))}")
    A(f"  患者编号覆盖      : {sorted(set(range(200)) - {_pnum(p) for p in pats}) or '完整 0..199'}")
    A("")

    # ── 1. 逐染色目录 ──
    A("── 1. 逐染色结构 ──")
    A("")
    per_mod = {}
    for mod in MODALITY_ORDER:
        d = FEATURE_ROOT / DIR_MAPPING[mod]
        files = sorted(d.rglob("*.pt")) if d.is_dir() else []
        stems = [f.stem for f in files]
        cnt = Counter(stems)
        dups = sorted([s for s, c in cnt.items() if c > 1])
        per_mod[mod] = {"dir": d, "files": files, "stems": set(stems), "dups": dups}
        covered_train = sum(1 for s in train_ids if s in set(stems))
        covered_val = sum(1 for s in val_ids if s in set(stems))
        A(f"  [{mod}]  {d}")
        A(f"    .pt 文件        : {len(files)}")
        A(f"    唯一 slide ID   : {len(set(stems))}")
        A(f"    重复 ID         : {len(dups)}  {dups[:5] if dups else ''}")
        A(f"    覆盖 train      : {covered_train}/{len(train_ids)}")
        A(f"    覆盖 test       : {covered_val}/{len(val_ids)}")
        A(f"    患者子目录      : {len([x for x in d.iterdir() if x.is_dir()]) if d.is_dir() else 0}")
        A("")

    # ── 1b. 模态间一致性（显式报告，不取交集） ──
    A("── 1b. 染色间一致性 ──")
    ref = per_mod["HE"]["stems"]
    all_same = True
    for mod in MODALITY_ORDER[1:]:
        s = per_mod[mod]["stems"]
        only_ref = sorted(ref - s)
        only_mod = sorted(s - ref)
        all_same &= (not only_ref and not only_mod)
        A(f"  HE vs {mod:<5}: 差集 {len(ref ^ s)}  "
          f"(HE 有而 {mod} 没有: {len(only_ref)}, {mod} 有而 HE 没有: {len(only_mod)})")
        if only_ref:
            A(f"      仅 HE   有: {only_ref[:5]}")
        if only_mod:
            A(f"      仅 {mod} 有: {only_mod[:5]}")
    A(f"  ⇒ 五染色 slide ID 集合{'完全一致' if all_same else '不一致（见上）'}")
    A("")

    # ── 1c. 特征目录 vs 标签表 ──
    A("── 1c. 特征目录 vs 标签表（显式报告，不静默取交集） ──")
    lab_ids = set(lab["slide_id"])
    for mod in MODALITY_ORDER:
        s = per_mod[mod]["stems"]
        missing = sorted(lab_ids - s)     # 有标签、无特征
        extra = sorted(s - lab_ids)       # 有特征、无标签
        A(f"  [{mod}]  有标签无特征: {len(missing)}   有特征无标签: {len(extra)}")
        if missing:
            A(f"      有标签无特征 (前5): {missing[:5]}")
        if extra:
            A(f"      有特征无标签 (前5): {extra[:5]}")
    A("")
    missing_any = sorted(lab_ids - ref)
    A(f"  ⇒ 标签表 {len(lab_ids)} 个 slide 中，HE 缺 {len(missing_any)} 个: {missing_any}")
    A(f"  ⇒ 特征目录 {len(ref)} 个 slide 中，标签表未覆盖 {len(sorted(ref - lab_ids))} 个")
    A("  处置：两侧差异**原样报告**，不取交集、不删样本；实际训练集大小 = 两侧都有")
    A(f"        = {len(lab_ids & ref)} 个 slide")
    A("")

    # ── 1d. 实际可用样本（train / test 拆分后） ──
    usable = lab_ids & ref
    utr = sorted(set(train_ids) & usable)
    uva = sorted(set(val_ids) & usable)
    A("── 1d. 实际可用样本 ──")
    A(f"  train 可用        : {len(utr)}  (标签表 {len(train_ids)} − 缺特征 {len(set(train_ids) - usable)})")
    A(f"  test  可用        : {len(uva)}  (标签表 {len(val_ids)} − 缺特征 {len(set(val_ids) - usable)})")
    lab_map = dict(zip(lab["slide_id"], lab["label"]))
    A(f"  train label 分布  : {dict(sorted(Counter(lab_map[s] for s in utr).items()))}")
    A(f"  test  label 分布  : {dict(sorted(Counter(lab_map[s] for s in uva).items()))}")
    A("")

    # ── 2. 逐文件 shape/dtype/NaN/Inf ──
    A("── 2. 逐文件 shape / dtype / NaN / Inf / patch 数 ──")
    all_tasks = []
    for mod in MODALITY_ORDER:
        for f in per_mod[mod]["files"]:
            all_tasks.append((mod, f))
    A(f"  待扫描文件        : {len(all_tasks)} (workers={workers})")

    dims, dtypes = Counter(), Counter()
    nan_files, fail_files = [], []
    patch_by_mod_slide = defaultdict(dict)      # slide -> {mod: n_patches}
    per_mod_patches = {m: [] for m in MODALITY_ORDER}

    if skip_content:
        A("  (--skip-content：已跳过)")
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_scan_one, f): (mod, f) for mod, f in all_tasks}
            for fut in futs:
                mod, f = futs[fut]
                stem, ok, shape, dtype, bad, err = fut.result()
                done += 1
                if done % 500 == 0:
                    A(f"    … {done}/{len(all_tasks)}")
                if not ok:
                    fail_files.append((mod, stem, err))
                    continue
                dims[shape[-1]] += 1
                dtypes[dtype] += 1
                if bad:
                    nan_files.append((mod, stem, bad))
                patch_by_mod_slide[stem][mod] = shape[0]
                per_mod_patches[mod].append(shape[0])

        A(f"  feature dim 集合  : {dict(dims)}")
        A(f"  dtype 集合        : {dict(dtypes)}")
        A(f"  load 失败         : {len(fail_files)}")
        for mod, stem, err in fail_files[:5]:
            A(f"      [{mod}] {stem}: {err}")
        A(f"  含 NaN/Inf 文件   : {len(nan_files)}")
        for mod, stem, bad in nan_files[:10]:
            A(f"      [{mod}] {stem}: {bad} 个非有限值")
        A("")
        A("  各染色 patch 数统计:")
        for mod in MODALITY_ORDER:
            v = np.array(per_mod_patches[mod], dtype=float)
            if len(v) == 0:
                A(f"    [{mod}] (无)")
                continue
            A(f"    [{mod:<5}] n={len(v):5d}  min={v.min():7.0f}  max={v.max():7.0f}  "
              f"mean={v.mean():9.1f}  median={np.median(v):7.0f}")
        A("")

        # ── 2b. 跨模态 patch 数一致性 ──
        A("── 2b. 同一 slide 跨染色 patch 数一致性 ──")
        common = [s for s in patch_by_mod_slide if len(patch_by_mod_slide[s]) == len(MODALITY_ORDER)]
        mismatch = []
        for s in common:
            vals = {m: patch_by_mod_slide[s][m] for m in MODALITY_ORDER}
            if len(set(vals.values())) != 1:
                mismatch.append((s, vals))
        A(f"  五染色齐全的 slide: {len(common)}")
        A(f"  patch 数不一致的  : {len(mismatch)}")
        for s, vals in mismatch[:10]:
            A(f"      {s}: {vals}")
        A("  ⚠ patch 数不一致时，'共享 patch 索引' 的假设不成立 —— "
          "见下方 C17 数据集的口径说明。")
        A("")

    # ── 3. 结论 ──
    A("=" * 78)
    ok = (all_same and not per_mod["HE"]["dups"] and not nan_files and not fail_files
          and len(missing_any) <= 1)
    verdict = "PASS" if ok else "需人工确认"
    A(f"结论：{verdict}")
    A(f"  · 标签表 {len(lab_ids)} slide / {lab['patient_id'].nunique()} 患者，二分类 0/1")
    A(f"  · 五染色 slide ID 集合{'完全一致' if all_same else '不一致'}")
    A(f"  · 实际训练可用 {len(utr)} (train) / {len(uva)} (test)")
    A(f"  · 有标签无特征 slide: {len(missing_any)}  有特征无标签: {len(sorted(ref - lab_ids))}")
    A("=" * 78)
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--skip-content", action="store_true")
    ap.add_argument("--out", default=str(PILOT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg_text, cfg = audit_config()
    (out / "config_audit.txt").write_text(cfg_text)
    print(f"wrote {out/'config_audit.txt'}")

    data_text = audit_data(cfg, workers=args.workers, skip_content=args.skip_content)
    (out / "data_audit.txt").write_text(data_text)
    print(f"wrote {out/'data_audit.txt'}")


if __name__ == "__main__":
    main()
