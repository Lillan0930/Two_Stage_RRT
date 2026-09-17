#!/usr/bin/env python
"""冻结 C17 + ABMIL 主实验配置 → ``final_config.yaml``，并生成 ``seeds.json``。

规格 §6：``final_config.yaml`` 一旦确定，后续 14 个组合全部冻结，不再调参。

本脚本只做三件事，**不做任何搜索/调参**：

1. 生成（或读取）``seeds.json`` —— §8 要求：42 + 9 个随机唯一整数，
   **只生成一次**，14 个组合共用，重启读文件、绝不重新随机。
2. 用 ``build_config`` 把 14 个组合逐个解析出来，**机器校验**它们之间
   除 ``data.modalities`` / seed 外逐字段相同 —— 这是"公平性"的硬证据。
3. 落盘 ``final_config.yaml``：冻结模板 + 14 组合清单 + seeds + 校验结论。

用法::

    python scripts/finalize_c17_config.py
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ours_main_c17_abmil import (                              # noqa: E402
    COMBINATIONS, FIXED_SEED, GPU_POOL, MODALITY_ORDER, N_SEEDS, OUT_ROOT,
    TRAINING, build_config, load_or_create_seeds,
)

FINAL_PATH = OUT_ROOT / "final_config.yaml"

#: 允许随 (组合, seed) 变化的字段路径 —— 除此之外必须逐字段相同
ALLOWED_VARYING = {
    ("data", "modalities"),
    ("data", "dir_mapping"),
    ("data", "sample_seed"),
    ("model", "init_seed"),
    ("environment", "seed"),
    ("seeds", "model_seed"),
    ("seeds", "sampling_seed"),
    ("output", "save_dir"),
    ("output", "log_dir"),
    ("output", "img_dir"),
    ("experiment", "combination"),
    ("experiment", "modalities"),
    ("experiment", "seed"),
}


def flatten(d, prefix=()):
    out = {}
    for k, v in d.items():
        key = prefix + (k,)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def masked(flat):
    """丢掉**允许随组合/seed 变化**的整棵子树（含嵌套 dict 展开出的子键）。

    ``ALLOWED_VARYING`` 里存的是字段路径前缀：``('data','dir_mapping')``
    会把 ``data.dir_mapping.HE`` / ``data.dir_mapping.PR`` … 一并放行。
    """
    out = {}
    for k, v in flat.items():
        if any(k[:len(p)] == p for p in ALLOWED_VARYING):
            continue
        out[k] = v
    return out


def main():
    seeds, seed_meta, created = load_or_create_seeds()
    print(f"[finalize] seeds.json {'CREATED' if created else 'reused'}: {seeds}")

    # ── 逐组合解析，机器校验公平性 ──
    ref_name, ref_mods = COMBINATIONS[0]
    ref = masked(flatten(build_config(ref_name, ref_mods, seeds[0])))

    violations = []
    for name, mods in COMBINATIONS:
        for seed in seeds:
            cur = masked(flatten(build_config(name, mods, seed)))
            for k in sorted(set(cur) | set(ref)):
                if k not in cur:
                    violations.append(f"{name}/s{seed}: 缺字段 {'.'.join(k)}")
                elif k not in ref:
                    violations.append(f"{name}/s{seed}: 多出字段 {'.'.join(k)}")
                elif cur[k] != ref[k]:
                    violations.append(
                        f"{name}/s{seed}: {'.'.join(k)} = {cur[k]!r} ≠ 基准 {ref[k]!r}")
    if violations:
        print("[finalize] ✗ 公平性校验失败：")
        for v in violations[:40]:
            print("   ", v)
        return 1
    n_checked = len(COMBINATIONS) * len(seeds)
    print(f"[finalize] ✓ 公平性校验通过：{n_checked} 个 (组合,seed) 除 "
          f"modalities/seed/输出路径外逐字段相同（{len(ref)} 个不变字段）")

    # ── 落盘 ──
    template = build_config(ref_name, ref_mods, seeds[0])
    frozen = {
        "data": {k: v for k, v in template["data"].items()
                 if k not in ("modalities", "dir_mapping", "sample_seed")},
        "model": {k: v for k, v in template["model"].items() if k != "init_seed"},
        "training": template["training"],
        "data_split": template["data_split"],
        "environment": {k: v for k, v in template["environment"].items() if k != "seed"},
        "protocol": template["protocol"],
        "experiment": {"name": template["experiment"]["name"],
                       "stage2_type": template["experiment"]["stage2_type"],
                       "fusion": template["experiment"]["fusion"],
                       "mil": template["experiment"]["mil"],
                       "loss": template["experiment"]["loss"]},
    }

    doc = {
        "#": ("C17 + ABMIL 主实验冻结配置。规格 §6：本文件一旦确定，14 个组合"
              "全部冻结，不再调参。"),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "C17",
        "task": "二分类（label ∈ {0,1}），沿用历史任务定义，未改为二/四分类以外口径",
        "modality_order": MODALITY_ORDER,
        "seeds_file": str(OUT_ROOT / "seeds.json"),
        "seeds": seeds,
        "n_seeds": N_SEEDS,
        "gpu_pool": GPU_POOL,
        "combinations": [{"name": n, "modalities": m} for n, m in COMBINATIONS],
        "n_combinations": len(COMBINATIONS),
        "n_runs_total": len(COMBINATIONS) * N_SEEDS,
        "frozen_config": frozen,
        "provenance": {
            "encoder_cfg": ("取自 comparative_exp/RRT+abMIL/C17/config_full.yaml 的 "
                            "C17 HE 最优 RRT 参数；5 个染色共用同一份超参，权重完全独立"),
            "model_geometry": "config_full.yaml（mlp_dim/n_layers/n_heads/drop_path/"
                              "trans_dropout/epeg/crmsa_k/cr_msa/all_shortcut）",
            "mil": "config_full.yaml 的 abmil_hidden_dim=64 与 dropout=0.05",
            "training": "config_full.yaml 的 training 段（epochs/batch/lr/wd/patience）",
            "split": "沿用历史患者级 split patient_000..099 / patient_100..199（未改）",
            "label": "沿用历史 /home/Public/lillan/data/C17_binary_label.csv（未改）",
        },
        "known_deltas_vs_historical_baseline": [
            "模型：历史 RRT_ABMIL（单模态专用）→ HEAuxUnifiedModel（HE-anchored 统一模型）",
            "优化器：历史 optim.AdamW → 本项目 optim.Adam",
            "scheduler：历史 CosineAnnealingLR(eta_min=lr*0.01) → 本项目无 eta_min(=0)",
            "patch 采样实现：历史未播种 torch.randperm → 本项目稳定 hash 采样（协议等价，"
            "且新流程必须共享索引以保持五染色空间对应）",
        ],
        "fairness_check": {
            "checked_runs": n_checked,
            "invariant_fields": len(ref),
            "allowed_varying": sorted('.'.join(k) for k in ALLOWED_VARYING),
            "result": "PASS — 14 个组合之间除 modalities 与 seed 外逐字段相同",
        },
    }

    with open(FINAL_PATH, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False,
                       width=100, allow_unicode=True)
    sha = hashlib.sha256(FINAL_PATH.read_bytes()).hexdigest()
    with open(FINAL_PATH, "a") as f:
        f.write(f"\n# sha256: {sha}\n")
    print(f"[finalize] wrote {FINAL_PATH}")
    print(f"[finalize] sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
