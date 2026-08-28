#!/usr/bin/env python3
"""训练协议受控实验 driver（3 条件 × 3 seeds，修复 sampler + LR 变量隔离）。

三个条件（各 seeds=[42,123,456]）：

  Condition 1  he_rrt_samplerfix_lr1e4
      HE-only baseline，修复 sampler 后重跑，统一 lr=1e-4（无 lr_stage1/2）。
  Condition 2  twostage_r4_noepeg_samplerfix_diff_lr
      修复 sampler，保留旧差分 LR（lr_stage1=1e-5, lr_stage2=2e-5,
      learning_rate=1e-4 仅作 optimizer 默认）。
  Condition 3  twostage_r4_noepeg_samplerfix_unified_lr1e4
      修复 sampler + 统一 LR（learning_rate=1e-4，删除 lr_stage1/2）。

所有条件：Stage1 HE/PR 各自 best config、Stage2 r4 no-EPEG（对称双边 dispatch、
concat 后 ABMIL）、270/129 test-as-val、monitor val_auc、patience 10、
num_epochs 80、cosine、max_patches 2500、sampling random、train per_epoch=True、
val/test per_epoch=False、batch 1、num_workers 2、dropout 0.25、
abmil_hidden_dim 256、weight_decay 1e-5、label_smoothing 0、aux_loss_weight 0、
modality_dropout 0、kd_enabled False。全部从头随机初始化，无任何 pretrained。

断点续跑：result.json + test_predictions.csv 完整即跳过；--force 覆盖。

用法:
  python scripts/run_twostage_training_protocol_control.py --gpus 0 1 2 3 4 5 6 7
"""
import os, sys, json, subprocess, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
SEEDS = [42, 123, 456]
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
MAX_PATCHES = 2500
RESULTS_ROOT = PROJECT / "results"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")
VERIFY_OUT = RESULTS_ROOT / "twostage_training_protocol_verify.json"

# Stage1 各自 best —— 绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}

# Stage2 固定 r4 no-EPEG
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}

# 三条件：目录名 → 是否为 HE-only
CONDITIONS = {
    "he_rrt_samplerfix_lr1e4": True,
    "twostage_r4_noepeg_samplerfix_diff_lr": False,
    "twostage_r4_noepeg_samplerfix_unified_lr1e4": False,
}

SUMMARY_OUT = RESULTS_ROOT / "twostage_training_protocol_control_summary.json"
README_OUT = RESULTS_ROOT / "twostage_training_protocol_control_README.md"


def build_config(condition: str, seed: int):
    he_only = CONDITIONS[condition]
    modalities = ["HE"] if he_only else MODALITIES
    dir_mapping = {"HE": "C16_HE_features"} if he_only else DIR_MAPPING

    training = {
        "batch_size": 1, "num_epochs": 80,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": {"type": "cosine"},
        "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
        "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
        "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
        "no_validation": False,
    }
    if condition == "twostage_r4_noepeg_samplerfix_diff_lr":
        training["lr_stage1"] = 1e-5
        training["lr_stage2"] = 2e-5
    # 其它条件：统一 LR，不写 lr_stage1 / lr_stage2

    model = {
        "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
        "region_num": 4, "n_layers": 2, "n_heads": 4,
        "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
        "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
        "crmsa_heads": 8, "crmsa_mlp": False,
        "fusion_type": "two_stage_region", "fusion_stage": "middle",
        "use_gated_fusion": False, "abmil_hidden_dim": 256,
        "use_mclc": False, "aggregate_modalities": True,
    }
    if not he_only:
        model["stage2_type"] = "staining_msa"
        model["encoder_cfg"] = STAGE1_ENCODER_CFG
        model["stage2_cfg"] = STAGE2_CFG

    return {
        "data": {
            "dataset_type": "c16", "modalities": modalities,
            "dir_mapping": dir_mapping,
            "train_label_file": TRAIN_LABEL_FILE,
            "val_label_file": VAL_LABEL_FILE,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": MAX_PATCHES, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": model,
        "training": training,
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": "", "log_dir": "", "img_dir": "",
        },
    }


def is_complete(seed_dir: Path) -> bool:
    r = seed_dir / "result.json"
    p = seed_dir / "test_predictions.csv"
    if not (r.exists() and p.exists()):
        return False
    try:
        return "auc" in json.loads(r.read_text())
    except Exception:
        return False


def run_wave(pairs):
    procs = []
    for condition, seed, gpu, cfg_path in pairs:
        out_root = RESULTS_ROOT / condition
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((condition, seed, p, log_f))
        print(f"[launch] {condition} seed={seed} gpu={gpu}", flush=True)
    for condition, seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = RESULTS_ROOT / condition / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {condition} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] {condition} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def _fmt(v):
    return f"{v:.4f}"


def generate_summary():
    cond_aucs = {}
    cond_meta = {}
    for condition in CONDITIONS:
        aucs, seeds_done, best_epochs, lrs = [], [], [], []
        out_root = RESULTS_ROOT / condition
        for s in SEEDS:
            rpath = out_root / f"seed{s}" / "result.json"
            if not rpath.exists():
                continue
            r = json.loads(rpath.read_text())
            seeds_done.append(s)
            aucs.append(r["auc"])
            best_epochs.append(r["best_epoch"])
            lrs.append(r["actual_optimizer_lrs"])
        cond_aucs[condition] = {s: a for s, a in zip(seeds_done, aucs)}
        cond_meta[condition] = {
            "per_seed": {s: a for s, a in zip(seeds_done, aucs)},
            "best_epoch": {s: e for s, e in zip(seeds_done, best_epochs)},
            "actual_optimizer_lrs": lrs[0] if lrs else None,
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "std_auc": float(np.std(aucs)) if aucs else None,
        }

    # 读取 verify 输出（梯度范数 / 相对更新 / sampler MD5 / patch 一致性 / 无预训练）
    verify = None
    if VERIFY_OUT.exists():
        verify = json.loads(VERIFY_OUT.read_text())

    he_name = "he_rrt_samplerfix_lr1e4"
    diff_name = "twostage_r4_noepeg_samplerfix_diff_lr"
    uni_name = "twostage_r4_noepeg_samplerfix_unified_lr1e4"

    def paired_delta(a_name, b_name):
        a = cond_aucs[a_name]
        b = cond_aucs[b_name]
        common = sorted(set(a) & set(b))
        deltas = [a[s] - b[s] for s in common]
        return {
            "common_seeds": common,
            "per_seed_delta": {str(s): round(a[s] - b[s], 6) for s in common},
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        }

    summary = {
        "task": "twostage_training_protocol_control",
        "conditions": {
            "he_rrt_samplerfix_lr1e4": {
                "desc": "HE-only baseline (sampler fix, unified lr 1e-4)",
                **cond_meta[he_name],
            },
            "twostage_r4_noepeg_samplerfix_diff_lr": {
                "desc": "two-stage r4 no-EPEG (sampler fix, diff lr 1e-5/2e-5)",
                **cond_meta[diff_name],
            },
            "twostage_r4_noepeg_samplerfix_unified_lr1e4": {
                "desc": "two-stage r4 no-EPEG (sampler fix, unified lr 1e-4)",
                **cond_meta[uni_name],
            },
        },
        "paired_delta_auc": {
            "diff_lr_minus_he": paired_delta(diff_name, he_name),
            "unified_lr_minus_he": paired_delta(uni_name, he_name),
            "unified_lr_minus_diff_lr": paired_delta(uni_name, diff_name),
        },
        "protocol": {
            "train": 270, "val_test": 129, "monitor": "val_auc", "mode": "max",
            "patience": 10, "num_epochs": 80, "scheduler": "cosine",
            "sampling": "random", "train_per_epoch": True, "val_test_per_epoch": False,
            "batch_size": 1, "num_workers": 2, "max_patches": 2500,
            "weight_decay": 1e-5, "dropout": 0.25, "abmil_hidden_dim": 256,
            "label_smoothing": 0.0, "aux_loss_weight": 0.0,
            "modality_dropout": 0.0, "kd_enabled": False,
        },
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "stage2_cfg": STAGE2_CFG,
        "pretrained_loaded": False,
        "initialization": "random_from_scratch",
        "verify": verify,
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    # ---- README.md ----
    lines = []
    lines.append("# Two-stage 训练协议受控实验\n")
    lines.append("修复 per-epoch sampler（persistent_workers=False）+ LR 变量隔离。"
                 "3 条件 × 3 seeds（42/123/456），270/129 test-as-val，"
                 "全部从头随机初始化，无 pretrained。\n")

    lines.append("## AUC 表格\n")
    lines.append("| seed | HE samplerfix | Two-stage diff LR | Two-stage unified LR |")
    lines.append("|---|---|---|---|")
    for s in SEEDS:
        a = cond_aucs[he_name].get(s)
        b = cond_aucs[diff_name].get(s)
        c = cond_aucs[uni_name].get(s)
        lines.append(f"| {s} | {_fmt(a) if a is not None else '—'} | "
                     f"{_fmt(b) if b is not None else '—'} | "
                     f"{_fmt(c) if c is not None else '—'} |")
    mh = cond_meta[he_name]["mean_auc"]
    md = cond_meta[diff_name]["mean_auc"]
    mu = cond_meta[uni_name]["mean_auc"]
    sh = cond_meta[he_name]["std_auc"]
    sd = cond_meta[diff_name]["std_auc"]
    su = cond_meta[uni_name]["std_auc"]
    lines.append(f"| **mean** | {_fmt(mh) if mh is not None else '—'} | "
                 f"{_fmt(md) if md is not None else '—'} | "
                 f"{_fmt(mu) if mu is not None else '—'} |")
    lines.append(f"| **std** | {_fmt(sh) if sh is not None else '—'} | "
                 f"{_fmt(sd) if sd is not None else '—'} | "
                 f"{_fmt(su) if su is not None else '—'} |")
    lines.append("")

    lines.append("## paired ΔAUC (mean ± std)\n")
    d1 = paired_delta(diff_name, he_name)
    d2 = paired_delta(uni_name, he_name)
    d3 = paired_delta(uni_name, diff_name)
    lines.append(f"- Two-stage diff LR − HE samplerfix = "
                 f"{_fmt(d1['mean_delta']) if d1['mean_delta'] is not None else '—'} "
                 f"± {_fmt(d1['std_delta']) if d1['std_delta'] is not None else '—'}")
    lines.append(f"- Two-stage unified LR − HE samplerfix = "
                 f"{_fmt(d2['mean_delta']) if d2['mean_delta'] is not None else '—'} "
                 f"± {_fmt(d2['std_delta']) if d2['std_delta'] is not None else '—'}")
    lines.append(f"- Two-stage unified LR − Two-stage diff LR = "
                 f"{_fmt(d3['mean_delta']) if d3['mean_delta'] is not None else '—'} "
                 f"± {_fmt(d3['std_delta']) if d3['std_delta'] is not None else '—'}")
    lines.append("")

    lines.append("## best_epoch\n")
    lines.append("```")
    lines.append(json.dumps({c: cond_meta[c]["best_epoch"] for c in CONDITIONS}, indent=2))
    lines.append("```\n")

    lines.append("## 实际 optimizer LR（每活跃模块）\n")
    lines.append("```")
    lines.append(json.dumps({c: cond_meta[c]["actual_optimizer_lrs"] for c in CONDITIONS}, indent=2))
    lines.append("```\n")

    lines.append("## 协议验证（verify 脚本输出）\n")
    lines.append(f"- 预训练权重加载：**{summary['pretrained_loaded']}** "
                 f"(initialization={summary['initialization']})")
    if verify is not None and verify.get("checks"):
        g = verify["checks"].get("check3_grad_norms", {}).get("grad_norm", {})
        u = verify["checks"].get("check4_relative_updates", {})
        s5 = verify["checks"].get("check5_per_epoch_sampler", {})
        s7 = verify["checks"].get("check7_patch_consistency", {})
        lines.append("- gradient norms（forward/backward）:")
        lines.append("```")
        lines.append(json.dumps(g, indent=2))
        lines.append("```")
        lines.append("- relative parameter updates（optimizer.step 后）:")
        lines.append("```")
        lines.append(json.dumps(u, indent=2))
        lines.append("```")
        lines.append(f"- per-epoch sampler MD5: "
                     f"epoch0={s5.get('epoch0_indices_md5', '—')[:12]}…, "
                     f"epoch1={s5.get('epoch1_indices_md5', '—')[:12]}…, "
                     f"repeat={s5.get('epoch0_repeat_indices_md5', '—')[:12]}…, "
                     f"train_persistent_workers={s5.get('train_persistent_workers', '—')}")
        lines.append(f"- patch count/dim 一致性: "
                     f"patch_count_mismatch={s7.get('patch_count_mismatch', '—')}, "
                     f"feature_dim_mismatch={s7.get('feature_dim_mismatch', '—')}")
    lines.append("")

    lines.append("## 附加工件\n")
    lines.append("- `results/twostage_training_protocol_verify.json` — 7 项协议检查输出")
    lines.append("- 每个 seed 目录含 `result.json` / `test_predictions.csv` / "
                 "`protocol_check.json` / `config.json` / `ckpt/best_model.pt` / `logs/run.log`\n")
    README_OUT.write_text("\n".join(lines) + "\n")

    print(f"[summary] written {SUMMARY_OUT.name} + {README_OUT.name}", flush=True)
    print(f"[summary] HE={_fmt(mh) if mh is not None else '—'} "
          f"diffLR={_fmt(md) if md is not None else '—'} "
          f"uniLR={_fmt(mu) if mu is not None else '—'}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在且完整的 seed（默认跳过）")
    ap.add_argument("--summary-only", action="store_true",
                    help="只重新生成 summary + README，不训练")
    args = ap.parse_args()

    if args.summary_only:
        generate_summary()
        return

    # 收集待跑 (condition, seed)
    todo = []
    for condition in CONDITIONS:
        out_root = RESULTS_ROOT / condition
        out_root.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            seed_dir = out_root / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] {condition} seed={seed} (complete)", flush=True)
                continue
            todo.append((condition, seed, seed_dir))

    print(f"to-run: {len(todo)} seed(s) | gpus={args.gpus} | force={args.force}",
          flush=True)

    if todo:
        # 先写所有 config
        cfg_by_key = {}
        for condition, seed, seed_dir in todo:
            for d in ["ckpt", "logs", "img"]:
                (seed_dir / d).mkdir(parents=True, exist_ok=True)
            cfg = build_config(condition, seed)
            cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
            cfg["output"]["log_dir"] = str(seed_dir / "logs")
            cfg["output"]["img_dir"] = str(seed_dir / "img")
            cfg_path = seed_dir / "config.json"
            cfg_path.write_text(json.dumps(cfg, indent=2))
            cfg_by_key[(condition, seed)] = cfg_path

        # 按 wave 分配 GPU（全部 (condition, seed) 拍平后顺序分配）
        flat = [(c, s, cfg_by_key[(c, s)]) for (c, s, _d) in todo]
        for i in range(0, len(flat), len(args.gpus)):
            chunk = flat[i:i + len(args.gpus)]
            pairs = [(c, s, g, cfg_path)
                     for (c, s, cfg_path), g in zip(chunk, args.gpus[:len(chunk)])]
            run_wave(pairs)

    # 幂等重生成 summary + README（只要有 ≥1 个 result.json）
    if any((RESULTS_ROOT / c / f"seed{s}" / "result.json").exists()
           for c in CONDITIONS for s in SEEDS):
        generate_summary()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
