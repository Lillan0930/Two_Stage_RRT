#!/usr/bin/env python3
"""对称双模态 official-style CR-MSA 受控实验 driver（两版本 × 5 seeds）。

两个版本在**完全相同协议**下运行，只改 Stage2 region_num：
  A. symmetric_r4_noepeg : Stage2 region_num=4, epeg=False
  B. symmetric_r8_noepeg : Stage2 region_num=8, epeg=False

固定项：
  - Stage1 HE/PR encoder_cfg = 各自 best（绝对不改）
  - Stage2: crmsa_heads=8 / crmsa_k=3 / drop_out=0.1 / drop_path=0.0 /
            crmsa_mlp=False / ffn=False / qkv_bias=True / epeg=False
  - 协议 = HE_rrt_abmil 270/129 test-as-val：seeds=[42,123,456,789,1024]，
    max_patches=2500，sampling=random（corrected sampler），num_epochs=80，
    cosine，patience=10，lr_stage1=1e-5，lr_stage2=2e-5，weight_decay=1e-5，
    dropout=0.25，abmil_hidden_dim=256

断点续跑：已存在且完整（result.json + test_predictions.csv）的 seed 自动跳过；
缺失/不完整的 seed 继续跑；默认不覆盖，只有 --force 才覆盖。

用法:
  python scripts/run_symmetric_official.py --gpus 2 3 6 7
  python scripts/run_symmetric_official.py --gpus 2 3 --force
"""
import os, sys, json, math, subprocess, argparse
from pathlib import Path

import numpy as np

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
SEEDS = [42, 123, 456, 789, 1024]
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
MAX_PATCHES = 2500
RESULTS_ROOT = PROJECT / "results"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_symmetric_seed.py")

# 每个版本独立的 result 目录（不覆盖旧结果）
VERSIONS = {
    "symmetric_r4_noepeg": 4,
    "symmetric_r8_noepeg": 8,
}

# Stage1 各自 best —— 绝对不改
STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4,
           "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}


def build_stage2_cfg(region_num: int):
    return {
        "region_num": region_num, "crmsa_heads": 8, "crmsa_k": 3,
        "drop_out": 0.1, "drop_path": 0.0, "epeg": False, "epeg_k": 15,
        "crmsa_mlp": False, "ffn": False, "qkv_bias": True,
    }


def build_config(seed: int, region_num: int):
    return {
        "data": {
            "dataset_type": "c16", "modalities": MODALITIES,
            "dir_mapping": DIR_MAPPING,
            "train_label_file": TRAIN_LABEL_FILE,
            "val_label_file": VAL_LABEL_FILE,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": MAX_PATCHES, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "stage2_type": "staining_msa",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            "encoder_cfg": STAGE1_ENCODER_CFG,
            "stage2_cfg": build_stage2_cfg(region_num),
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4,
            "lr_stage1": 1e-5, "lr_stage2": 2e-5,
            "weight_decay": 1e-5,
            "scheduler": {"type": "cosine"},
            "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
            "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
            "no_validation": False,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {"save_dir": "", "log_dir": "", "img_dir": ""},
    }


def is_complete(seed_dir: Path) -> bool:
    r = seed_dir / "result.json"
    p = seed_dir / "test_predictions.csv"
    if not (r.exists() and p.exists()):
        return False
    try:
        j = json.loads(r.read_text())
        return "val_auc" in j
    except Exception:
        return False


def run_wave(pairs, out_root: Path):
    procs = []
    for name, seed, gpu, cfg_path in pairs:
        log_f = open(out_root / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(cfg_path),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((seed, p, log_f))
        print(f"[launch] {name} seed={seed} gpu={gpu}", flush=True)
    for seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = out_root / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] seed={seed} auc={r['val_auc']:.4f} "
              f"acc={r['val_acc']:.4f} f1={r['val_f1']:.4f} "
              f"epoch={r['best_epoch']}", flush=True)


def pad_stats(N: int, region_num: int):
    """复刻 CrossStainingCRMSA._pad 的 region_num 分支（region_size=0 路径）。"""
    H = W = int(math.ceil(math.sqrt(N)))
    _n = -H % region_num
    H = W = H + _n
    add_length = H * W - N
    padding_ratio = add_length / (H * W) if H * W > 0 else 0.0
    return H, W, add_length, padding_ratio


def compute_padding_stats(region_num: int, out_root: Path):
    """train + test 每 slide 的 N/H/W/add_length/padding_ratio + 汇总（与 seed 无关）。"""
    from train import build_feature_dirs
    from data.c16_multimodal_dataset import C16MultimodalDataset

    feature_dirs = build_feature_dirs(FEATURE_BASE, MODALITIES, DIR_MAPPING)
    out = {}
    for split, label_file in [("train", TRAIN_LABEL_FILE), ("test", VAL_LABEL_FILE)]:
        ds = C16MultimodalDataset(
            feature_dirs=feature_dirs, label_file=label_file,
            max_patches=MAX_PATCHES, preload=False, verbose=False,
            sampling="random", sample_seed=0, per_epoch=False,
        )
        per_slide, ratios = [], []
        for i in range(len(ds)):
            item = ds[i]
            N = int(item["features"][MODALITIES[0]].shape[0])
            H, W, add_length, pr = pad_stats(N, region_num)
            per_slide.append({"slide": item["slide_id"], "N": N, "H": H, "W": W,
                              "add_length": add_length,
                              "padding_ratio": round(pr, 6)})
            ratios.append(pr)
        out[split] = {
            "n_slides": len(ratios),
            "mean_padding_ratio": round(float(np.mean(ratios)), 6),
            "median_padding_ratio": round(float(np.median(ratios)), 6),
            "max_padding_ratio": round(float(np.max(ratios)), 6),
            "per_slide": per_slide,
        }
    out["region_num"] = region_num
    out["note"] = "Stage2 CrossStainingCRMSA._pad (region_size=0 path); padding ratio = add_length/(H*W)"
    (out_root / "padding_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"[padding] region_num={region_num}: "
          f"train mean={out['train']['mean_padding_ratio']:.4f} "
          f"max={out['train']['max_padding_ratio']:.4f} | "
          f"test mean={out['test']['mean_padding_ratio']:.4f} "
          f"max={out['test']['max_padding_ratio']:.4f}", flush=True)
    return out


def load_he_baseline():
    p = RESULTS_ROOT / "HE_rrt_abmil" / "summary.json"
    j = json.loads(p.read_text())
    per_seed = {r["seed"]: r["val_auc"] for r in j["per_seed"]}
    return per_seed, j["mean_auc"], j["std_auc"]


def load_old_symmetric():
    """旧 symmetric r4 epeg=True（optuna best trial 0，仅 3 seeds 42/123/456）。"""
    p = RESULTS_ROOT / "HE_PR_twostage_rrt" / "summary.json"
    j = json.loads(p.read_text())
    seeds = j["protocol"]["seeds"]
    aucs = j["best"]["per_seed_auc"]
    per_seed = dict(zip(seeds, aucs))
    return per_seed, j["best"]["mean_auc"], j["best"]["std_auc"]


def _fmt(v):
    return f"{v:.4f}"


def generate_summary(name: str, region_num: int, out_root: Path):
    results, aucs, accs, f1s = [], [], [], []
    for s in SEEDS:
        rpath = out_root / f"seed{s}" / "result.json"
        if not rpath.exists():
            continue
        r = json.loads(rpath.read_text())
        results.append(r)
        aucs.append(r["val_auc"])
        accs.append(r["val_acc"])
        f1s.append(r["val_f1"])

    if not results:
        print(f"[summary] {name}: no seed results yet, skip", flush=True)
        return

    he_per_seed, he_mean, he_std = load_he_baseline()
    old_per_seed, old_mean, old_std = load_old_symmetric()

    # paired ΔAUC vs HE-only（同 seed）
    deltas = []
    delta_by_seed = {}
    for r in results:
        s = r["seed"]
        if s in he_per_seed:
            d = r["val_auc"] - he_per_seed[s]
            deltas.append(d)
            delta_by_seed[str(s)] = round(d, 6)

    # 与旧 symmetric r4 epeg=True 对比（只在公共 seed 42/123/456 上）
    common = [s for s in SEEDS if s in old_per_seed and s in {r["seed"] for r in results}]
    new_common_aucs = [next(r["val_auc"] for r in results if r["seed"] == s) for s in common]
    new_common_mean = float(np.mean(new_common_aucs)) if new_common_aucs else None

    summary = {
        "task": name,
        "stage2": build_stage2_cfg(region_num),
        "stage1_encoder_cfg": STAGE1_ENCODER_CFG,
        "protocol": {"train": 270, "val_test": 129, "monitor": "val_auc",
                     "mode": "max", "patience": 10, "num_epochs": 80,
                     "scheduler": "cosine", "sampling": "random",
                     "lr_stage1": 1e-5, "lr_stage2": 2e-5, "weight_decay": 1e-5,
                     "dropout": 0.25, "abmil_hidden_dim": 256, "max_patches": 2500},
        "seeds": [r["seed"] for r in results],
        "per_seed": results,
        "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs)),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
        "paired_delta_auc_vs_he_only": {
            "he_mean_auc": he_mean, "he_std_auc": he_std,
            "per_seed_delta": delta_by_seed,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "std_delta": float(np.std(deltas)) if deltas else None,
        },
        "vs_old_symmetric_r4_epeg_true": {
            "old_mean_auc": old_mean, "old_std_auc": old_std,
            "old_seeds": list(old_per_seed.keys()),
            "common_seeds": common,
            "new_mean_auc_on_common": new_common_mean,
            "delta_mean_auc": round(new_common_mean - old_mean, 6) if new_common_mean is not None else None,
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # ---- README.md ----
    lines = []
    lines.append(f"# {name}\n")
    lines.append("对称双模态 official-style Cross-Staining CR-MSA（Stage2），"
                 "270 train / 129 test-as-val，5 seeds。\n")
    lines.append("## Stage2 配置")
    lines.append("```")
    lines.append(json.dumps(build_stage2_cfg(region_num), indent=2))
    lines.append("```\n")
    lines.append("## Stage1 encoder（固定，各自 best）")
    lines.append("```")
    lines.append(json.dumps(STAGE1_ENCODER_CFG, indent=2))
    lines.append("```\n")

    lines.append("## Per-seed 指标\n")
    lines.append("| seed | AUC | Acc | F1 | Sens | Spec | epoch |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['seed']} | {_fmt(r['val_auc'])} | {_fmt(r['val_acc'])} | "
                     f"{_fmt(r['val_f1'])} | {_fmt(r['val_sensitivity'])} | "
                     f"{_fmt(r['val_specificity'])} | {r['best_epoch']} |")
    lines.append("")
    lines.append(f"- **AUC**: {_fmt(np.mean(aucs))} ± {_fmt(np.std(aucs))}")
    lines.append(f"- **Acc**: {_fmt(np.mean(accs))} ± {_fmt(np.std(accs))}")
    lines.append(f"- **F1**: {_fmt(np.mean(f1s))} ± {_fmt(np.std(f1s))}\n")

    lines.append("## 与 HE-only 同 seed 的 paired ΔAUC\n")
    lines.append(f"HE-only 5-seed: AUC {_fmt(he_mean)} ± {_fmt(he_std)}\n")
    lines.append("| seed | this AUC | HE-only AUC | ΔAUC |")
    lines.append("|---|---|---|---|")
    for r in results:
        s = r["seed"]
        if s in he_per_seed:
            lines.append(f"| {s} | {_fmt(r['val_auc'])} | {_fmt(he_per_seed[s])} | "
                         f"{_fmt(r['val_auc'] - he_per_seed[s])} |")
    lines.append("")
    if deltas:
        lines.append(f"- **paired ΔAUC**: {_fmt(np.mean(deltas))} ± {_fmt(np.std(deltas))}\n")

    lines.append("## 与旧 symmetric r4 epeg=True 对比\n")
    lines.append(f"旧结果（Stage2 region_num=4, epeg=True，仅 3 seeds 42/123/456）: "
                 f"AUC {_fmt(old_mean)} ± {_fmt(old_std)}\n")
    if new_common_mean is not None:
        lines.append(f"本版本在公共 3 seeds 上: AUC {_fmt(new_common_mean)}，"
                     f"Δ = {_fmt(new_common_mean - old_mean)}\n")

    lines.append("## 附加工件\n")
    lines.append("- `padding_stats.json` — Stage2 padding 统计（train/test 每 slide N/H/W/add_length/padding_ratio + 汇总）")
    lines.append("- `seed42/stage2_magnitude_seed42.json` — seed42 幅度诊断 ||delta||/||z||（HE/PR）\n")
    (out_root / "README.md").write_text("\n".join(lines) + "\n")
    print(f"[summary] {name}: AUC {_fmt(np.mean(aucs))} ± {_fmt(np.std(aucs))} "
          f"| ΔAUC_vs_HE {_fmt(np.mean(deltas)) if deltas else float('nan')} "
          f"| written summary.json + README.md", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[2, 3, 6, 7])
    ap.add_argument("--force", action="store_true",
                    help="覆盖已存在且完整的 seed（默认跳过）")
    args = ap.parse_args()

    todo = []  # (name, seed, out_root)
    for name, region_num in VERSIONS.items():
        out_root = RESULTS_ROOT / name
        for seed in SEEDS:
            seed_dir = out_root / f"seed{seed}"
            if is_complete(seed_dir) and not args.force:
                print(f"[skip] {name} seed={seed} (complete)", flush=True)
                continue
            todo.append((name, region_num, seed, seed_dir))

    print(f"to-run: {len(todo)} seed(s) | gpus={args.gpus} | force={args.force}",
          flush=True)

    # 按版本分组，逐版本逐 wave 跑（每 wave = len(gpus) 个 seed）
    for name, region_num in VERSIONS.items():
        out_root = RESULTS_ROOT / name
        seeds_to_run = [s for (n, _r, s, _d) in todo if n == name]
        if seeds_to_run:
            out_root.mkdir(parents=True, exist_ok=True)
            # 先写好所有 config（避免 wave 内 worker 找不到 config）
            cfg_by_seed = {}
            for seed in seeds_to_run:
                seed_dir = out_root / f"seed{seed}"
                for d in ["ckpt", "logs", "img"]:
                    (seed_dir / d).mkdir(parents=True, exist_ok=True)
                cfg = build_config(seed, region_num)
                cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
                cfg["output"]["log_dir"] = str(seed_dir / "logs")
                cfg["output"]["img_dir"] = str(seed_dir / "img")
                cfg_path = seed_dir / "config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2))
                cfg_by_seed[seed] = cfg_path

            for i in range(0, len(seeds_to_run), len(args.gpus)):
                chunk = seeds_to_run[i:i + len(args.gpus)]
                pairs = [(name, s, g, cfg_by_seed[s])
                         for s, g in zip(chunk, args.gpus[:len(chunk)])]
                run_wave(pairs, out_root)

        # padding 统计 + summary 幂等重生成（即使本次无 seed 要跑，也补生成）
        if out_root.exists() and any(
                (out_root / f"seed{s}" / "result.json").exists() for s in SEEDS):
            compute_padding_stats(region_num, out_root)
            generate_summary(name, region_num, out_root)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
