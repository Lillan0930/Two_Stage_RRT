#!/usr/bin/env python3
"""Two-RRT + Plain Concat 正式公平对照（Task 2）。

与 twostage_r4_noepeg_samplerfix_unified_lr1e4 完全同协议，唯一差异：
    model.stage2_type = "concat"   （CrossStainingCRMSA → identity concat）

即：
    z_final = torch.cat([z_he, z_pr], dim=1)   # 无跨染色融合
    → ABMIL → logits

其余（split / sampler / max_patches / optimizer / lr / wd / scheduler / loss /
dropout / encoder_cfg / stage2_cfg / seed）一律不变。跑 3 seeds：42 / 123 / 456。

用法:
  python scripts/run_concat_control.py --gpus 0 1 2
"""
import os, sys, json, subprocess, argparse
from pathlib import Path

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
PY = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
SEED_RUNNER = str(PROJECT / "scripts" / "_run_protocol_seed.py")
SEEDS = [42, 123, 456]
CONDITION = "twostage_r4_noepeg_samplerfix_concat"
RESULTS_ROOT = PROJECT / "results"

FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"
MODALITIES = ["HE", "PR"]
DIR_MAPPING = {"HE": "C16_HE_features", "PR": "C16_PR_features"}
TRAIN_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_train_labels.csv")
VAL_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")

STAGE1_ENCODER_CFG = {
    "HE": {"region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4, "drop_path": 0.0},
    "PR": {"region_num": 8, "epeg_k": 15, "crmsa_k": 5, "n_heads": 8,
           "drop_path": 0.11554210024949738},
}
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": True,
}


def build_config(seed: int):
    return {
        "data": {
            "dataset_type": "c16", "modalities": MODALITIES,
            "dir_mapping": DIR_MAPPING,
            "train_label_file": TRAIN_LABEL_FILE,
            "val_label_file": VAL_LABEL_FILE,
            "feature_base_dir": FEATURE_BASE,
            "input_dim": 768, "num_classes": 2,
            "max_patches": 2500, "preload": False,
            "sampling": "random", "sample_seed": seed, "no_validation": False,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "stage2_type": "concat",              # ← 唯一差异（原 unified-LR 为 staining_msa）
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            "encoder_cfg": STAGE1_ENCODER_CFG,
            "stage2_cfg": STAGE2_CFG,
        },
        "training": {
            "batch_size": 1, "num_epochs": 80,
            "learning_rate": 1e-4,
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
    if not r.exists():
        return False
    try:
        return "auc" in json.loads(r.read_text())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = RESULTS_ROOT / CONDITION
    out_root.mkdir(parents=True, exist_ok=True)

    todo = []
    for seed in SEEDS:
        seed_dir = out_root / f"seed{seed}"
        if is_complete(seed_dir) and not args.force:
            print(f"[skip] {CONDITION} seed={seed} (complete)", flush=True)
            continue
        todo.append((seed, seed_dir))

    print(f"to-run: {len(todo)} seed(s) | gpus={args.gpus} | force={args.force}", flush=True)

    for seed, seed_dir in todo:
        for d in ["ckpt", "logs", "img"]:
            (seed_dir / d).mkdir(parents=True, exist_ok=True)
        cfg = build_config(seed)
        cfg["output"]["save_dir"] = str(seed_dir / "ckpt")
        cfg["output"]["log_dir"] = str(seed_dir / "logs")
        cfg["output"]["img_dir"] = str(seed_dir / "img")
        (seed_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    procs = []
    for seed, seed_dir in todo:
        gpu = args.gpus[0]
        log_f = open(seed_dir / f"stdout_seed{seed}.log", "w")
        p = subprocess.Popen([PY, SEED_RUNNER, "--config", str(seed_dir / "config.json"),
                              "--gpu", str(gpu)],
                             stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((seed, p, log_f))
        print(f"[launch] {CONDITION} seed={seed} gpu={gpu}", flush=True)

    for seed, p, log_f in procs:
        p.wait()
        log_f.close()
        rpath = out_root / f"seed{seed}" / "result.json"
        if p.returncode != 0 or not rpath.exists():
            print(f"[FAIL] {CONDITION} seed={seed} rc={p.returncode}", flush=True)
            continue
        r = json.loads(rpath.read_text())
        print(f"[done] {CONDITION} seed={seed} auc={r['auc']:.4f} "
              f"acc={r['accuracy']:.4f} f1={r['f1']:.4f} epoch={r['best_epoch']}",
              flush=True)

    print("CONCAT CONTROL DONE", flush=True)


if __name__ == "__main__":
    main()
