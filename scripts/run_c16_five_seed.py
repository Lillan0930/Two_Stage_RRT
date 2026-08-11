#!/usr/bin/env python3
"""
C16 Five-Seed Benchmark: HE-only vs Two-stage HE+PR
====================================================
Each seed gets one GPU, runs both HE-only and Two-stage sequentially.
After training: evaluate best checkpoint on independent test set.

Usage (from TwoStageRRT/):
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_five_seed.py

Seeds: 42, 123, 456, 789, 1024
GPUs:  2, 3, 4, 5, 6 (one per seed)
"""

import os, sys, time, subprocess, json
from pathlib import Path
import numpy as np

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
BASE_OUT = Path(__file__).resolve().parent.parent / "results" / "c16_five_seed"
SEEDS = [42, 123, 456, 789, 1024]
GPUS = [2, 3, 4, 5, 6]  # one GPU per seed

TEST_LABEL = str(PROJECT / "data/C16_labels/c16_test_labels.csv")
FEATURE_BASE = "/home/Public/lillan/features_result/C16_features"


def build_config(seed: int, modalities: list, dir_mapping: dict, out_dir: Path):
    return {
        "data": {
            "dataset_type": "c16",
            "train_label_file": str(PROJECT / "data/C16_labels/c16_train_labels.csv"),
            "feature_base_dir": FEATURE_BASE,
            "modalities": modalities,
            "dir_mapping": dir_mapping,
            "input_dim": 768, "num_classes": 2, "max_patches": 5000, "preload": False,
            "val_ratio": 0.2,
        },
        "model": {
            "mil_type": "abmil", "mlp_dim": 512, "dropout": 0.25, "use_gated": False,
            "region_num": 4, "n_layers": 2, "n_heads": 4,
            "drop_path": 0.0, "trans_dropout": 0.1, "epeg": True, "epeg_k": 9,
            "crmsa_k": 3, "cr_msa": True, "all_shortcut": True,
            "fusion_type": "two_stage_region", "fusion_stage": "middle",
            "use_gated_fusion": False, "abmil_hidden_dim": 256,
            "use_mclc": False, "aggregate_modalities": True,
            "crmsa_heads": 8, "crmsa_mlp": False,
        },
        "training": {
            "batch_size": 1, "num_epochs": 25, "learning_rate": 1e-4,
            "weight_decay": 1e-5, "scheduler": {"type": "plateau"},
            "early_stopping": {"patience": 10, "monitor": "val_auc", "mode": "max"},
            "use_amp": False, "focal_loss": False, "label_smoothing": 0.0,
            "kd_enabled": False, "modality_dropout": 0.0, "aux_loss_weight": 0.0,
        },
        "data_split": {"val_start": 100},
        "environment": {"device": "cuda:0", "num_workers": 2, "seed": seed},
        "output": {
            "save_dir": str(out_dir / "ckpt"),
            "log_dir": str(out_dir / "logs"),
            "img_dir": str(out_dir / "img"),
        },
    }


def write_run_script(exp_name: str, seed: int, gpu: int, out_dir: Path):
    """Write a self-contained training + test-eval script."""
    script = f'''
import os, sys, json, logging, time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"
sys.path.insert(0, "{PROJECT}")
os.chdir("{PROJECT}")

from train import Trainer, build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

OUT_DIR = Path("{out_dir}")
cfg_path = OUT_DIR / "config.json"
with open(cfg_path) as f:
    cfg = json.load(f)

# ── Logging ──
logger = logging.getLogger("exp")
logger.handlers.clear()
logger.setLevel(logging.INFO)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(OUT_DIR / "logs" / "run.log"))
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

# ── Train ──
exp_name = "{exp_name}"
seed = {seed}
print(f"[{{exp_name}} seed={{seed}} GPU={gpu}] Starting...", flush=True)
t0 = time.time()
trainer = Trainer(cfg, logger, f"seed{{seed}}")
_, val_auc = trainer.train()
train_time = time.time() - t0
print(f"[{{exp_name}} seed={{seed}}] Internal Val AUC={{val_auc:.4f}} in {{train_time:.0f}}s", flush=True)

# ── Test Eval ──
ckpt = torch.load(str(OUT_DIR / "ckpt" / "best_model.pt"), map_location="cuda:0", weights_only=False)
mc = ckpt["config"].get("model", ckpt["config"])
num_mod = len(cfg["data"]["modalities"])

model = MM_RRT_ABMIL(
    num_modalities=num_mod, input_dim=mc.get("input_dim", 768), num_classes=2,
    mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
    n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
    drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
    epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
    crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
    all_shortcut=mc.get("all_shortcut", True),
    crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
    fusion_type=mc.get("fusion_type", "two_stage_region"),
    fusion_stage=mc.get("fusion_stage", "middle"),
    stage2_type=mc.get("stage2_type", "staining_msa"),
    abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
)
state_dict = ckpt["model_state_dict"]
filtered = {{k: v for k, v in state_dict.items()
             if k in model.state_dict() and model.state_dict()[k].shape == v.shape}}
model.load_state_dict(filtered, strict=False)
model = model.cuda().eval()

feature_dirs = build_feature_dirs("{FEATURE_BASE}", cfg["data"]["modalities"],
                                   cfg["data"]["dir_mapping"])
test_ds = C16MultimodalDataset(feature_dirs=feature_dirs, label_file="{TEST_LABEL}",
                               max_patches=5000, preload=False, verbose=False)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                     collate_fn=c16_multimodal_collate_fn, num_workers=2,
                     pin_memory=True, persistent_workers=True)

all_logits, all_labels = [], []
with torch.no_grad():
    for batch in test_dl:
        feats = [torch.stack(m).cuda() for m in batch["features"]]
        labels = batch["labels"].cuda()
        logits, _, _, _ = model(feats)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

logits = torch.cat(all_logits)
labels_np = torch.cat(all_labels).numpy()
probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
preds = torch.argmax(logits, dim=-1).numpy()

test_auc = roc_auc_score(labels_np, probs)
test_acc = accuracy_score(labels_np, preds)
test_f1 = f1_score(labels_np, preds)

result = {{
    "exp": exp_name, "seed": seed,
    "val_auc": float(val_auc),
    "test_auc": float(test_auc), "test_acc": float(test_acc), "test_f1": float(test_f1),
    "train_time_s": float(train_time),
}}
with open(OUT_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"[{{exp_name}} seed={{seed}}] Test AUC={{test_auc:.4f}} Acc={{test_acc:.4f}} F1={{test_f1:.4f}}", flush=True)
'''
    script_path = out_dir / "run_script.py"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path


def launch(exp_name: str, seed: int, modalities: list, dir_mapping: dict, gpu: int):
    out_dir = BASE_OUT / exp_name / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = build_config(seed, modalities, dir_mapping, out_dir)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    script = write_run_script(exp_name, seed, gpu, out_dir)

    log_f = open(out_dir / "logs" / "stdout.log", "w")
    proc = subprocess.Popen(
        [PYTHON, str(script)],
        stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(PROJECT),
    )
    return proc, log_f, out_dir


def collect_results():
    """Read result.json from all completed experiments."""
    results = []
    for exp_name in ["he_only", "two_stage"]:
        for seed in SEEDS:
            result_file = BASE_OUT / exp_name / f"seed{seed}" / "result.json"
            if result_file.exists():
                with open(result_file) as f:
                    results.append(json.load(f))
    return results


def main():
    print("=" * 70)
    print("C16 Five-Seed Benchmark: HE-only vs Two-stage HE+PR")
    print(f"Seeds: {SEEDS}")
    print(f"GPUs:  {dict(zip(SEEDS, GPUS))}")
    print(f"Split: StratifiedShuffleSplit(random_state=42)")
    print("=" * 70)

    all_procs = []

    for seed, gpu in zip(SEEDS, GPUS):
        # HE-only
        proc, log_f, out_dir = launch(
            "he_only", seed, ["HE"], {"HE": "C16_HE_features"}, gpu)
        all_procs.append((f"he_only_s{seed}", proc, log_f, out_dir))
        print(f"  he_only     seed={seed:>4}  GPU={gpu}  → {out_dir}")

        # Two-stage
        proc, log_f, out_dir = launch(
            "two_stage", seed, ["HE", "PR"],
            {"HE": "C16_HE_features", "PR": "C16_PR_features"}, gpu)
        all_procs.append((f"two_stage_s{seed}", proc, log_f, out_dir))
        print(f"  two_stage   seed={seed:>4}  GPU={gpu}  → {out_dir}")

    print(f"\n▶ {len(all_procs)} jobs running across 5 GPUs. Waiting...\n")

    # ── Wait for all ──
    for name, proc, log_f, out_dir in all_procs:
        ret = proc.wait()
        log_f.close()
        status = "✓" if ret == 0 else f"✗(rc={ret})"
        result_file = out_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            info = f"Val={r['val_auc']:.4f} Test={r['test_auc']:.4f}"
        else:
            info = "no result"
        print(f"  [{status}] {name:<20s}  {info}")

    # ── Summary ──
    results = collect_results()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for exp_name in ["he_only", "two_stage"]:
        exp_results = [r for r in results if r["exp"] == exp_name]
        if not exp_results:
            print(f"\n{exp_name}: no results")
            continue
        val_aucs = [r["val_auc"] for r in exp_results]
        test_aucs = [r["test_auc"] for r in exp_results]
        test_accs = [r["test_acc"] for r in exp_results]
        test_f1s = [r["test_f1"] for r in exp_results]

        print(f"\n{exp_name}:")
        for r in sorted(exp_results, key=lambda x: x["seed"]):
            print(f"  seed={r['seed']:>4}: Val={r['val_auc']:.4f}  "
                  f"Test AUC={r['test_auc']:.4f}  Acc={r['test_acc']:.4f}  F1={r['test_f1']:.4f}")

        print(f"  ─────────────────────────────────────────")
        print(f"  Val AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        print(f"  Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"  Test Acc: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"  Test F1:  {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")

    # ── Head-to-head ──
    he_test = [r["test_auc"] for r in results if r["exp"] == "he_only"]
    ts_test = [r["test_auc"] for r in results if r["exp"] == "two_stage"]
    if len(he_test) == len(ts_test) == 5:
        diffs = [t - h for h, t in zip(sorted(he_test), sorted(ts_test))]
        print(f"\n  Δ (Two-stage − HE-only) per seed: "
              + "  ".join(f"{d:+.4f}" for d in diffs))
        print(f"  Mean Δ: {np.mean(diffs):+.4f} ± {np.std(diffs):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
