#!/usr/bin/env python3
"""
C16 Two-stage R²T — Real PR vs Random PR, 5 seeds each.

Usage (from TwoStageRRT/):
    /home/cxl/miniconda3/envs/rrtmil/bin/python scripts/run_c16_batch.py

Seeds: 42, 123, 456, 789, 1024
GPUs:  2, 3, 4, 5, 6  (one per seed, Real+Random share same GPU per seed)
"""

import os, sys, time, subprocess, json
from pathlib import Path

PYTHON = "/home/cxl/miniconda3/envs/rrtmil/bin/python"
PROJECT = Path("/home/Public/lillan/TwoStageRRT")
BASE_OUT = Path("/home/Public/lillan/work_results/c16_two_stage_batch")
SEEDS = [42, 123, 456, 789, 1024]

# GPU assignment: spread across 6 free GPUs (2-7)
# Real PR gets first batch of 5 GPUs, Random PR overlaps where possible
GPU_MAP = {
    ('real_PR', 42): 2,    ('random_PR', 42): 7,
    ('real_PR', 123): 3,   ('random_PR', 123): 2,
    ('real_PR', 456): 4,   ('random_PR', 456): 3,
    ('real_PR', 789): 5,   ('random_PR', 789): 4,
    ('real_PR', 1024): 6,  ('random_PR', 1024): 5,
}


def build_run_script(exp_name, seed, pr_dir, gpu, out_dir):
    """Write the per-experiment run script."""
    script = f'''
import os, sys, json, logging, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu}'
sys.path.insert(0, '{PROJECT}')
from train import Trainer

with open('{out_dir}/config.json') as f:
    cfg = json.load(f)

logger = logging.getLogger('exp')
logger.handlers.clear()
logger.setLevel(logging.INFO)
log_dir = Path('{out_dir}/logs')
log_dir.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(log_dir / 'run.log'))
fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(fh)

print(f'[{exp_name} seed={seed} GPU={gpu}] Starting...', flush=True)
t0 = time.time()
trainer = Trainer(cfg, logger, 's{seed}')
_, auc = trainer.train()
elapsed = time.time() - t0
print(f'[{exp_name} seed={seed}] DONE AUC={{auc:.4f}} in {{elapsed:.0f}}s', flush=True)
'''
    script_path = out_dir / "run_script.py"
    with open(script_path, 'w') as f:
        f.write(script)
    return script_path


def build_config(seed, pr_dir):
    return {
        'data': {
            'dataset_type': 'c16',
            'train_label_file': str(PROJECT / 'data/C16_labels/c16_train_labels.csv'),
            'val_label_file': str(PROJECT / 'data/C16_labels/c16_test_labels.csv'),
            'feature_base_dir': '/home/Public/lillan/features_result/C16_features',
            'modalities': ['HE', 'PR'],
            'dir_mapping': {'HE': 'C16_HE_features', 'PR': pr_dir},
            'input_dim': 768, 'num_classes': 2, 'max_patches': 5000, 'preload': False,
            'val_ratio': 0.2,
        },
        'model': {
            'mil_type': 'abmil', 'mlp_dim': 512, 'dropout': 0.25, 'use_gated': False,
            'region_num': 4, 'n_layers': 2, 'n_heads': 4,
            'drop_path': 0.0, 'trans_dropout': 0.1, 'epeg': True, 'epeg_k': 9,
            'crmsa_k': 3, 'cr_msa': True, 'all_shortcut': True,
            'fusion_type': 'two_stage_region', 'fusion_stage': 'middle',
            'use_gated_fusion': False, 'abmil_hidden_dim': 256,
            'use_mclc': False, 'aggregate_modalities': True, 'crmsa_heads': 8,
        },
        'training': {
            'batch_size': 1, 'num_epochs': 25, 'learning_rate': 1e-4, 'weight_decay': 1e-5,
            'scheduler': {'type': 'plateau'},
            'early_stopping': {'patience': 10, 'monitor': 'val_auc', 'mode': 'max'},
            'use_amp': False, 'focal_loss': False, 'label_smoothing': 0.0,
            'kd_enabled': False, 'modality_dropout': 0.0, 'aux_loss_weight': 0.0,
        },
        'data_split': {'val_start': 100},
        'environment': {'device': 'cuda:0', 'num_workers': 2, 'seed': seed},
        'output': None,
    }


def launch(exp_name, seed, pr_dir, gpu):
    """Prepare config + script, launch subprocess, return (proc, out_dir)."""
    out_dir = BASE_OUT / exp_name / f"seed{seed}"
    for d in [out_dir / "ckpt", out_dir / "logs", out_dir / "img"]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = build_config(seed, pr_dir)
    cfg['output'] = {
        'save_dir': str(out_dir / 'ckpt'),
        'log_dir': str(out_dir / 'logs'),
        'img_dir': str(out_dir / 'img'),
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    script = build_run_script(exp_name, seed, pr_dir, gpu, out_dir)

    log_f = open(out_dir / "logs" / "stdout.log", 'w')
    proc = subprocess.Popen(
        [PYTHON, str(script)],
        stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(PROJECT),
    )
    return proc, log_f, out_dir


def extract_auc(out_dir):
    log_file = out_dir / "logs" / "stdout.log"
    if log_file.exists():
        for line in log_file.read_text().split('\n'):
            if 'DONE AUC=' in line:
                try:
                    return float(line.split('AUC=')[1].split()[0])
                except:
                    pass
    return None


def main():
    print("=" * 70)
    print("C16 Two-stage R²T — Real PR vs Random PR")
    print(f"Seeds: {SEEDS}")
    print(f"GPUs:  {GPU_MAP}")
    print(f"Python: {PYTHON}")
    print("=" * 70)

    all_procs = []

    # ── Launch all experiments ──
    # Real PR and Random PR for the same seed share a GPU → run simultaneously
    # is fine since each seed uses a different GPU, and Real+Random on same GPU
    # will contend → we launch Real first, wait briefly, then launch Random
    # Actually, just launch everything — GPU memory (~10GB for V100-16GB) should fit
    # two processes on 32GB GPUs. For 16GB GPUs it may OOM.

    for seed in SEEDS:
        # Real PR
        gpu = GPU_MAP[('real_PR', seed)]
        proc, log_f, out_dir = launch("real_PR", seed, "C16_PR_features", gpu)
        all_procs.append((f"real_PR_s{seed}", proc, log_f, out_dir))
        print(f"  real_PR  seed={seed:>4}  GPU={gpu}")

        # Random PR
        gpu = GPU_MAP[('random_PR', seed)]
        proc, log_f, out_dir = launch("random_PR", seed, "C16_PR_random_features", gpu)
        all_procs.append((f"random_PR_s{seed}", proc, log_f, out_dir))
        print(f"  random_PR seed={seed:>4}  GPU={gpu}")

    print(f"\n▶ {len(all_procs)} jobs running. Waiting...\n")

    # ── Wait and report ──
    for name, proc, log_f, out_dir in all_procs:
        ret = proc.wait()
        log_f.close()
        auc = extract_auc(out_dir)
        status = "✓" if ret == 0 else f"✗(rc={ret})"
        auc_s = f"AUC={auc:.4f}" if auc else "?"
        print(f"  [{status}] {name:<20s}  {auc_s}  → {out_dir}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for exp_type in ["real_PR", "random_PR"]:
        aucs = []
        print(f"\n{exp_type}:")
        for seed in SEEDS:
            out_dir = BASE_OUT / exp_type / f"seed{seed}"
            auc = extract_auc(out_dir)
            if auc:
                aucs.append(auc)
                print(f"  seed={seed:>4}: AUC={auc:.4f}")
            else:
                print(f"  seed={seed:>4}: AUC=? (check {out_dir}/logs/stdout.log)")
        if aucs:
            mean = sum(aucs) / len(aucs)
            print(f"  Mean AUC: {mean:.4f} ± {max(abs(a-mean) for a in aucs):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
