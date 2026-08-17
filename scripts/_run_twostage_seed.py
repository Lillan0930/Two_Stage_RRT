#!/usr/bin/env python3
"""Two-stage (HE+PR) Stage-2 单 config 单 seed runner（由 run_twostage_stage2_optuna.py 驱动）。

读取一个完整 config.json（已含 crmsa_k/drop_path/stage2_lr/seed），在指定 GPU 上训练，
写 result.json（val_auc/val_acc/... 即 129 test-as-val 指标），清理 ckpt/img。

用法:
  python scripts/_run_twostage_seed.py --config <path/config.json> --gpu 3
"""
import os, sys, json, shutil, logging, argparse
from pathlib import Path

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    # 设好 env 再 import train，避免 CUDA 提前初始化到错误设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from train import Trainer

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out_dir = cfg_path.parent
    seed = cfg["environment"]["seed"]
    sc = cfg["model"]["stage2_cfg"]

    logger = logging.getLogger(f"ts_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"two-stage seed={seed} | crmsa_k={sc['crmsa_k']} "
                f"drop_path={sc['drop_path']} stage2_lr={cfg['training']['lr_stage2']}")

    trainer = Trainer(cfg, logger, f"s{seed}")
    trainer.train()

    result = {
        "seed": seed,
        "val_auc": float(trainer.best_val_auc),
        "val_acc": float(trainer.best_val_acc),
        "val_sensitivity": float(trainer.best_val_sensitivity),
        "val_specificity": float(trainer.best_val_specificity),
        "best_epoch": int(trainer.best_epoch),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[seed {seed} gpu {args.gpu}] val_auc={result['val_auc']:.4f} "
          f"acc={result['val_acc']:.4f} epoch={result['best_epoch']}", flush=True)

    # 清理 ckpt/img 省盘，保留 config/result/run.log
    shutil.rmtree(out_dir / "ckpt", ignore_errors=True)
    shutil.rmtree(out_dir / "img", ignore_errors=True)


if __name__ == "__main__":
    main()
