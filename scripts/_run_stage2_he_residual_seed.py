#!/usr/bin/env python3
"""Stage2 改造实验 单 seed worker（训练 + 官方 Test 评估 + 产物）。

读取一个完整 config.json（含 modalities / encoder_cfg / stage2_cfg / seed /
learning_rate），在指定 GPU 上训练，随后用 `trainer.train()` 返回的（已加载
best checkpoint 的）模型直接在官方 Test（129）上评估，写：

  - result.json            : best_val_auc / test_auc / acc / f1 / sens / spec /
                             precision / best_epoch / train_time_s / stage2 元信息
  - test_predictions.csv   : 129 test 的 slide_id,label,probability,prediction

复用 Trainer 的模型（不手工重建），避免 `_run_fixed_split_exp.py` 里重建模型时
dropout 等参数缺省不一致的问题。

用法:
  python scripts/_run_stage2_he_residual_seed.py --config <path/config.json> --gpu 3
"""
import os, sys, json, time, logging, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
    recall_score, precision_score

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

TEST_LABEL_FILE = str(PROJECT / "data/C16_labels/c16_test_labels.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from train import Trainer, build_feature_dirs
    from data.c16_multimodal_dataset import (
        C16MultimodalDataset, c16_multimodal_collate_fn)

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())
    out_dir = cfg_path.parent
    seed = cfg["environment"]["seed"]

    logger = logging.getLogger(f"stage2_seed{seed}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(out_dir / "logs" / "run.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    logger.info(f"stage2 seed={seed} modalities={cfg['data']['modalities']} "
                f"stage2_type={cfg['model'].get('stage2_type')} "
                f"lr={cfg['training'].get('learning_rate')}")

    t0 = time.time()
    trainer = Trainer(cfg, logger, f"s{seed}")
    model, _best_metric = trainer.train()
    train_time = time.time() - t0

    model.eval()
    device = next(model.parameters()).device

    # ── Official Test (129) eval ──
    data_cfg = cfg["data"]
    feature_dirs = build_feature_dirs(
        data_cfg["feature_base_dir"], data_cfg["modalities"],
        data_cfg.get("dir_mapping", None))
    mp = data_cfg.get("max_patches", 2500)
    sp = data_cfg.get("sampling", "random")
    ss = data_cfg.get("sample_seed", seed)

    test_ds = C16MultimodalDataset(
        feature_dirs=feature_dirs,
        label_file=TEST_LABEL_FILE,
        max_patches=mp if mp > 0 else None, preload=False, verbose=False,
        sampling=sp, sample_seed=ss, per_epoch=False,
    )
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                         collate_fn=c16_multimodal_collate_fn, num_workers=0,
                         pin_memory=True)

    slide_ids_list, probs_list, labels_list, preds_list = [], [], [], []
    with torch.no_grad():
        for batch in test_dl:
            feats = [torch.stack(m).to(device) for m in batch["features"]]
            logits = model(feats)[0]
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = int(torch.argmax(logits, dim=-1)[0].item())
            slide_ids_list.append(batch["slide_ids"][0])
            probs_list.append(prob)
            labels_list.append(batch["labels"][0].item())
            preds_list.append(pred)

    labels_np = np.array(labels_list)
    probs_np = np.array(probs_list)
    preds_np = np.array(preds_list)

    result = {
        "seed": int(seed),
        "best_epoch": int(trainer.best_epoch),
        "best_val_auc": float(trainer.best_val_auc),
        "test_auc": float(roc_auc_score(labels_np, probs_np)),
        "test_acc": float(accuracy_score(labels_np, preds_np)),
        "test_f1": float(f1_score(labels_np, preds_np)),
        "test_sensitivity": float(recall_score(labels_np, preds_np)),
        "test_specificity": float(recall_score(1 - labels_np, 1 - preds_np)),
        "test_precision": float(precision_score(labels_np, preds_np)),
        "train_time_s": float(train_time),
        "stage2_type": cfg["model"].get("stage2_type", None),
        "residual_scale": cfg["model"].get("stage2_cfg", {}).get("residual_scale", None),
        "disable_cross": cfg["model"].get("stage2_cfg", {}).get("disable_cross", None),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "test_predictions.csv", "w") as f:
        f.write("slide_id,label,probability,prediction\n")
        for sid, lbl, p, pr in zip(slide_ids_list, labels_list, probs_list, preds_list):
            f.write(f"{sid},{int(lbl)},{p:.6f},{int(pr)}\n")

    print(f"[seed {seed} gpu {args.gpu}] val_auc={result['best_val_auc']:.4f} "
          f"test_auc={result['test_auc']:.4f} acc={result['test_acc']:.4f} "
          f"f1={result['test_f1']:.4f} epoch={result['best_epoch']}", flush=True)


if __name__ == "__main__":
    main()
