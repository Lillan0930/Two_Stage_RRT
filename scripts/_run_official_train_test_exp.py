#!/usr/bin/env python3
"""
Single-experiment runner — 官方 Train/Test 协议（270 train / 129 test，无 val）。

- 从 <out_dir>/config.json 读配置（data.no_validation=True）
- Trainer 固定 num_epochs 训练（无 early stopping / val 选择），存 last checkpoint
- 加载 last checkpoint，在官方 129 test 上评估
- 保存 metrics.json + test_predictions.csv
"""
import os, sys, json, time, logging
import torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
    recall_score

OUT_DIR = Path(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "7")

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from train import Trainer, build_feature_dirs
from models.mm_rrt_abmil import MM_RRT_ABMIL
from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

with open(OUT_DIR / "config.json") as f:
    cfg = json.load(f)

logger = logging.getLogger("exp")
logger.handlers.clear()
logger.setLevel(logging.INFO)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(OUT_DIR / "logs" / "run.log"))
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

seed = cfg["environment"]["seed"]
t0 = time.time()

trainer = Trainer(cfg, logger, f"s{seed}")
_, _ = trainer.train()
train_time = time.time() - t0
print(f"TRAIN_TIME={train_time:.0f}", flush=True)

# ── Test eval with per-slide predictions ──
ckpt_path = OUT_DIR / "ckpt" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
mc = ckpt["config"]["model"]
dc = cfg["data"]
num_mod = len(dc["modalities"])

best_epoch = ckpt.get("epoch", -1)

model = MM_RRT_ABMIL(
    num_modalities=num_mod, modality_list=dc["modalities"],
    input_dim=768, num_classes=2,
    mlp_dim=mc.get("mlp_dim", 512), region_num=mc.get("region_num", 4),
    n_layers=mc.get("n_layers", 2), n_heads=mc.get("n_heads", 4),
    drop_path=mc.get("drop_path", 0.0), trans_dropout=mc.get("trans_dropout", 0.1),
    epeg=mc.get("epeg", True), epeg_k=mc.get("epeg_k", 9),
    crmsa_k=mc.get("crmsa_k", 3), cr_msa=mc.get("cr_msa", True),
    all_shortcut=mc.get("all_shortcut", True),
    crmsa_heads=mc.get("crmsa_heads", 8), crmsa_mlp=mc.get("crmsa_mlp", False),
    fusion_type=mc.get("fusion_type", "two_stage_region"),
    stage2_type=mc.get("stage2_type", "staining_msa"),
    abmil_hidden_dim=mc.get("abmil_hidden_dim", 256),
    encoder_cfg=mc.get("encoder_cfg"),
    stage2_cfg=mc.get("stage2_cfg"),
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model = model.cuda().eval()

feature_dirs = build_feature_dirs(dc["feature_base_dir"], dc["modalities"], dc.get("dir_mapping"))
mp = dc.get("max_patches", 5000)
sp = dc.get("sampling", "first")
ss = dc.get("sample_seed", 0)

test_ds = C16MultimodalDataset(
    feature_dirs=feature_dirs,
    label_file=str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
    max_patches=mp if mp > 0 else None, preload=False, verbose=False,
    sampling=sp, sample_seed=ss, per_epoch=False,
)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                     collate_fn=c16_multimodal_collate_fn, num_workers=0,
                     pin_memory=True)

slide_ids_list, probs_list, labels_list, preds_list = [], [], [], []
with torch.no_grad():
    for batch in test_dl:
        feats = [torch.stack(m).cuda() for m in batch["features"]]
        logits, _, _, _ = model(feats)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        pred = int(torch.argmax(logits, dim=-1)[0].item())
        slide_ids_list.append(batch["slide_ids"][0])
        probs_list.append(prob)
        labels_list.append(batch["labels"][0].item())
        preds_list.append(pred)

labels_np = np.array(labels_list)
probs_np = np.array(probs_list)
preds_np = np.array(preds_list)

metrics = {
    "seed": int(seed),
    "best_epoch": int(best_epoch),
    "auc": float(roc_auc_score(labels_np, probs_np)),
    "accuracy": float(accuracy_score(labels_np, preds_np)),
    "f1": float(f1_score(labels_np, preds_np)),
    "sensitivity": float(recall_score(labels_np, preds_np)),
    "specificity": float(recall_score(1 - labels_np, 1 - preds_np)),
    "train_time_s": float(train_time),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Per-slide predictions
pred_df = pd.DataFrame({
    "slide_id": slide_ids_list,
    "label": labels_list,
    "probability": [round(p, 6) for p in probs_list],
    "prediction": preds_list,
})
pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False)

print(f"TEST_AUC={metrics['auc']:.4f} ACC={metrics['accuracy']:.4f} "
      f"F1={metrics['f1']:.4f} SENS={metrics['sensitivity']:.4f} "
      f"SPEC={metrics['specificity']:.4f}", flush=True)
