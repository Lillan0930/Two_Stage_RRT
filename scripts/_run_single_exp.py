#!/usr/bin/env python3
"""Single-experiment runner: reads config.json, runs Trainer, evaluates on test."""
import os, sys, json, time, logging
import torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

OUT_DIR = Path(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
_, val_auc = trainer.train()
train_time = time.time() - t0
print(f"VAL_AUC={val_auc:.4f} TIME={train_time:.0f}", flush=True)

# Test eval
ckpt_path = OUT_DIR / "ckpt" / "best_model.pt"
ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
mc = ckpt["config"]["model"]
dc = cfg["data"]
num_mod = len(dc["modalities"])

model = MM_RRT_ABMIL(
    num_modalities=num_mod, input_dim=768, num_classes=2,
    mlp_dim=mc.get("mlp_dim",512), region_num=mc.get("region_num",4),
    n_layers=mc.get("n_layers",2), n_heads=mc.get("n_heads",4),
    drop_path=mc.get("drop_path",0.0), trans_dropout=mc.get("trans_dropout",0.1),
    epeg=mc.get("epeg",True), epeg_k=mc.get("epeg_k",9),
    crmsa_k=mc.get("crmsa_k",3), cr_msa=mc.get("cr_msa",True),
    all_shortcut=mc.get("all_shortcut",True),
    crmsa_heads=mc.get("crmsa_heads",8), crmsa_mlp=mc.get("crmsa_mlp",False),
    fusion_type="two_stage_region",
    abmil_hidden_dim=mc.get("abmil_hidden_dim",256),
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
    sampling=sp, sample_seed=ss,
)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                     collate_fn=c16_multimodal_collate_fn, num_workers=2,
                     pin_memory=True, persistent_workers=True)

logits_list, labels_list = [], []
with torch.no_grad():
    for batch in test_dl:
        feats = [torch.stack(m).cuda() for m in batch["features"]]
        logits, _, _, _ = model(feats)
        logits_list.append(logits.cpu())
        labels_list.append(batch["labels"].cpu())

logits = torch.cat(logits_list)
labels_np = torch.cat(labels_list).numpy()
probs = torch.softmax(logits, dim=-1)[:,1].numpy()
preds = torch.argmax(logits, dim=-1).numpy()

result = {
    "val_auc": float(val_auc),
    "test_auc": float(roc_auc_score(labels_np, probs)),
    "test_acc": float(accuracy_score(labels_np, preds)),
    "test_f1": float(f1_score(labels_np, preds)),
    "train_time_s": float(train_time),
}
with open(OUT_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"TEST_AUC={result['test_auc']:.4f} ACC={result['test_acc']:.4f} F1={result['test_f1']:.4f}", flush=True)
