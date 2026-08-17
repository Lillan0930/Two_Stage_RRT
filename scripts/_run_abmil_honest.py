#!/usr/bin/env python3
"""
Honest plain-ABMIL baseline —— 精确复现 Base_mil config_c16_pr_best_acc.yaml 的
AttentionMIL（hidden 384 / classifier_depth 1 / focal γ2.4 / lr 1.88e-4 / wd 3.8e-6），
但使用诚实协议：修正的 2500-patch 随机采样 + fixed 25 epochs + cosine + LAST checkpoint
（无 early stopping、无 test-set 选模型）。

读 <out_dir>/config.json（data.model.training），训练后评估官方 129 test，
写 metrics.json + test_predictions.csv。
"""
import os, sys, json, time, logging
import torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

OUT_DIR = Path(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("EXP_GPU", "7")

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from data.c16_multimodal_dataset import C16MultimodalDataset, c16_multimodal_collate_fn

with open(OUT_DIR / "config.json") as f:
    cfg = json.load(f)

logger = logging.getLogger("abmil")
logger.handlers.clear()
logger.setLevel(logging.INFO)
(OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(str(OUT_DIR / "logs" / "run.log"))
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(fh)

seed = cfg["environment"]["seed"]
torch.manual_seed(seed); np.random.seed(seed)


class AttentionMIL(nn.Module):
    """Base_mil AttentionMIL (classifier_depth=1) 复现。"""
    def __init__(self, input_dim=768, hidden_dim=384, num_classes=2, dropout_rate=0.101):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout_rate))
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, num_classes)  # depth 1

    def forward(self, x):
        A = self.attention_w(self.attention_V(x)).transpose(0, 1)
        A = F.softmax(A, dim=1)
        Z = torch.mm(A, x)
        return self.classifier(Z).squeeze(0)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            fl = self.alpha[labels] * fl
        return fl.mean()


dc, mc, tc = cfg["data"], cfg["model"], cfg["training"]
feature_dirs = {"PR": str(Path(dc["feature_base_dir"]) / "C16_PR_features")}

train_ds = C16MultimodalDataset(
    feature_dirs, dc["train_label_file"], max_patches=dc["max_patches"],
    preload=False, verbose=False, sampling="random", sample_seed=42, per_epoch=True)
test_ds = C16MultimodalDataset(
    feature_dirs, str(PROJECT / "data/C16_labels/c16_test_labels.csv"),
    max_patches=dc["max_patches"], preload=False, verbose=False,
    sampling="random", sample_seed=42, per_epoch=False)

train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                      collate_fn=c16_multimodal_collate_fn, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False,
                     collate_fn=c16_multimodal_collate_fn, num_workers=0)

model = AttentionMIL(768, mc["hidden_dim"], 2, mc["dropout_rate"]).cuda()

# auto inverse-freq class weights（与 train.py 一致）
from collections import Counter
cnt = Counter(s["label"] for s in train_ds.samples)
n = len(train_ds.samples)
weights = torch.tensor([n / (2 * cnt[c]) for c in range(2)]).cuda()

use_focal = tc.get("focal_loss", False)
if use_focal:
    criterion = FocalLoss(gamma=tc.get("focal_gamma", 2.0), alpha=weights)
else:
    criterion = nn.CrossEntropyLoss(weight=weights)

opt = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"],
                       weight_decay=tc["weight_decay"])
epochs = tc["num_epochs"]
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

t0 = time.time()
for ep in range(epochs):
    model.train()
    train_ds.set_epoch(ep)
    tot, cor, totloss = 0, 0, 0.0
    for b in train_dl:
        feats = b["features"][0][0].cuda()          # [N,768]
        lab = b["labels"].cuda()                     # [1]
        logits = model(feats).unsqueeze(0)           # [1,2]
        loss = criterion(logits, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += 1
        cor += (logits.argmax(-1) == lab).sum().item()
        totloss += loss.item()
    sched.step()
    if ep % 5 == 0 or ep == epochs - 1:
        logger.info(f"Epoch {ep+1}/{epochs} Loss {totloss/tot:.4f} Acc {cor/tot:.4f}")

torch.save({"epoch": epochs - 1, "model_state_dict": model.state_dict()},
           OUT_DIR / "ckpt" / "best_model.pt")
train_time = time.time() - t0

# ── 129 test eval ──
model.eval()
ids, probs, labels = [], [], []
with torch.no_grad():
    for b in test_dl:
        feats = b["features"][0][0].cuda()
        logits = model(feats)
        p = F.softmax(logits, dim=0)[1].item()
        ids.append(b["slide_ids"][0])
        probs.append(p)
        labels.append(b["labels"][0].item())

labels = np.array(labels); probs = np.array(probs)
preds = (probs >= 0.5).astype(int)
metrics = {
    "seed": int(seed),
    "auc": float(roc_auc_score(labels, probs)),
    "accuracy": float(accuracy_score(labels, preds)),
    "f1": float(f1_score(labels, preds)),
    "sensitivity": float(recall_score(labels, preds)),
    "specificity": float(recall_score(1 - labels, 1 - preds)),
    "train_time_s": float(train_time),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
pd.DataFrame({"slide_id": ids, "label": labels, "probability": probs,
              "prediction": preds}).to_csv(OUT_DIR / "test_predictions.csv", index=False)
print(f"AUC={metrics['auc']:.4f} ACC={metrics['accuracy']:.4f} "
      f"SENS={metrics['sensitivity']:.4f} SPEC={metrics['specificity']:.4f}", flush=True)
