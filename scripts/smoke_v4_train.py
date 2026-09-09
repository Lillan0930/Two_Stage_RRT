#!/usr/bin/env python3
"""v4 训练冒烟测试：用 run_stage2_v4.py 的 config 构建真实模型，
跑一个合成 forward+backward，确认 Q/K centering 生效、mu_pr EMA 更新、梯度流动、
无 shape 错误、诊断 12 键可算。"""
import os, sys, json
from pathlib import Path

import torch

PROJECT = Path("/home/Public/lillan/Two_Sage_RRT-/TwoStageRRT")
sys.path.insert(0, str(PROJECT))
os.chdir(str(PROJECT))

from scripts.run_stage2_v4 import build_config

cfg = build_config(42)
import tempfile
_tmp = Path(tempfile.mkdtemp(prefix="smoke_v4_"))
cfg["output"]["save_dir"] = str(_tmp / "ckpt")
cfg["output"]["log_dir"] = str(_tmp / "logs")
cfg["output"]["img_dir"] = str(_tmp / "img")
for _d in ["save_dir", "log_dir", "img_dir"]:
    Path(cfg["output"][_d]).mkdir(parents=True, exist_ok=True)
from train import Trainer
import logging
_logger = logging.getLogger("smoke_v4")
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())

trainer = Trainer(cfg, _logger, "smoke")
model = trainer.create_model()
model.train()

N = 32
device = next(model.parameters()).device
he = torch.randn(1, N, 768, device=device)
pr = torch.randn(1, N, 768, device=device)

print("stage2_type:", cfg["model"]["stage2_type"])
print("mu_pr before:", model.cross_region_mod.mu_pr.abs().sum().item(),
      "init_flag:", model.cross_region_mod.prototype_initialized.item())

out = model([he, pr])
logits = out[0] if isinstance(out, tuple) else out
print("logits shape:", tuple(logits.shape))

loss = logits.square().sum()
loss.backward()

print("mu_pr after:", model.cross_region_mod.mu_pr.abs().sum().item(),
      "init_flag:", model.cross_region_mod.prototype_initialized.item())

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
buf_names = [n for n, _ in model.cross_region_mod.named_buffers()]
print(f"trainable params: {n_params}, with grad: {n_grad}")
print("cross_region_mod buffers:", buf_names)

assert model.cross_region_mod.prototype_initialized.item() == 1.0
assert model.cross_region_mod.mu_pr.abs().sum().item() > 0
assert n_grad > 0

# Q/K centering diagnostics computable in eval
model.eval()
with torch.no_grad():
    d = model.cross_region_mod.diagnose(
        model.rrt_he(model.patch_to_emb[0](model.dp(he))),
        model.rrt_ihc(model.patch_to_emb[1](model.dp(pr))))
    for k in ['q_raw_cos', 'q_centered_cos', 'k_raw_cos', 'k_centered_cos',
              'attn_weight_std', 'entropy_norm', 'score_std', 'r_slide', 'r_region']:
        assert torch.isfinite(d[k]).all(), f"{k} not finite"
    print("q_raw_cos %.4f → q_centered_cos %.4f" % (
        d['q_raw_cos'].item(), d['q_centered_cos'].item()))
    print("k_raw_cos %.4f → k_centered_cos %.4f" % (
        d['k_raw_cos'].item(), d['k_centered_cos'].item()))

# eval freeze check
mu_before = model.cross_region_mod.mu_pr.clone()
with torch.no_grad():
    model([he, pr])
assert torch.equal(mu_before, model.cross_region_mod.mu_pr), "eval must freeze mu_pr"

opt, _ = trainer.create_optimizer_scheduler(model)
total_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
print(f"optimizer params: {total_opt} (trainable={n_params})")
assert total_opt == n_params, "optimizer must cover ALL trainable params (mu_pr is buffer)"

print("\nSMOKE PASS")
