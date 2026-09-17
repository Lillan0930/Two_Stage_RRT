#!/usr/bin/env python
"""跑 Ours 主实验（C17 + ABMIL）的**单个** (组合, seed) run。

一个 run = 一个独立进程 = 一张 GPU（由 ``CUDA_VISIBLE_DEVICES`` 钉住）。
失败只会写进这个 run 自己的 ``train.log`` / ``status.json``，不影响别的任务。

产物（规格 §7），全部落在 ``<out_root>/<组合>/seed<seed>/``：

    config.yaml      本 run 的完整 resolved config
    train.log        stdout/stderr + trainer 日志
    best_model.pt    trainer 存的 best checkpoint
    metrics.json     seed / best_epoch / 6 个主指标 / 口径变体
    history.csv      逐 epoch 曲线
    status.json      running | completed | failed
"""
import argparse
import csv
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
import yaml                                                   # noqa: E402

import train as T                                             # noqa: E402
from ours_main_c17_abmil import (                             # noqa: E402
    MODALITY_ORDER, build_config, dump_config_yaml, primary_metrics,
    secondary_metrics, seed_dir,
)


def seed_everything(seed):
    """规格 §4 要求的 5 个入口，一个都不少。

    ``train.set_seed`` 只覆盖 torch/np（且不含 ``random`` 与
    ``torch.cuda.manual_seed``），所以这里显式补齐，而不是调它。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class HistoryTrainer(T.Trainer):
    """在**不修改** train.py 的前提下，把逐 epoch 曲线和最终 best-model 指标取出来。

    两个钩子：
      * ``train_epoch`` —— 记录本 epoch 的 lr 与 train_loss
      * ``validate``    —— ``return_probs=True`` 的那一次是训练结束后加载
        best checkpoint 的最终评估，把它的 metrics 存为最终指标；
        其余（每 epoch 一次）按 epoch 号记进 history。

    ``self.current_epoch`` 在 epoch 循环里于 ``train_epoch`` 之前设置，且训练
    结束后的最终评估会复用同一个 epoch 号 —— 所以 history 采取"该 epoch 已有
    记录就不再覆盖"，先写入的那条才是真正的 epoch 指标。
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.epoch_rows = []
        self.final_metrics = None
        self._seen_epochs = set()
        self._last_lr = None

    def create_optimizer_scheduler(self, model):
        """把实际建出来的 optimizer / scheduler 留在 self 上。

        ``Trainer.train()`` 里它们是局部变量，不落 self；规格 §2/§7 要求记录
        **实际生效**的 optimizer / scheduler / eta_min，所以在这里接一下。
        只是把引用存下来，不改任何构造逻辑。
        """
        optimizer, scheduler = super().create_optimizer_scheduler(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        return optimizer, scheduler

    def train_epoch(self, model, train_loader, criterion, optimizer, scaler=None):
        try:
            self._last_lr = float(optimizer.param_groups[0]['lr'])
        except Exception:                       # noqa: BLE001 - lr 只是记录项
            self._last_lr = None
        return super().train_epoch(model, train_loader, criterion, optimizer, scaler=scaler)

    def validate(self, model, val_loader, criterion, return_probs=False):
        out = super().validate(model, val_loader, criterion, return_probs=return_probs)
        metrics = out[1] if isinstance(out, tuple) else out
        # 本 epoch 的 val loss 只能从返回值取：调用方是在 validate 返回之后才
        # self.val_losses.append(val_loss) 的，读 self.val_losses[-1] 会拿到上一个 epoch。
        cur_val_loss = float(out[0]) if isinstance(out, tuple) else 0.0

        if return_probs:
            # 训练结束、加载 best checkpoint 之后的那一次 = 正式指标
            self.final_metrics = metrics
            return out

        ep = int(getattr(self, 'current_epoch', len(self.epoch_rows))) + 1
        if ep in self._seen_epochs:
            return out
        self._seen_epochs.add(ep)

        def g(k):
            v = metrics.get(k)
            return float(v) if v is not None else 0.0

        row = {
            'epoch': ep,
            'train_loss': float(self.train_losses[-1]) if self.train_losses else 0.0,
            'val_auc': g('auc'),
            'val_accuracy': g('accuracy'),
            'val_recall': g('sensitivity_class_1'),          # 肿瘤召回
            'val_precision': g('precision_class_1'),         # 肿瘤精确率
            'lr': self._last_lr if self._last_lr is not None else 0.0,
        }
        p, r = row['val_precision'], row['val_recall']
        row['val_f1'] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        row['val_specificity'] = g('sensitivity_class_0')    # 正常类召回（≠ specificity_class_0）
        row['val_f1_macro'] = g('f1')
        row['val_recall_macro'] = g('recall')
        row['val_precision_macro'] = g('precision')
        row['val_loss'] = cur_val_loss
        self.epoch_rows.append(row)
        return out


HISTORY_COLUMNS = [
    'epoch', 'train_loss', 'val_auc', 'val_accuracy', 'val_recall',
    'val_precision', 'val_f1', 'val_specificity', 'lr',
    # 口径变体，附在主列之后，便于复核
    'val_loss', 'val_f1_macro', 'val_recall_macro', 'val_precision_macro',
]


def write_status(sd, status, **extra):
    payload = {"status": status, **extra}
    tmp = sd / "status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(sd / "status.json")          # 原子替换，避免半截文件被当成完成


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True, help="组合目录名，如 HE+PR+ER")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu", type=int, default=None, help="仅用于日志，实际由 CUDA_VISIBLE_DEVICES 决定")
    ap.add_argument("--out-root", default=None, help="覆写结果根目录（仅冒烟用）")
    ap.add_argument("--out-dir", default=None,
                    help="直接指定本 run 的产物目录（HE-only baseline 用）。"
                         "默认 <out_root>/<组合>/seed<seed>。"
                         "注意：不改变 O.OUT_ROOT，故 config 里的 seeds_file 仍指向"
                         "共享的 seeds.json。")
    ap.add_argument("--epochs", type=int, default=None,
                    help="覆写 num_epochs（仅冒烟用；正式 140 runs 绝不传这个参数）")
    args = ap.parse_args()

    if args.out_root:
        import ours_main_c17_abmil as _O
        _O.OUT_ROOT = Path(args.out_root)
    if args.epochs is not None:
        import ours_main_c17_abmil as _O
        _O.TRAINING["num_epochs"] = args.epochs

    name, seed = args.combo, args.seed
    modalities = [m for m in MODALITY_ORDER
                  if m in name.split("+")]
    assert modalities and modalities[0] == "HE", f"组合名无法解析: {name!r}"

    sd = Path(args.out_dir) if args.out_dir else seed_dir(name, seed)
    sd.mkdir(parents=True, exist_ok=True)

    write_status(sd, "running", combo=name, seed=seed, gpu=args.gpu,
                 pid=os.getpid(),
                 started_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"))

    cfg = build_config(name, modalities, seed)
    # ``build_config`` 内部把 output.* 定为 <out_root>/<组合>/seed<seed>；
    # 传了 --out-dir 就必须把这三条一起改写，否则 trainer 的
    # best_model.pt / img/ / logs/ 会落回组合目录，跟本 run 的
    # metrics.json / status.json 分家，还会多出一个形似组合的目录
    # （HE-only baseline 踩过这个坑：he/seedX 有 status.json=completed，
    #  却没有 best_model.pt，断点续跑判据永远判否）。
    cfg["output"] = {
        "save_dir": str(sd),
        "log_dir": str(sd / "logs"),
        "img_dir": str(sd / "img"),
    }
    cfg_path = sd / "config.yaml"
    dump_config_yaml(cfg, cfg_path)          # 浮点写十进制，dump→load 类型无损

    try:
        seed_everything(seed)

        logger, timestamp = T.setup_logging(cfg['output']['log_dir'],
                                            f"ours_{name.replace('+', '_')}_s{seed}")
        logger.info(f"[ours-c17] combination={name} seed={seed} gpu={args.gpu}")
        logger.info(f"[ours-c17] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                    f"→ torch.cuda.device_count()={torch.cuda.device_count()}")
        logger.info(f"[ours-c17] modalities={modalities}")
        logger.info(f"[ours-c17] config: {json.dumps(cfg)}")

        trainer = HistoryTrainer(cfg, logger, timestamp)
        model, best_metric = trainer.train()

        # 本 run 的 CUDA 显存峰值 —— 用于 GPU 调度决策（能否同卡双开）
        try:
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        except Exception:                                      # noqa: BLE001
            peak_alloc = peak_reserved = None
        logger.info(f"[ours-c17] peak CUDA memory: allocated={peak_alloc} MB, "
                    f"reserved={peak_reserved} MB")

        if trainer.final_metrics is None:
            raise RuntimeError("训练结束但没有拿到 best-model 的最终评估指标")

        pm = primary_metrics(trainer.final_metrics)
        # 规格 §2/§7：把**实际生效**的 optimizer / scheduler 记下来，
        # 避免再出现"配置写 AdamW、实际跑 Adam"这类不一致。
        sched = trainer.scheduler
        actual = {
            "optimizer_actual": getattr(trainer, "optimizer_type_actual", None),
            "scheduler_actual": type(sched).__name__ if sched is not None else None,
            "eta_min_actual": getattr(trainer, "eta_min_actual", None),
            "lr_actual": float(trainer.optimizer.param_groups[0]["lr"]),
            "weight_decay_actual": float(trainer.optimizer.param_groups[0]["weight_decay"]),
            "optimizer_class": type(trainer.optimizer).__name__,
        }
        payload = {
            # 规格 §12：断点续跑的判据就读这个字段
            "status": "completed",
            "seed": seed,
            "combination": name,
            "modalities": modalities,
            "best_epoch": int(trainer.best_epoch) + 1,       # trainer 内部 0-based
            "best_val_metric": float(best_metric),
            "monitor": trainer.monitor_metric_name,
            "epochs_run": len(trainer.epoch_rows),
            "peak_cuda_memory_mb": (round(peak_alloc, 1)
                                    if peak_alloc is not None else None),
            "peak_cuda_reserved_mb": (round(peak_reserved, 1)
                                      if peak_reserved is not None else None),
            **pm,
            **actual,
            "secondary": secondary_metrics(trainer.final_metrics),
            "metric_definitions": {
                "task": "C17 二分类（label ∈ {0,1}），沿用历史任务定义",
                "AUC": "roc_auc_score(y_true, P(class 1))，阈值无关",
                "Accuracy": "accuracy_score",
                "Recall": "sensitivity_class_1 = TP/(TP+FN)，class 1 召回（非 macro）",
                "Precision": "precision_class_1 = TP/(TP+FP)，class 1 精确率",
                "F1": "class 1 的 F1 = 2PR/(P+R)",
                "Specificity": "sensitivity_class_0 = TN/(TN+FP)，class 0 召回"
                               "（二分类下 == specificity_class_1）",
                "macro_variants": "macro 口径另存于 secondary：f1_macro / precision_macro / recall_macro",
                "note": "Specificity 不能用 specificity_class_0 —— utils/metrics.py 的 per-class "
                        "循环把第 i 类当正类，specificity_class_0 化简后等于 class 1 召回。",
            },
        }
        (sd / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

        with open(sd / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
            w.writeheader()
            for row in trainer.epoch_rows:
                w.writerow({k: row.get(k, "") for k in HISTORY_COLUMNS})

        write_status(sd, "completed", combo=name, seed=seed, gpu=args.gpu,
                     best_epoch=payload["best_epoch"], AUC=pm["AUC"],
                     epochs_run=payload["epochs_run"],
                     finished_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"))

        logger.info(f"[ours-c17] DONE {name} seed={seed} "
                    f"AUC={pm['AUC']:.4f} best_epoch={payload['best_epoch']} "
                    f"epochs={payload['epochs_run']}")
        # 直接 _exit，绕开 DataLoader 常驻 worker 与残留 CUDA context 的清理等待；
        # 先 flush 日志，否则最后几行会丢。
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        # 进程正常退出码必须是 0（即使 AUC 很低）——只有异常才是失败
        os._exit(0)

    except BaseException as exc:                              # noqa: BLE001
        tb = traceback.format_exc()
        try:
            with open(sd / "train.log", "a") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"[ours-c17] FAILED {name} seed={seed}\n")
                f.write(tb + "\n")
        except Exception:                                     # noqa: BLE001
            pass
        write_status(sd, "failed", combo=name, seed=seed, gpu=args.gpu,
                     error=f"{type(exc).__name__}: {exc}", traceback=tb[-4000:],
                     finished_at=__import__("datetime").datetime.now().isoformat(timespec="seconds"))
        print(tb, file=sys.stderr, flush=True)
        os._exit(1)


if __name__ == "__main__":
    main()
