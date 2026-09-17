"""Ours 主实验（C16 + ABMIL）的共享定义：组合表、配置生成、指标口径、seeds 管理。

被 ``run_one_ours_seed.py``（单 run）与 ``drive_ours_main_c16_abmil.py``（队列驱动）共用。

**公平性（规格 §13）**：14 个组合之间只允许 ``data.modalities`` 与 seed 不同。
所有其它字段（RRT / routing / cross attention / EMA prototype / residual_scale /
temperature / ABMIL / optimizer / scheduler / 训练协议）逐字段同源，由
:func:`build_config` 从唯一一份模板生成，不存在按组合覆写的入口。
"""
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 固定模态顺序（规格 §3）—— HE 必须是 anchor，其余按此顺序规范化
MODALITY_ORDER = ["HE", "PR", "ER", "HER2", "Ki67"]

#: 染色 → 特征目录名
DIR_MAPPING = {
    "HE": "C16_HE_features",
    "PR": "C16_PR_features",
    "ER": "C16_ER_features",
    "HER2": "C16_HER2_features",
    "Ki67": "C16_Ki67_features",
}

FEATURE_ROOT = "/home/Public/lillan/features_result/C16_features"
TRAIN_LABELS = str(REPO_ROOT / "data/C16_labels/c16_train_labels.csv")
VAL_LABELS = str(REPO_ROOT / "data/C16_labels/c16_test_labels.csv")

#: 正式结果根目录（规格 §5）
OUT_ROOT = Path("/home/Public/lillan/work_results/ours_main/C16+abmil")

#: 规格 §3 的 14 个组合。目录名逐字采用规格写法；模态列表按 MODALITY_ORDER 规范化。
COMBINATIONS = [
    ("HE+PR",                    ["HE", "PR"]),
    ("HE+ER",                    ["HE", "ER"]),
    ("HE+HER2",                  ["HE", "HER2"]),
    ("HE+Ki67",                  ["HE", "Ki67"]),
    ("HE+PR+ER",                 ["HE", "PR", "ER"]),
    ("HE+PR+HER2",               ["HE", "PR", "HER2"]),
    ("HE+PR+Ki67",               ["HE", "PR", "Ki67"]),
    ("HE+HER2+Ki67",             ["HE", "HER2", "Ki67"]),
    ("HE+ER+Ki67",               ["HE", "ER", "Ki67"]),
    ("HE+PR+HER2+ER",            ["HE", "PR", "ER", "HER2"]),
    ("HE+PR+HER2+Ki67",          ["HE", "PR", "HER2", "Ki67"]),
    ("HE+PR+ER+Ki67",            ["HE", "PR", "ER", "Ki67"]),
    ("HE+HER2+ER+Ki67",          ["HE", "ER", "HER2", "Ki67"]),
    ("HE+PR+ER+HER2+Ki67",       ["HE", "PR", "ER", "HER2", "Ki67"]),
]

#: HE-only 基线。**不属于规格 §3 的 14 个组合**，是后加的一条对照行：
#: 历史 val-as-test 的 ``he_only``（results/stage2_he_residual_cross_val_as_test）
#: 只有 3 个 seed，无法与 14 个组合放在同一张表里比。这里用**完全相同的
#: harness / 协议 / HE encoder 配置 / 同一套 10 个 seed** 重跑，唯一区别是
#: modalities=["HE"]（``models/he_aux_unified.py`` 无 aux 时 H_fused = H）。
#: 有了它，"加辅助染色到底有没有用"才有一个 seed 配对的对照。
BASELINE_COMBINATIONS = [
    ("HE", ["HE"]),
]

#: 实际要跑的全部条目 = 14 个正式组合 + 基线
ALL_COMBINATIONS = COMBINATIONS + BASELINE_COMBINATIONS


def is_baseline(name):
    return any(name == n for n, _ in BASELINE_COMBINATIONS)


#: 主 seed（规格 §4）
FIXED_SEED = 42

#: 每个组合的 seed 数
N_SEEDS = 10

#: 允许使用的物理 GPU（规格 §10）
GPU_POOL = [1, 2, 4, 6, 7]

#: 每个染色统一的 Stage-1 RRT 配置。四个辅助染色与 HE 完全相同 —— 这是本轮
#: 为满足 §13 横向可比而做的唯一一处口径决定（历史 v3 里 PR 曾单独调过
#: region 8 / epeg_k 15 / crmsa_k 5 / n_heads 8 / drop_path 0.1155，本轮弃用）。
UNIFORM_ENCODER_CFG = {
    "region_num": 4, "epeg_k": 9, "crmsa_k": 3, "n_heads": 4, "drop_path": 0.0,
}

#: Stage-2 cross 分支配置，与历史 v3 逐字段一致
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
}

#: MIL 头：ABMIL(hidden 256, dropout 0.25)，与历史 v3 一致
MIL_CFG = {"name": "abmil", "kwargs": {"hidden_dim": 256, "dropout_rate": 0.25}}

#: 训练协议，与历史 v3 逐字段一致
TRAINING = {
    "batch_size": 1,
    "num_epochs": 80,
    "learning_rate": 0.0001,
    "weight_decay": 0.00001,      # 不能写 1e-05：yaml.safe_load 会读成字符串
    "scheduler": {"type": "cosine"},
    "use_amp": False,
    "focal_loss": False,
    "label_smoothing": 0.0,
    "kd_enabled": False,
    "modality_dropout": 0.0,
    "aux_loss_weight": 0.0,
    "early_stopping": {"monitor": "val_auc", "mode": "max", "patience": 10},
    "no_validation": False,
}

DATASET_KWARGS = {
    "dataset_type": "c16",
    "input_dim": 768,
    "num_classes": 2,
    "max_patches": 2500,
    "preload": False,
    "sampling": "random",
    "no_validation": False,
    "strict_modalities": True,
}

NUM_WORKERS = 2

#: metrics.json 里必须出现的 6 个主指标（规格 §7）
PRIMARY_METRICS = ["AUC", "Accuracy", "Recall", "Precision", "F1", "Specificity"]


def modality_slug(modalities):
    return '_'.join(m.lower() for m in modalities)


def combo_dir(name):
    return OUT_ROOT / name


def seed_dir(name, seed):
    return combo_dir(name) / f"seed{seed}"


def build_config(name, modalities, seed):
    """生成一个 (组合, seed) 的完整 resolved config。

    除 ``modalities`` 与三个 seed 外，任何字段都不随组合变化。
    """
    for m in modalities:
        assert m in MODALITY_ORDER, f"unknown stain {m!r}"
    assert modalities[0] == "HE", "HE 必须是 anchor（模态列表第一位）"
    canonical = [m for m in MODALITY_ORDER if m in modalities]
    assert canonical == list(modalities), f"{modalities} 未按 MODALITY_ORDER 排序"

    sd = seed_dir(name, seed)
    cfg = {
        "data": {
            "dataset_type": "c16",
            "modalities": list(modalities),
            "dir_mapping": {m: DIR_MAPPING[m] for m in modalities},
            "train_label_file": TRAIN_LABELS,
            "val_label_file": VAL_LABELS,
            "feature_base_dir": FEATURE_ROOT,
            # 三个 seed 都显式给定 ⇒ 三个 *_source 都是 'explicit'，不存在
            # "运行时从哪里继承" 的歧义；采样种子随 model seed 变化是本轮
            # 唯一允许的 seed 耦合。
            "sample_seed": seed,
            **{k: v for k, v in DATASET_KWARGS.items() if k != "sample_seed"},
        },
        "model": {
            "stage2_type": "he_aux_unified",
            "mlp_dim": 512,
            "dropout": 0.25,
            "init_seed": seed,
            "region_num": 4,
            "n_layers": 2,
            "n_heads": 4,
            "drop_path": 0.0,
            "trans_dropout": 0.1,
            "epeg": True,
            "epeg_k": 9,
            "crmsa_k": 3,
            "cr_msa": True,
            "all_shortcut": True,
            "crmsa_heads": 8,
            "crmsa_mlp": False,
            # 五个染色全部显式写出同一份配置（§13 统一口径）
            "encoder_cfg": {m: dict(UNIFORM_ENCODER_CFG) for m in MODALITY_ORDER},
            "stage2_cfg": dict(STAGE2_CFG),
            "mil_cfg": json.loads(json.dumps(MIL_CFG)),
        },
        "training": json.loads(json.dumps(TRAINING)),
        "data_split": {"val_start": 100},
        # CUDA_VISIBLE_DEVICES 已经把进程钉在一张物理卡上，进程内一律 cuda:0
        "environment": {"device": "cuda:0", "num_workers": NUM_WORKERS, "seed": seed},
        "seeds": {"model_seed": seed, "sampling_seed": seed},
        "output": {
            "save_dir": str(sd),
            "log_dir": str(sd / "logs"),
            "img_dir": str(sd / "img"),
        },
        "protocol": {
            "evaluation_protocol": "Train / Val-as-Test",
            "train": 270,
            "val_as_test": 129,
            "note": ("官方 test 集在训练中用于 checkpoint selection / early stopping，"
                     "属既定 val-as-test 开发协议，非严格意义的独立 untouched test"),
        },
        "experiment": {
            "name": "Ours main (C16 + ABMIL)",
            "combination": name,
            "modalities": list(modalities),
            "stage2_type": "he_aux_unified",
            "fusion": "v3-style HE-anchored residual cross fusion",
            "mil": "abmil",
            "loss": "CE only",
            "role": "baseline" if is_baseline(name) else "combination",
            "note_baseline": (
                "HE-only 对照：与 14 个组合同一 harness / 协议 / HE encoder 配置 / "
                "同一套 10 个 seed，唯一区别是 modalities=['HE']，无 aux 分支时 "
                "H_fused = H。" if is_baseline(name) else None
            ),
            "seed": seed,
            "seeds_file": str(OUT_ROOT / "seeds.json"),
        },
    }
    return cfg


# ── seeds 管理（规格 §4） ────────────────────────────────────────────────

def load_or_create_seeds(path=None, n_random=9):
    """读取 ``seeds.json``；不存在则**只生成一次**并落盘。

    生成后 14 个组合全部复用同一套 10 个 seed；程序重启直接读文件，
    不会重新生成（规格 §4：程序重启后不能重新生成）。
    """
    path = Path(path or (OUT_ROOT / "seeds.json"))
    if path.exists():
        data = json.loads(path.read_text())
        seeds = [int(s) for s in data["seeds"]]
        assert seeds[0] == FIXED_SEED, f"{path} 里第一个 seed 不是 {FIXED_SEED}"
        assert len(seeds) == N_SEEDS, f"{path} 里 seed 数不是 {N_SEEDS}: {len(seeds)}"
        assert len(set(seeds)) == len(seeds), f"{path} 里有重复 seed"
        return seeds, data, False

    rng = random.SystemRandom()          # 真随机源，不依赖任何固定种子
    picked = set()
    while len(picked) < n_random:
        v = rng.randrange(1, 100000)
        if v != FIXED_SEED:
            picked.add(v)
    seeds = [FIXED_SEED] + sorted(picked)
    data = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "fixed_seed": FIXED_SEED,
        "source": "random.SystemRandom().randrange(1, 100000), 9 个非 42 的唯一整数",
        "note": ("14 个组合共用这一套 10 个 seed；文件存在即复用，从不重新生成。"
                 "第一个 seed 42 是固定的，其余 9 个是真正随机生成的整数。"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return seeds, data, True


# ── 指标口径（规格 §7） ─────────────────────────────────────────────────

def primary_metrics(val_metrics):
    """把 trainer 的 metrics dict 映射成规格要求的 6 个主指标。

    口径（C16 是二分类，0=normal，1=tumor）：

        AUC         = roc_auc_score(y_true, P(tumor))          —— 阈值无关
        Accuracy    = accuracy_score
        Recall      = sensitivity_class_1  = TP/(TP+FN) 肿瘤召回
        Precision   = precision_class_1    = TP/(TP+FP) 肿瘤精确率
        F1          = 2PR/(P+R)（肿瘤类）
        Specificity = specificity_class_0  = TN/(TN+FP) 正常类召回

    **Recall 明确取肿瘤类，不是 macro recall**（规格 §7 特别点名）。
    为保持一组指标内部自洽，Precision / F1 同样取肿瘤类。
    macro 版本在 metrics.json 里以 ``*_macro`` 另存，随时可查。
    """
    def g(k, d=0.0):
        v = val_metrics.get(k, d)
        return float(v) if v is not None else d

    p = g('precision_class_1')
    r = g('sensitivity_class_1')
    f1_tumor = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "AUC": g('auc'),
        "Accuracy": g('accuracy'),
        "Recall": r,
        "Precision": p,
        "F1": f1_tumor,
        # 正常类召回。不能用 specificity_class_0：utils/metrics.py 的 per-class 循环
        # 把第 i 类当作正类，所以 specificity_class_0 是"以 normal 为正类时的
        # TN/(TN+FP)"，化简后恰好等于肿瘤召回 sensitivity_class_1，会让
        # Specificity 与 Recall 恒等。正确键是 sensitivity_class_0
        # （== specificity_class_1，二分类下两者恒等）。
        "Specificity": g('sensitivity_class_0'),
    }


def secondary_metrics(val_metrics):
    """额外保存的口径变体，供复核 —— 不进主表。"""
    def g(k):
        v = val_metrics.get(k)
        return float(v) if v is not None else None

    return {
        "f1_macro": g('f1'),
        "precision_macro": g('precision'),
        "recall_macro": g('recall'),
        "sensitivity_macro": g('sensitivity_macro'),
        "specificity_macro": g('specificity_macro'),
        "sensitivity_class_0": g('sensitivity_class_0'),
        "specificity_class_1": g('specificity_class_1'),
        "precision_class_0": g('precision_class_0'),
        "auc_he": g('auc_he'),
        "auc_pr": g('auc_pr'),
    }


def is_completed(name, seed):
    """规格 §11：``best_model.pt`` + ``metrics.json`` + ``status == 'completed'`` 三者齐备。"""
    sd = seed_dir(name, seed)
    if not (sd / "best_model.pt").is_file():
        return False
    if not (sd / "metrics.json").is_file():
        return False
    st = sd / "status.json"
    if not st.is_file():
        return False
    try:
        return json.loads(st.read_text()).get("status") == "completed"
    except Exception:
        return False
