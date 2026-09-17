"""Ours 主实验（C17 + ABMIL）的共享定义：组合表、配置生成、指标口径、seeds 管理。

被 ``run_one_c17_seed.py``（单 run）与 ``drive_ours_main_c17_abmil.py``（队列驱动）共用。

**公平性**：14 个组合之间只允许 ``data.modalities`` 与 seed 不同。所有其它字段
（RRT encoder / routing / cross attention / EMA prototype / residual_scale /
temperature / ABMIL / optimizer / scheduler / 训练协议）逐字段同源，由
:func:`build_config` 从唯一一份模板生成，不存在按组合覆写的入口。

**历史对齐**：模型/训练参数取自
``work_results/comparative_exp/RRT+abMIL/C17/config_full.yaml``（见
``_pilot/config_audit.txt``）。数据 split 与标签定义沿用历史，未做任何改动。
"""
import json
import random
from decimal import Decimal
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── YAML 浮点写出 ─────────────────────────────────────────────────────────
# YAML 1.1 的 float 正则**不认** `2.42e-06` 这种指数写法（要求小数点两侧都有
# 数字），PyYAML 会把它读回 **str**。本项目的 lr / weight_decay / eta_min 全是
# 这个量级，一旦有人 reload config.yaml 就会拿到字符串并静默出错。
# 这里注册一个把指数形式展开成十进制的 representer，保证 dump→load 往返无损。

class _DecimalFloatDumper(yaml.SafeDumper):
    """SafeDumper + 十进制浮点表示。"""


def _represent_float(dumper, data):
    if data != data or data in (float('inf'), float('-inf')):
        return dumper.represent_scalar('tag:yaml.org,2002:float', repr(data))
    r = repr(data)
    if 'e' in r or 'E' in r:
        r = format(Decimal(r), 'f')          # 2.42e-06 → 0.00000242
        if '.' not in r:
            r += '.0'
    return dumper.represent_scalar('tag:yaml.org,2002:float', r)


_DecimalFloatDumper.add_representer(float, _represent_float)


def dump_config_yaml(cfg, path):
    """写出 config.yaml，浮点用十进制 —— dump→load 往返类型无损。"""
    with open(path, "w") as f:
        yaml.dump(cfg, f, Dumper=_DecimalFloatDumper, sort_keys=False,
                  default_flow_style=False, allow_unicode=True, width=100)

#: 固定模态顺序 —— HE 必须是 anchor，其余按此顺序规范化
MODALITY_ORDER = ["HE", "PR", "ER", "HER2", "Ki67"]

#: 染色 → 特征目录名（历史目录名不统一：ER/HER2 是单数 feature）
DIR_MAPPING = {
    "HE": "C17_HE_new_features",
    "PR": "C17_PR_new_features",
    "ER": "C17_ER_new_feature",
    "HER2": "C17_HER2_new_feature",
    "Ki67": "C17_Ki67_new_features",
}

FEATURE_ROOT = "/home/Public/lillan/features_result/C17_features"

#: 历史标签文件 —— 定义未改
LABEL_FILE = "/home/Public/lillan/data/C17_binary_label.csv"

#: 历史患者级 split：编号 < val_start 为 train，>= val_start 为 test-as-val。
#: 与 run_experiment.py 的 train_patients(000..099) / test_patients(100..199) 一致。
VAL_START = 100

#: 正式结果根目录
OUT_ROOT = Path("/home/Public/lillan/work_results/ours_main/C17+abmil")

#: 14 个组合。目录名逐字采用规格写法；模态列表按 MODALITY_ORDER 规范化。
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

#: 主 seed
FIXED_SEED = 42

#: 每个组合的 seed 数
N_SEEDS = 10

#: 允许使用的物理 GPU（规格补充条款的显存划分）
#: 0 / 4 / 5 = 32G，6 / 7 = 16G。注意本轮任务书正文 §11 写的是 "1, 2, 4, 6, 7"，
#: 与补充条款的 "0/4/5 为 32G、6/7 为 16G" 冲突（本机实测 1=32G、2=16G，
#: 与补充条款的容量描述不符）。以**补充条款**为准：它给出了每张卡的显存大小
#: 与据此的并发规则，语义更完整。
GPU_POOL = [0, 4, 5, 6, 7]

#: 32G 卡集合（Quad / Quintuple 只允许落在这些卡上）
BIG_CARDS = [0, 4, 5]

# ═══════════════════════════════════════════════════════════════════════════
# 统一配置（14 个组合共用；唯一可变项是 modalities 与 seed）
# ═══════════════════════════════════════════════════════════════════════════

#: Stage-1 RRT encoder 配置 —— 来自 config_full.yaml 的 C17 HE 最优 RRT 参数。
#: 四个辅助染色与 HE 完全相同；**权重完全独立**（HEAuxUnifiedModel 里每个染色
#: 有自己的 `patch_to_emb` 与 `RRTEncoder`，只是超参同源）。
UNIFORM_ENCODER_CFG = {
    "region_num": 4,
    "epeg_k": 15,
    "crmsa_k": 3,
    "n_heads": 4,
    "drop_path": 0.25,
}

#: Stage-1 结构参数（config_full.yaml）
N_LAYERS = 1
MLP_DIM = 256
TRANS_DROPOUT = 0.1

#: 投影层 dropout（config_full.yaml 的 `dropout: 0.05`）
MODEL_DROPOUT = 0.05

#: Stage-2 v3-style HE-anchored residual cross fusion —— 与历史 v3 逐字段一致
STAGE2_CFG = {
    "region_num": 4, "crmsa_heads": 8, "crmsa_k": 3, "drop_out": 0.1,
    "drop_path": 0.0, "epeg": False, "epeg_k": 15, "crmsa_mlp": False,
    "ffn": False, "qkv_bias": False,
    "temperature": 0.2, "residual_scale": 0.1,
    "disable_cross": False, "prototype_momentum": 0.99,
}

#: MIL 头：ABMIL(hidden 64, dropout 0.05) —— 取自 config_full.yaml 的
#: `abmil_hidden_dim: 64` 与 `dropout: 0.05`（历史 RRT_ABMIL 把 dropout 同时
#: 传给投影层与 MIL 的 attention dropout）。
MIL_CFG = {"name": "abmil", "kwargs": {"hidden_dim": 64, "dropout_rate": 0.05}}

#: 训练协议 —— 取自 config_full.yaml 的 training 段
TRAINING = {
    "batch_size": 1,
    "num_epochs": 30,
    "learning_rate": 0.000242,
    "weight_decay": 0.00000759,     # 不能写 7.59e-06：yaml.safe_load 会读成字符串
    # 优化器：显式 AdamW，对齐历史 run_experiment.py:406
    # `optim.AdamW(model.parameters(), lr, weight_decay)`（其余取 torch 默认）
    "optimizer": {"type": "adamw"},
    # scheduler：对齐历史 run_experiment.py:407-409
    # `CosineAnnealingLR(T_max=epochs, eta_min=lr*0.01)`
    "scheduler": {"type": "cosine", "eta_min": 0.000242 * 0.01},
    "use_amp": False,
    "focal_loss": False,
    "label_smoothing": 0.0,
    "kd_enabled": False,
    "modality_dropout": 0.0,
    "aux_loss_weight": 0.0,
    # min_epochs 语义对齐历史 EarlyStopping(stop_epoch=10)：
    # 0-based epoch >= 10 才允许早停
    "early_stopping": {"monitor": "val_auc", "mode": "max",
                       "patience": 15, "min_epochs": 10},
    "no_validation": False,
}

#: 数据集参数（config_full.yaml: max_patches 5000, input_dim 768, num_classes 2）
MAX_PATCHES = 5000

DATASET_KWARGS = {
    "dataset_type": "c17",
    "input_dim": 768,
    "num_classes": 2,
    "max_patches": MAX_PATCHES,
    "preload": False,
    "sampling": "random",
    "no_validation": False,
    "strict_modalities": True,
    "require_same_patch_count": True,
}

NUM_WORKERS = 2

#: metrics.json 里必须出现的 6 个主指标
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
            "dataset_type": "c17",
            "modalities": list(modalities),
            "dir_mapping": {m: DIR_MAPPING[m] for m in modalities},
            "label_file": LABEL_FILE,
            "feature_base_dir": FEATURE_ROOT,
            "val_start": VAL_START,
            # 采样种子随 model seed 变化，是本轮唯一允许的 seed 耦合。
            "sample_seed": seed,
            **{k: v for k, v in DATASET_KWARGS.items() if k != "sample_seed"},
        },
        "model": {
            "stage2_type": "he_aux_unified",
            "mlp_dim": MLP_DIM,
            "dropout": MODEL_DROPOUT,
            "init_seed": seed,
            # 模型级结构（config_full.yaml）
            "region_num": UNIFORM_ENCODER_CFG["region_num"],
            "n_layers": N_LAYERS,
            "n_heads": UNIFORM_ENCODER_CFG["n_heads"],
            "drop_path": UNIFORM_ENCODER_CFG["drop_path"],
            "trans_dropout": TRANS_DROPOUT,
            "epeg": True,
            "epeg_k": UNIFORM_ENCODER_CFG["epeg_k"],
            "crmsa_k": UNIFORM_ENCODER_CFG["crmsa_k"],
            "cr_msa": True,
            "all_shortcut": False,
            "crmsa_heads": 8,
            "crmsa_mlp": False,
            # 五个染色全部显式写出同一份配置（统一口径）
            "encoder_cfg": {m: dict(UNIFORM_ENCODER_CFG) for m in MODALITY_ORDER},
            "stage2_cfg": dict(STAGE2_CFG),
            "mil_cfg": json.loads(json.dumps(MIL_CFG)),
        },
        "training": json.loads(json.dumps(TRAINING)),
        "data_split": {"val_start": VAL_START},
        # CUDA_VISIBLE_DEVICES 已把进程钉在一张物理卡上，进程内一律 cuda:0
        "environment": {"device": "cuda:0", "num_workers": NUM_WORKERS, "seed": seed},
        "seeds": {"model_seed": seed, "sampling_seed": seed},
        "output": {
            "save_dir": str(sd),
            "log_dir": str(sd / "logs"),
            "img_dir": str(sd / "img"),
        },
        "protocol": {
            "evaluation_protocol": "Train / Test-as-Val",
            "split": "patient-level by index: patient_000..099 train / patient_100..199 test",
            "note": ("官方 test 集在训练中用于 checkpoint selection / early stopping，"
                     "与历史 C17 RRT_ABMIL baseline 的协议一致（best test AUC + "
                     "EarlyStopping(patience=15, min_epochs=10)），属既定 val-as-test "
                     "开发协议，非严格意义的独立 untouched test。"),
        },
        "experiment": {
            "name": "Ours main (C17 + ABMIL)",
            "combination": name,
            "modalities": list(modalities),
            "stage2_type": "he_aux_unified",
            "fusion": "v3-style HE-anchored residual cross fusion",
            "mil": "abmil",
            "loss": "CE only",
            "seed": seed,
            "seeds_file": str(OUT_ROOT / "seeds.json"),
        },
    }
    return cfg


# ── seeds 管理 ───────────────────────────────────────────────────────────

def load_or_create_seeds(path=None, n_random=9):
    """读取 ``seeds.json``；不存在则**只生成一次**并落盘。

    生成后 14 个组合全部复用同一套 10 个 seed；程序重启直接读文件，
    不会重新生成。
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


# ── 指标口径 ─────────────────────────────────────────────────────────────

def primary_metrics(val_metrics):
    """把 trainer 的 metrics dict 映射成规格要求的 6 个主指标。

    C17 是二分类（0/1），口径与 C16 主实验一致：

        AUC         = roc_auc_score(y_true, P(class 1))   —— 阈值无关
        Accuracy    = accuracy_score
        Recall      = sensitivity_class_1  = TP/(TP+FN)  正类召回
        Precision   = precision_class_1    = TP/(TP+FP)  正类精确率
        F1          = 2PR/(P+R)（正类）
        Specificity = sensitivity_class_0  = TN/(TN+FP)  负类召回

    **不能**用 ``specificity_class_0``：``utils/metrics.py`` 的 per-class 循环把
    第 i 类当作正类，所以 ``specificity_class_0`` 化简后恰好等于正类召回，
    会让 Specificity 与 Recall 恒等。正确的键是 ``sensitivity_class_0``。

    macro 版本在 metrics.json 里以 ``*_macro`` 另存。
    """
    def g(k, d=0.0):
        v = val_metrics.get(k, d)
        return float(v) if v is not None else d

    p = g('precision_class_1')
    r = g('sensitivity_class_1')
    f1_pos = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "AUC": g('auc'),
        "Accuracy": g('accuracy'),
        "Recall": r,
        "Precision": p,
        "F1": f1_pos,
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
    }


def is_completed(name, seed):
    """断点续跑判据（规格 §12，逐条照做）。

    只有同时满足才认为完成::

        best_model.pt exists
        metrics.json exists
        metrics.json["status"] == "completed"

    任何一条不成立 → 视为 incomplete，重新运行。注意判据取自 **metrics.json**
    的 status（不是 status.json），这是规格明确写死的。
    """
    sd = seed_dir(name, seed)
    if not (sd / "best_model.pt").is_file():
        return False
    mp = sd / "metrics.json"
    if not mp.is_file():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "completed"
    except Exception:                                          # noqa: BLE001
        return False


# ── HE-only 对照臂 ─────────────────────────────────────────────────────────
# 主实验 14 个组合全部是 HE+X，没有 HE-only，因此无法回答"辅助染色相对 HE
# 单独使用是否有增益"。这一臂用同一个 build_config("HE", ["HE"], seed) 补上，
# 产物落在与 14 个组合目录并列的 he/ 下。
#
# 放在本模块（而不是 drive_c17_he_baseline.py）是为了让汇总侧也能读到它，
# 同时避免汇总脚本反向 import 执行器造成循环依赖。
def he_combo_dir():
    """HE-only 臂的目录。**按调用时取 OUT_ROOT**，与 combo_dir 一致，
    这样 --out-root 覆写后两者仍然同步。"""
    return OUT_ROOT / "he"


def he_seed_dir(seed):
    return he_combo_dir() / f"seed{seed}"


def he_is_completed(seed):
    """与 is_completed 同一判据，只是路径换成 he/。"""
    sd = he_seed_dir(seed)
    if not (sd / "best_model.pt").is_file():
        return False
    mp = sd / "metrics.json"
    if not mp.is_file():
        return False
    try:
        return json.loads(mp.read_text()).get("status") == "completed"
    except Exception:                                          # noqa: BLE001
        return False
