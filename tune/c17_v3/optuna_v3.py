#!/usr/bin/env python
"""C17 + ABMIL + v3 —— 快速 Optuna 超参搜索（**不改 v3 模型结构**）。

搜索对象是 5 个"有潜力"的组合，每个组合一个独立 study：

    HE+Ki67        search seed 42    -> GPU2
    HE+PR          search seed 42    -> GPU3
    HE+HER2+Ki67   search seed 42    -> GPU4
    HE+PR+ER       search seed 42    -> GPU5
    HE+PR+HER2+ER  search seed 42    -> GPU6
                                       GPU7 = hold-out validation worker

五个模态全上的 HE+PR+ER+HER2+Ki67 不搜。

**搜索 seed 统一为 42**（用户 2026-09-18 裁定）。不同组合是彼此独立的
study，共用同一个 seed 不构成冲突；统一反而消掉了"组合间随机初始化差异"
这一项噪声。**禁止**按 val AUC 给各组合挑"最好 seed"当搜索 seed。

**hold-out validation seeds = 3611 / 6430 / 39998**，一律**禁止参与搜索**
（脚本里有硬断言）。搜索结束后由 GPU7 用 best params 在这 3 个 seed 上跑，
与这三个 seed 已有的原始正式 baseline 做 paired comparison。

统一训练协议（搜索 / hold-out 验证 / 历史正式 run 三者同源）
------------------------------------------------------------
    epochs=30  patience=15  min_epochs=10  T_max=30  eta_min=lr*0.01

``T_max`` 跟随 ``num_epochs``（``train.py:1061``），所以 epochs=30 自动得到
T_max=30。``eta_min`` 是 ``learning_rate`` 的**派生量**：``build_config`` 里
它写死成 ``0.000242*0.01``（``ours_main_c17_abmil.py:160``），不跟着 trial 的
lr 走，所以 objective 里显式同步，训练后再从 ``trainer.eta_min_actual``
读回来断言。

"12 条有效 trial" 与 startup 采样（2026-10-15 修复）
----------------------------------------------------
有效 trial = **COMPLETE / PRUNED 且配置在 study 内首次出现**。FAIL（含被就地判
FAIL 的孤儿 RUNNING）、重复配置，**都不计入** 12 条有效搜索预算。判重按 6 个参数
的逐位值（float 用 repr），见 :func:`valid_trials` / :func:`n_valid`。

startup 阶段（``COMPLETE+PRUNED < n_startup_trials``）原先交给 TPESampler 内部的
RandomSampler，但 ``LazyRandomState.__init__`` 每次构造都会 ``rng.seed(seed)``，
而 driver 是 ``--trials-per-process 1``（一条 trial 一个进程）——**每个进程的第一
次抽样都从同一个 RNG 状态出发，永远抽到同一个点**。2026-09-18 那批 study 里
T1 == T2 逐位相同、5 个组合全中，就是这个原因（见
``_interrupted_backup/README.md``）。现在改为：startup 阶段的配置由本脚本用
``SAMPLER_SEED`` 确定性抽出、显式 ``enqueue_trial`` 入队，抽的时候跳过所有已用过的
配置 —— 于是**同一 study 状态下任何进程都得到同一个"下一点"，且各点互不重复**；
TPE 仍按 ``n_startup_trials`` 从第 5 条 trial（number=4）正常接手。

**同一个病在 TPE 阶段还有第二处，2026-10-15 一并修掉**：TPE 的候选点来自
``TPESampler`` 自己的 ``self._rng``，而它也是每进程按常数播种的。于是只要
argmax 稳定（``n_below`` 在 n<=10 时恒为 1，below 组就只有那一条），
``mpe_below`` 逐位不变 -> 候选点云每轮重放 -> **提案锁死成不动点**。那批里
HE+Ki67 的 #8..#23 是 16 条连续 FAIL、每条都与 #7 逐位相同，链死在 8/12
（见 ``_rng_bug_backup/README.md``）。现在种子改为 **``SAMPLER_SEED + 已有
记录条数``**，随 trial 号推进 —— 机制、以及为什么"一进程串行 12 条"这个
替代方案不能替代它，都写在 :func:`sampler_seed_for` 里。

口径（这是整个脚本的地基，改动前先读这一段）
------------------------------------------------
*   **数据划分一个字都不动**。每条 trial 的 config 由
    ``ours_main_c17_abmil.build_config()`` 生成 —— 和 140 次正式 run 用的
    是同一个函数、同一份 ``C17_features`` 原路径（不复制特征、不复制数据集）、
    同一个 patient-level 划分（patient_000..099 train / patient_100..199
    test-as-val）。**不重新分折、不重采样。**
*   **目标函数 = val_as_test AUC**，取自 trainer 训练结束后重新加载
    best checkpoint 再评估一次的那个数（``trainer.best_val_auc``），
    也就是正式 run 里写进 ``metrics.json`` 的 ``AUC``。
*   **一个 trial 一个 seed**：组合的 best seed 固定不变，trial 之间只动 6 个
    超参，谁都不许改 seed、不许改数据顺序。

6 个超参与其真实落点（§2 要求"参数真的进到模型里"）
------------------------------------------------------
    learning_rate        -> training.learning_rate            -> AdamW param_group['lr']
    weight_decay         -> training.weight_decay             -> AdamW param_group['weight_decay']
    dropout              -> model.dropout                     -> HEAuxUnifiedModel.dropout / .dp.p
    residual_scale       -> model.stage2_cfg.residual_scale   -> cross_branch.residual_scale
    tau                  -> model.stage2_cfg.temperature      -> cross_branch.tau
    prototype_momentum   -> model.stage2_cfg.prototype_momentum -> cross_branch.beta

    ``stage2_cfg`` 的三项在 ``HEAuxUnifiedModel._make_cross_branch()`` 里被读出来
    传给每一个 ``HEResidualCrossCRMSAv3`` 分支。**光写进 config 不算数**，
    所以每条 trial 训练完都会调 :func:`verify_reached` 从**建好的模型对象上**
    把 6 个值读回来逐一比对，对不上直接抛异常。

不改的东西
------------
学习率调度器 / 优化器类型 / region_num / n_heads / n_layers / epeg_k /
crmsa_k / hidden_dim / RRT 结构 / ABMIL 结构 / 融合结构 —— 一律不搜。
调度器按正式配置（cosine + eta_min）；优化器只有**一个** param group
（``train.py:1007-1021`` 的 else 分支：``he_aux_unified`` 既没有 ``logit_mixer``
也没有 ``use_correction_only``，全部可训练参数进同一组、同一个 lr），
所以本轮只搜一个 ``learning_rate``，不搜 lr_stage1/lr_stage2。

运行方式（**一个进程跑一条 trial**，进程退出即释放显存）
------------------------------------------------------------
    python tune/c17_v3/optuna_v3.py --combo HE+Ki67 --gpu 2 --smoke
    python tune/c17_v3/optuna_v3.py --combo HE+Ki67 --gpu 2
    python tune/c17_v3/optuna_v3.py --combo HE+Ki67 --gpu 2 --dry-run

正式跑 60 条用 ``tune/c17_v3/run_optuna.sh``（GPU 映射写死在里面：
搜索 GPU2/3/4/5/6，验证 GPU7）。hold-out 验证另见
``tune/c17_v3/validate_holdout.py``。
"""
import argparse
import csv
import gc
import json
import logging
import os
import random
import shutil
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np                                            # noqa: E402
import optuna                                                 # noqa: E402
import torch                                                  # noqa: E402
import yaml                                                   # noqa: E402

import train as T                                             # noqa: E402
import ours_main_c17_abmil as O                               # noqa: E402
from run_one_c17_seed import seed_everything                  # noqa: E402
from run_one_c17_seed import HistoryTrainer                   # noqa: E402

#: 本脚本自己的根目录 = ``<repo>/tune/c17_v3``。
#: 所有新增产物都落在这棵子树下 —— 它是 ``tune/`` 的真子集，所以 §6「都放在
#: tune/ 下面」成立；而清理范围收窄到这一棵，是因为 ``tune/`` 根下还有
#: 2026-08-13~16 的旧 Optuna 工作（``tune.db`` / ``tune_pr.py`` /
#: ``tune_stage2.py`` / ``best_results/`` / ``folds/``），按 ``tune/`` 全域清理
#: 会把它们一起删掉。清理**永不越出 TUNE_ROOT**。
TUNE_ROOT = Path(__file__).resolve().parent
TMP_ROOT = TUNE_ROOT / "tmp"

#: 搜索 seed —— **5 个组合统一用 42**（用户 2026-09-18 裁定）。
#: 不同组合各是一个独立 study，共用同一个 seed 不构成冲突；统一反而能消掉
#: "组合间随机初始化差异"这一项噪声。**禁止**按 val AUC 给各组合挑"最好 seed"。
SEARCH_SEED = 42

#: hold-out validation seeds —— **严禁参与搜索**。搜索结束后由 GPU7 的
#: 验证 worker 用 best params 在这 3 个 seed 上跑，与已有的原始正式 baseline
#: 做 paired comparison。
HOLDOUT_SEEDS = (3611, 6430, 39998)

#: 组合 -> (目录名, 搜索 seed)。搜索 seed 一律 SEARCH_SEED。
COMBOS = {
    "HE+Ki67":        ("HE_Ki67",        SEARCH_SEED),
    "HE+PR":          ("HE_PR",          SEARCH_SEED),
    "HE+HER2+Ki67":   ("HE_HER2_Ki67",   SEARCH_SEED),
    "HE+PR+ER":       ("HE_PR_ER",       SEARCH_SEED),
    "HE+PR+HER2+ER":  ("HE_PR_HER2_ER",  SEARCH_SEED),
}

N_TRIALS = 12            # §4：每组合 12 条 trial
#: 统一训练协议（用户 2026-09-18 定稿）—— **搜索与 hold-out 验证、以及历史
#: 正式 run 三者同源**：epochs=30 / patience=15 / min_epochs=10 / T_max=30 /
#: eta_min=lr*0.01。T_max 跟随 num_epochs（``train.py:1061``），所以 epochs=30
#: 自动给出 T_max=30。搜索期不再另设预算，与正式 run 逐字段一致。
SEARCH_EPOCHS = 30       # = 正式 run 的 num_epochs
SEARCH_PATIENCE = 15     # = 正式 run 的 early_stopping.patience
SEARCH_MIN_EPOCHS = 10   # = 正式 run 的 early_stopping.min_epochs
ETA_MIN_RATIO = 0.01     # eta_min = learning_rate * 0.01
SAMPLER_SEED = 2026      # §4

#: startup 阶段的条数 —— **必须与下面 TPESampler(n_startup_trials=...) 一致**。
#: 语义见文件头"12 条有效 trial 与 startup 采样"一节。
N_STARTUP = 4

#: 该组合已跑满并收尾的退出码（见文件末尾说明）
EXIT_FINISHED = 10

#: 参数落点核对的容差。float32 buffer 用松的那档，纯 Python float 用严格档 ——
#: 搜索空间里相邻取值至少差 0.025，1e-6 的容差不可能把"取错值"放过去。
TOL_F32 = 1e-6
TOL_STRICT = 1e-12

#: §3 的搜索空间（按用户口径修正过两处：dropout 补 0.05/0.10；
#: learning_rate 放宽到 5e-5 ~ 3e-4，好让当前实际值 2.42e-4 落在空间内）。
#: 元组形式：(类型, ...) —— float 为 (lo, hi, log)，cat 为选项列表。
SPACE = {
    "learning_rate":      ("float", 5e-5, 3e-4, True),
    "weight_decay":       ("float", 1e-6, 1e-4, True),
    "dropout":            ("cat", [0.05, 0.10, 0.15, 0.25, 0.35]),
    "residual_scale":     ("cat", [0.05, 0.075, 0.10, 0.15, 0.20]),
    "tau":                ("cat", [0.10, 0.15, 0.20, 0.30]),
    "prototype_momentum": ("cat", [0.95, 0.98, 0.99, 0.995]),
}

#: 6 个超参 -> 在 config 里的**真实路径**。写进 config 用这套路径，
#: 从模型读回来用 :func:`verify_reached`，两边必须对上。
APPLY = {
    "learning_rate":      ("training", "learning_rate"),
    "weight_decay":       ("training", "weight_decay"),
    "dropout":            ("model", "dropout"),
    "residual_scale":     ("model", "stage2_cfg", "residual_scale"),
    "tau":                ("model", "stage2_cfg", "temperature"),
    "prototype_momentum": ("model", "stage2_cfg", "prototype_momentum"),
}


def derive_eta_min(learning_rate):
    """``eta_min = learning_rate * 0.01``（用户 2026-09-18 协议）。

    **必须显式同步**：``build_config`` 里 eta_min 是写死的
    ``0.000242 * 0.01``（``ours_main_c17_abmil.py:160``），不会跟着
    trial 的 lr 走。不同步的话，lr=5e-5 时 eta_min 会停在 2.42e-06
    ——相当于 lr 的 4.8%，余弦退火几乎不降到底，等于偷换了 schedule。
    它不是独立的搜索维度，是 learning_rate 的派生量，所以不进 SPACE。
    """
    return float(learning_rate) * ETA_MIN_RATIO


# ── config 小工具 ────────────────────────────────────────────────────────

def modalities_of(combo):
    """组合名 -> 规范顺序的模态列表（与 ``build_config`` 的断言同源）。"""
    mods = [m for m in O.MODALITY_ORDER if m in combo.split("+")]
    assert mods and mods[0] == "HE", f"组合名无法解析: {combo!r}"
    return mods


def get_nested(cfg, path):
    d = cfg
    for k in path:
        d = d[k]
    return d


def set_nested(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def read_defaults(combo, seed):
    """§5：Trial 0 = **当前 v3 实际配置**，值必须从实际 config 读，不许硬编码。

    优先读这个组合正式跑过的 ``config.yaml``（就是那 10 次 run 用的那份）；
    读不到才退回 ``build_config()``。两者若在 6 个参数上有出入，逐条打出来，
    并以**磁盘上那份**为准 —— 因为那是真正跑出结果的配置。
    """
    built = O.build_config(combo, modalities_of(combo), seed)
    disk = O.seed_dir(combo, seed) / "config.yaml"
    if disk.is_file():
        cfg = yaml.safe_load(disk.read_text())
        src = str(disk)
        for name, path in APPLY.items():
            a, b = get_nested(cfg, path), get_nested(built, path)
            if a != b:
                print(f"[warn] {name}: 磁盘 config={a!r} 与 build_config={b!r} 不一致，"
                      f"以磁盘为准")
    else:
        cfg = built
        src = f"O.build_config({combo!r}, {modalities_of(combo)}, {seed}) [磁盘 config.yaml 不存在]"
    return {name: get_nested(cfg, path) for name, path in APPLY.items()}, src


def assert_in_space(params, where):
    """把"默认值不在搜索空间里"这种问题变成**当场报错**，而不是安静地跑一条
    空间外的 trial。optuna 对被 enqueue / 被 tell 的参数不做分布校验，
    所以这个检查必须自己做。"""
    for name, v in params.items():
        spec = SPACE[name]
        if spec[0] == "float":
            lo, hi = spec[1], spec[2]
            assert lo <= float(v) <= hi, f"{where}: {name}={v} 不在 [{lo}, {hi}] 内"
        else:
            assert v in spec[1], f"{where}: {name}={v} 不在 {spec[1]} 内"


# ── §2 参数确实进了模型 ──────────────────────────────────────────────────

def verify_reached(model, optimizer, params):
    """从**建好的模型对象**上把 6 个值读回来，逐一比对。

    只写进 config 不能证明什么 —— ``stage2_cfg`` 是经
    ``HEAuxUnifiedModel.__init__`` → ``_make_cross_branch()`` 才落到每个
    cross branch 上的，中间任何一环接错，config 里照样写着对的值。
    所以这里读的是**模型属性**和 **optimizer param_group**，不是 config。
    """
    got = {
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "dropout": float(model.dropout),
    }
    assert len(optimizer.param_groups) == 1, (
        f"优化器有 {len(optimizer.param_groups)} 个 param group，"
        f"本轮的单一 learning_rate 口径不再成立")
    assert abs(got["dropout"] - float(params["dropout"])) < 1e-12, \
        f"dropout 未生效: 模型 {got['dropout']} vs trial {params['dropout']}"
    if hasattr(model, "dp") and isinstance(model.dp, torch.nn.Dropout):
        assert abs(float(model.dp.p) - float(params["dropout"])) < 1e-12, \
            f"nn.Dropout.p 未跟随 dropout: {model.dp.p} vs {params['dropout']}"

    branches = getattr(model, "cross_branches", None)
    assert branches, "模型没有 cross_branches —— 辅助染色分支没建起来"
    per_branch = {}
    for stain, br in branches.items():
        rs = float(br.residual_scale)
        tau = float(br.tau)
        beta = float(br.beta)
        # 容差分两档：``residual_scale`` 在基类里是
        # ``register_buffer('residual_scale', torch.tensor(float(x)))``，
        # 落成 **float32** —— 0.1 读回来是 0.10000000149011612。
        # 拿 1e-12 比会把"参数完全正确"误判成"未生效"（踩过一次：
        # 训练跑了 17 个 epoch 拿到 0.9530，却在这条 assert 上被判 FAILED）。
        # tau / beta 走的是 ``self.tau = float(tau)``，纯 Python double，严格比。
        assert abs(rs - float(params["residual_scale"])) < TOL_F32, \
            f"[{stain}] residual_scale 未生效: {rs} vs {params['residual_scale']}"
        assert abs(tau - float(params["tau"])) < TOL_STRICT, \
            f"[{stain}] tau 未生效: {tau} vs {params['tau']}"
        assert abs(beta - float(params["prototype_momentum"])) < TOL_STRICT, \
            f"[{stain}] prototype_momentum 未生效: {beta} vs {params['prototype_momentum']}"
        per_branch[stain] = {"residual_scale": rs, "tau": tau,
                             "beta": beta}
    got["cross_branches"] = per_branch
    return got


# ── 采样 / 目标函数 ──────────────────────────────────────────────────────

def sample_params(trial):
    p = {}
    for name, spec in SPACE.items():
        if spec[0] == "float":
            p[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3])
        else:
            p[name] = trial.suggest_categorical(name, spec[1])
    return p


class DuplicateConfig(Exception):
    """trial 抽到的配置与 study 里已有的有效配置逐位相同。

    不训练、直接标 FAIL —— 零 GPU 成本，且按 :func:`n_valid` 的口径
    不占 12 条有效搜索预算。
    """


def trial_params_of(t):
    """trial 的配置。

    已 ``enqueue`` 但还没跑的 trial，``params`` 是**空的**，真正的值在
    ``system_attrs["fixed_params"]``（optuna 自己判重也是这么读的，
    ``study.py:1099`` 的 ``_should_skip_enqueue``）。
    """
    return t.system_attrs.get("fixed_params", t.params)


def config_key(params):
    """配置唯一键。

    float 用 ``repr``：Python 3 的 repr 是往返精确的最短表示，两个 float
    相等 <=> repr 相等，跨进程、跨 sqlite 往返都成立。
    """
    return tuple((k, repr(params[k])) for k in sorted(SPACE))


def valid_trials(study):
    """有效 trial：COMPLETE / PRUNED，且配置在 study 内**首次**出现。

    这是"12 条有效搜索预算"的唯一实现 —— FAIL（含被就地判 FAIL 的孤儿
    RUNNING）、重复配置，一律不计入。
    """
    seen, out = set(), []
    for t in sorted(study.trials, key=lambda x: x.number):
        if t.state not in (optuna.trial.TrialState.COMPLETE,
                           optuna.trial.TrialState.PRUNED):
            continue
        p = trial_params_of(t)
        if not p:
            continue
        k = config_key(p)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def n_valid(study):
    return len(valid_trials(study))


def existing_trial_count(study_db):
    """study.db 里已有的 trial 记录条数；库不存在/表还没建则 0。

    只读打开，绝不创建文件 —— 它要在 ``create_study`` **之前**调用，用来
    给采样器定种子（见 :func:`sampler_seed_for`）。

    为什么不直接 ``len(study.trials)``：那时 study 还没建起来。而这里读到的
    条数在 TPE 阶段**恰好等于下一个 trial 的 number**（trial number 就是按
    记录的先后 0,1,2,... 排的），所以两种写法等价，但这一种不必先有 study。
    """
    if not study_db.is_file():
        return 0
    con = sqlite3.connect(f"file:{study_db}?mode=ro", uri=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    except sqlite3.Error:
        # 空库 / 表还没建 —— 当作 0
        return 0
    finally:
        con.close()


def sampler_seed_for(n_records):
    """本轮进程该用的采样器种子 —— **随 trial 号推进，不是个常数**。

    ================================================================
    2026-10-15 第二个缺陷（第一个见文件头 "startup 采样" 一节）
    ================================================================
    症状：某条 trial 成了最优之后，后续每一条 TPE 提案都与它**逐位相同**，
    一直重复到把 driver 的轮次上限烧光。2026-10-15 那批里 HE+Ki67 的
    #8..#23 是 16 条连续 FAIL，每一条都与 #7 逐位相同，链死在 8/12。

    机制（``optuna/samplers/_tpe/sampler.py:553``）::

        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above)
        ret = TPESampler._compare(samples_below, acq_func_vals)

    提案点是**从 below 混合分布里抽出的候选**，来自 ``self._rng``；而
    ``self._rng`` 是 ``TPESampler(seed=SAMPLER_SEED)`` 在构造时播种的，
    driver 又是 ``--trials-per-process 1``（一条 trial 一个进程）——
    **每个新进程的 RNG 都从同一个状态重来**。同时
    ``n_below = min(ceil(0.1n), 25)``，n <= 10 时 **n_below = 1**，below 组
    只有 argmax 那一条：只要 argmax 不变，``mpe_below`` 就逐位不变 ->
    **候选点云每轮完全重放** -> ``np.argmax`` 取到同一个下标 -> 抽出同一个
    配置。``mpe_above`` 会随 history 变，所以下标**偶尔**会翻（这就是前几条
    TPE 提案还能各不相同的原因）；一旦它稳定，就锁死成不动点。

    这跟 startup 那个缺陷是**同一个病**：都是"每进程重新播种导致采样重放"。
    那边靠 enqueue 绕开了采样器，这边是 TPE 阶段，绕不开 —— 所以改成
    **按 trial 号播种**：种子 = ``SAMPLER_SEED + n_records``。

    这样得到三条性质：
      * **不重放**：每一轮的种子都不同 -> 候选点云不同 -> 即使 below 组连续
        多轮不变，提案仍然会变。（这是关键的回归点，dry-run 的 ④d 专门验它。）
      * **可复现**：种子只是 study 状态的函数（已有记录条数），同一状态下
        任何进程、任何时刻都得到同一个种子 —— 这正是要保住的确定性。
      * **抗重启**：进程被杀后重启，条数只增不减，种子继续往前走，不会退回
        到已经抽过的点上（"一进程串行 12 条"那种方案做不到这点：进程一崩，
        重起的进程又从 seed 2026 开始，而 argmax 通常纹丝不动 -> 照样锁死）。
    """
    return SAMPLER_SEED + int(n_records)


def open_study(study_db, slug, study_name=None):
    """建/载入 study（含 sampler 与 pruner）—— **main() 与 dry-run 共用这一段**。

    dry-run 模拟器必须调这个函数而不是自己复制一份 ``create_study``：种子这条
    逻辑正是被修的那个缺陷本身，验一份复制品等于没验。

    返回 ``(study, {"n_seed", "s_seed"})``。
    """
    n_seed = existing_trial_count(study_db)
    s_seed = sampler_seed_for(n_seed)
    study = optuna.create_study(
        study_name=study_name or f"c17_v3_{slug}",
        storage=f"sqlite:///{study_db}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=s_seed,
                                           n_startup_trials=N_STARTUP,
                                           multivariate=True),
        # pruner 的 n_startup_trials 是它自己的概念（前几条不做中位数剪枝），
        # 与 sampler 的 N_STARTUP 无关；本轮恰好都取 4。
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4,
                                           n_warmup_steps=8,
                                           interval_steps=2),
        load_if_exists=True,
    )
    return study, {"n_seed": n_seed, "s_seed": s_seed}


def used_keys(study):
    """已被"占用"的配置键集合 = 有效 trial 的 + 已入队待跑的。

    必须含 WAITING：否则"入队之后、跑起来之前进程被杀"会让下一轮又抽到
    同一个点、重复入队，白烧一条 trial。
    """
    keys = {config_key(trial_params_of(t)) for t in valid_trials(study)}
    for t in study.trials:
        if t.state == optuna.trial.TrialState.WAITING:
            p = trial_params_of(t)
            if p:
                keys.add(config_key(p))
    return keys


def draw_from_space(rs):
    """用给定 RandomState 从 SPACE 抽一个配置。

    分布与 :func:`sample_params` 对齐：float 走对数/线性均匀，cat 走等概率。
    """
    p = {}
    for name, spec in SPACE.items():
        if spec[0] == "float":
            lo, hi, log = spec[1], spec[2], spec[3]
            p[name] = float(np.exp(rs.uniform(np.log(lo), np.log(hi)))
                            if log else rs.uniform(lo, hi))
        else:
            opts = spec[1]
            p[name] = opts[int(rs.randint(len(opts)))]
    return p


def duplicate_of(study, params):
    """``params`` 与 study 里已有有效配置逐位相同时，返回那条 trial 的
    number，否则 None。判重口径与 :func:`n_valid` 完全一致。

    抽出来是因为 dry-run 模拟器要用**同一个**判重函数 —— 验的必须是正式跑
    的那段逻辑本身。
    """
    k = config_key(params)
    for t in valid_trials(study):
        if config_key(trial_params_of(t)) == k:
            return t.number
    return None


def next_startup_config(used):
    """startup 阶段的下一个配置：**确定性** 且 **不重复**。

    确定性：RNG 每次从 ``SAMPLER_SEED`` 重放，输入只有 ``used``（由 study
    状态导出）。同一个 study 状态在任何进程、任何时刻都得到同一个点 ——
    这正是被修掉的缺陷的反面（那边是"换进程就重来，于是永远同一点"）。
    不重复：从头扫，跳过 ``used`` 里已有的配置。
    """
    rs = np.random.RandomState(SAMPLER_SEED)
    for _ in range(100000):
        cfg = draw_from_space(rs)
        if config_key(cfg) not in used:
            return cfg
    raise RuntimeError("startup 采样 10 万次仍抽不到未用过的配置，SPACE 异常")


def prepare_study(study, defaults, max_trials, log):
    """打开 study 之后的固定动作：孤儿 RUNNING 判 FAIL + startup 阶段补队。

    **main() 与 dry-run 模拟器共用这一段** —— dry-run 验的就是正式跑的那段
    逻辑本身，不是它的复制品。

    返回 ``{"n_valid", "n_stale", "n_waiting", "enqueued"}``。
    """
    # 上次进程被杀留下的 RUNNING trial 会永久占位。打开时就地判 FAIL ——
    # FAIL 不计入 n_valid，所以 12 条有效预算不会被它吃掉。
    stale = [t for t in study.trials
             if t.state == optuna.trial.TrialState.RUNNING]
    for t in stale:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        log(f"[stale] trial {t.number} 上次未收尾（RUNNING），标记为 FAIL"
            f"（不计入有效预算）")

    n_val = n_valid(study)
    n_wait = 0
    enqueued = []
    if n_val < max_trials:
        # §5：Trial 0 用 enqueue。只在 study 还是空的时候入队，保证 Trial 0
        # 恰好是默认配置那一条；resume 时不会重复入队。
        if len(study.trials) == 0:
            study.enqueue_trial(defaults)
            log("[trial0] 已 enqueue 为 Trial 0（默认配置）")

        # startup 阶段补队：在 COMPLETE+PRUNED 走满 N_STARTUP 之前，每一步都由
        # 本脚本确定性抽出、互不重复（见文件头"startup 采样"一节）。补到
        # N_STARTUP - 已有效 - 已在队，所以一条不多一条不少。
        waiting = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.WAITING]
        n_wait = len(waiting)
        need = N_STARTUP - n_val - n_wait
        need = max(0, min(need, max_trials - n_val - n_wait))
        nxt = len(study.trials)              # 入队后依次拿到的 trial number
        if need:
            used = used_keys(study)
            for i in range(need):
                cfg = next_startup_config(used)
                study.enqueue_trial(cfg)
                used.add(config_key(cfg))
                enqueued.append(cfg)
                log(f"[startup] enqueue 确定性配置 -> trial #{nxt + i}: "
                    f"{json.dumps(cfg)}")
        elif n_val < N_STARTUP:
            log(f"[startup] 队列已有 {n_wait} 条待跑，补队至 "
                f"{n_val + n_wait}/{N_STARTUP}，不再入队")
    return {"n_valid": n_val, "n_stale": len(stale), "n_waiting": n_wait,
            "enqueued": enqueued}


def close_logging():
    """把 root logger 的 handler 关掉 —— ``T.setup_logging`` 用的是
    ``logging.basicConfig``，handler 挂在 root 上、文件句柄不会自己关。"""
    for h in logging.root.handlers[:]:
        try:
            h.flush()
            h.close()
        except Exception:                                      # noqa: BLE001
            pass
        logging.root.removeHandler(h)


def make_objective(combo, seed, max_epochs, patience, min_epochs, gpu, log):
    """闭包出 objective：每条 trial 自己建 config、自己训练、自己清理。"""
    slug = COMBOS[combo][0]
    modalities = modalities_of(combo)
    phys_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu))

    def objective(trial):
        params = sample_params(trial)
        # 重复配置不训练：判重口径与 n_valid 完全一致。enqueue 那条路已经
        # 保证不重复了，这里是兜底（万一 TPE 阶段抽出已有配置）。
        # 抛在 try 之前 —— 此时还没有日志 handler / 临时目录要收，finally 不必跑。
        dup = duplicate_of(trial.study, params)
        if dup is not None:
            log(f"[trial {trial.number:>2}] DUPLICATE 与已有 trial #{dup} 逐位相同，"
                f"跳过训练 params={json.dumps(params)}")
            raise DuplicateConfig(f"trial {trial.number} 与 #{dup} 配置重复")
        tdir = TMP_ROOT / f"trial_{slug}_{trial.number:03d}"
        if tdir.exists():
            shutil.rmtree(tdir)
        tdir.mkdir(parents=True, exist_ok=True)

        cfg = O.build_config(combo, modalities, seed)
        for name, path in APPLY.items():
            set_nested(cfg, path, params[name])
        # eta_min 是 learning_rate 的派生量，必须跟着 trial 的 lr 走
        cfg["training"]["scheduler"]["eta_min"] = derive_eta_min(params["learning_rate"])
        # 统一训练协议：预算三项与正式 run 一致，不再"顺手"改出别的变量。
        cfg["training"]["num_epochs"] = max_epochs
        cfg["training"]["early_stopping"]["patience"] = patience
        cfg["training"]["early_stopping"]["min_epochs"] = min_epochs
        cfg["output"] = {"save_dir": str(tdir),
                         "log_dir": str(tdir / "logs"),
                         "img_dir": str(tdir / "img")}
        cfg["experiment"]["role"] = ("Optuna 搜索 trial（不修改 v3 结构；"
                                     "数据划分与正式 run 逐字段相同）")
        cfg["experiment"]["optuna"] = {"combination": combo, "seed": seed,
                                       "trial": trial.number, "params": params,
                                       "physical_gpu": phys_gpu}
        tdir.mkdir(parents=True, exist_ok=True)
        O.dump_config_yaml(cfg, tdir / "config.yaml")

        es = cfg["training"]["early_stopping"]
        # §运行原则要求的日志字段：combo / search seed / physical GPU /
        # trial number / best params（best value 在收尾时打）
        log(f"[trial {trial.number:>2}] combo={combo} search_seed={seed} "
            f"gpu={phys_gpu} trial={trial.number}")
        log(f"[trial {trial.number:>2}] params={json.dumps(params)}")
        log(f"[trial {trial.number:>2}] epochs={max_epochs} "
            f"patience={patience} min_epochs={es.get('min_epochs')} "
            f"monitor={es.get('monitor')} "
            f"eta_min={cfg['training']['scheduler']['eta_min']:.6g} dir={tdir}")

        status = "failed"
        try:
            seed_everything(seed)
            logger, timestamp = T.setup_logging(str(tdir / "logs"),
                                                f"optuna_{slug}_t{trial.number}")
            logger.info(f"[optuna] combo={combo} seed={seed} trial={trial.number} "
                        f"physical_gpu={phys_gpu} params={json.dumps(params)}")
            logger.info(f"[optuna] modalities={modalities} "
                        f"feature_base_dir={cfg['data']['feature_base_dir']}")
            logger.info(f"[optuna] val_start={cfg['data']['val_start']} "
                        f"sample_seed={cfg['data']['sample_seed']} "
                        f"env_seed={cfg['environment']['seed']}")

            trainer = HistoryTrainer(cfg, logger, timestamp)
            model, _ = trainer.train(optuna_trial=trial)

            reached = verify_reached(model, trainer.optimizer, params)
            # eta_min 不在模型上，在 trainer 上（train.py:1063 存了 actual）
            eta_actual = float(getattr(trainer, "eta_min_actual", float("nan")))
            eta_want = derive_eta_min(params["learning_rate"])
            assert abs(eta_actual - eta_want) < 1e-18, \
                f"eta_min 未随 lr 生效: {eta_actual} vs lr*0.01={eta_want}"
            reached["eta_min"] = eta_actual
            logger.info(f"[optuna] 参数落点核对通过: {json.dumps(reached)}")

            auc = float(trainer.best_val_auc)
            trial.set_user_attr("best_epoch", int(trainer.best_epoch) + 1)
            trial.set_user_attr("epochs_run", len(trainer.epoch_rows))
            trial.set_user_attr("params_reached", reached)
            trial.set_user_attr("metrics", O.primary_metrics(trainer.final_metrics))
            status = "complete"
            log(f"[trial {trial.number:>2}] DONE combo={combo} search_seed={seed} "
                f"gpu={phys_gpu} trial={trial.number} "
                f"best_value={auc:.6f} best_epoch={int(trainer.best_epoch) + 1} "
                f"epochs={len(trainer.epoch_rows)}")
            log(f"[trial {trial.number:>2}] best_params={json.dumps(params)}")

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
            return auc

        except optuna.TrialPruned:
            status = "pruned"
            log(f"[trial {trial.number:>2}] PRUNED")
            raise
        except BaseException:                                  # noqa: BLE001
            status = "failed"
            tb = traceback.format_exc()
            try:
                with open(tdir / "train.log", "a") as f:
                    f.write("\n" + "=" * 70 + "\n" + tb + "\n")
            except Exception:                                  # noqa: BLE001
                pass
            log(f"[trial {trial.number:>2}] FAILED\n{tb}")
            raise
        finally:
            close_logging()
            gc.collect()
            torch.cuda.empty_cache()
            # §7 磁盘口径：只有"当前最优"那条的 checkpoint 值得留，其余
            # （含被剪枝的）当场删干净。判定放在 objective 外面做不了 ——
            # 此刻 study 里还没记上本条 trial，所以这里只登记状态，
            # 真正删/留由 sync_best_and_clean() 决定。
            # set_user_attr 只是在 RUNNING 的 trial 上写一行，但仍包一层：
            # 万一 storage 出错，别让它把真正的 TrialPruned / 训练异常顶掉。
            try:
                trial.set_user_attr("_tmp_dir", str(tdir))
                trial.set_user_attr("_status", status)
            except Exception as exc:                           # noqa: BLE001
                print(f"[warn] trial {trial.number} 写 user_attr 失败: {exc}",
                      file=sys.stderr, flush=True)

    return objective


# ── 结果落盘 ─────────────────────────────────────────────────────────────

def original_stats(combo):
    """正式 10 次 run 的 mean ± std 与 best seed / best AUC。

    **现算**，不从 summary.txt 里抠字 —— 免得两份数字对不上。
    """
    seeds = O.load_or_create_seeds()[0]
    rows = []
    for s in seeds:
        if not O.is_completed(combo, s):
            continue
        m = json.loads((O.seed_dir(combo, s) / "metrics.json").read_text())
        rows.append(m)
    if not rows:
        return None
    aucs = np.array([m["AUC"] for m in rows], dtype=float)
    best = max(rows, key=lambda m: m["AUC"])
    return {
        "seeds": [m["seed"] for m in rows],
        "n_completed": len(rows),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "best_seed": int(best["seed"]),
        "best_auc": float(best["AUC"]),
    }


def trial_table(study):
    """study -> 逐 trial 一行，按 trial number 排序。"""
    rows = []
    for t in sorted(study.trials, key=lambda x: x.number):
        rows.append({
            "trial": t.number,
            "state": t.state.name,
            "value": (None if t.value is None else float(t.value)),
            **{k: t.params.get(k) for k in SPACE},
            "best_epoch": t.user_attrs.get("best_epoch"),
            "epochs_run": t.user_attrs.get("epochs_run"),
        })
    return rows


def best_complete(study):
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None]
    if not done:
        return None
    # 值相同取 trial number 小的（更早出现），与 trainer 保存 best 的严格改进口径一致
    return max(done, key=lambda t: (float(t.value), -t.number))


def sync_best_and_clean(study, combo, log):
    """把当前最优 trial 的 checkpoint 落到 ``<COMBO>/best_model.pt``，
    并把所有非最优的 trial 临时目录删掉（§7：搜索期间最多留一份 checkpoint）。

    返回最优 trial（没有则 None）。"""
    slug = COMBOS[combo][0]
    combo_dir = TUNE_ROOT / slug
    combo_dir.mkdir(parents=True, exist_ok=True)
    best = best_complete(study)
    kept = None
    if best is not None:
        src = Path(best.user_attrs.get("_tmp_dir", "")) / "best_model.pt"
        if src.is_file():
            dst = combo_dir / "best_model.pt"
            shutil.copy2(src, dst)
            kept = str(dst)
            log(f"[best ] trial {best.number} AUC={float(best.value):.4f} "
                f"-> {dst}")
        else:
            log(f"[best ] trial {best.number} 的 best_model.pt 不在了: {src}")

    # 所有 trial 的临时目录一律清掉 —— **包括最优那条**：它的 checkpoint
    # 刚刚已经复制到 <组合>/best_model.pt，目录里剩下的 config/img/logs
    # 没有留存价值（指标明细在 study 的 user_attrs 里，最终会写进
    # best_result.json）。唯一的例外是最新一条 FAILED：还要留着排错。
    newest = max((t.number for t in study.trials), default=-1)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.FAIL and t.number == newest:
            log(f"[keep ] trial {t.number} FAILED，暂留日志目录待查")
            continue
        d = t.user_attrs.get("_tmp_dir")
        if d and Path(d).is_dir():
            shutil.rmtree(d, ignore_errors=True)
    return best


def finalize(study, combo, seed, args, log):
    """§8：把 best_params.json / best_result.json 写出来。"""
    slug = COMBOS[combo][0]
    combo_dir = TUNE_ROOT / slug
    best = best_complete(study)
    if best is None:
        log("[final] 没有任何完成的 trial，不写 best_*")
        return False

    params = {k: best.params.get(k) for k in SPACE}
    metrics = best.user_attrs.get("metrics")
    orig = original_stats(combo)
    table = trial_table(study)
    n_done = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for t in study.trials
                   if t.state == optuna.trial.TrialState.FAIL)

    (combo_dir / "best_params.json").write_text(json.dumps({
        "combination": combo,
        "search_seed": seed,
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "best_trial": best.number,
        "best_value": float(best.value),
        "params": params,
        "protocol": (f"epochs={args.max_epochs} patience={args.patience} "
                     f"min_epochs={args.min_epochs} T_max={args.max_epochs} "
                     f"eta_min=lr*{ETA_MIN_RATIO}"),
        "sampler_seed": SAMPLER_SEED,
        "search_space": {k: (list(v[1]) if v[0] == "cat"
                             else {"low": v[1], "high": v[2], "log": v[3]})
                         for k, v in SPACE.items()},
        "note": "6 个超参；不改 v3 结构；数据划分与正式 run 相同。",
        "holdout_validation": {"status": "pending", "worker": "GPU7",
                               "script": "tune/c17_v3/validate_holdout.py"},
    }, indent=2) + "\n")

    result = {
        "combination": combo,
        "dataset": "C17",
        "mil": "ABMIL",
        "model": "he_aux_unified (v3-style HE-anchored residual cross fusion)",
        "protocol": ("Train / Test-as-Val —— patient_000..099 train / "
                     "patient_100..199 test-as-val（与正式 140 runs 同一划分，"
                     "本次搜索未重新分折）"),
        "objective": "val_as_test AUC（trainer 重新加载 best checkpoint 后评估）",
        "search": {
            "search_seed": seed,
            "holdout_seeds": list(HOLDOUT_SEEDS),
            "holdout_validation": ("GPU7 worker（validate_holdout.py）在搜索完成后"
                                   "用 best params 跑 3611/6430/39998，"
                                   "与原始正式 baseline 做 paired comparison"),
            "protocol": (f"epochs={args.max_epochs} patience={args.patience} "
                         f"min_epochs={args.min_epochs} "
                         f"T_max={args.max_epochs} "
                         f"eta_min=lr*{ETA_MIN_RATIO}"),
            "n_trials_requested": args.max_trials,
            "max_epochs": args.max_epochs,
            "early_stopping_patience": args.patience,
            "early_stopping_min_epochs": args.min_epochs,
            "sampler": (f"TPESampler(seed={SAMPLER_SEED}+已有记录条数, "
                        f"n_startup_trials=4, multivariate=True)"
                        f" —— 种子随 trial 号推进，避免每进程重放 RNG 导致提案锁死"),
            "pruner": "MedianPruner(n_startup_trials=4, n_warmup_steps=8, interval_steps=2)",
            "trial0": "当前 v3 实际配置（study.enqueue_trial，值读自磁盘 config.yaml）",
            "search_space": {k: (list(v[1]) if v[0] == "cat"
                                 else [v[1], v[2], "log" if v[3] else "linear"])
                             for k, v in SPACE.items()},
        },
        "original_10runs": orig,
        "optuna": {
            "best_trial": best.number,
            "best_epoch": best.user_attrs.get("best_epoch"),
            "best_auc": float(best.value),
            "params": params,
            "epochs_run": best.user_attrs.get("epochs_run"),
            "metrics": metrics,
            "params_reached_in_model": best.user_attrs.get("params_reached"),
            "n_completed": n_done,
            "n_pruned": n_pruned,
            "n_failed": n_failed,
            "trials": table,
        },
    }
    # 早停预算由上文的 args.min_epochs 显式写入 config（统一协议），
    # 不再回头从磁盘 config 读 —— 那会把"搜索协议"和"历史配置"混为一谈。

    (combo_dir / "best_result.json").write_text(json.dumps(result, indent=2) + "\n")
    phys_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    log(f"[final] wrote {combo_dir / 'best_params.json'} 与 best_result.json")
    log(f"[final] combo={combo} search_seed={seed} gpu={phys_gpu} "
        f"best_trial={best.number} best_value={float(best.value):.6f}")
    log(f"[final] best_params={json.dumps(params)}")
    # hold-out validation 由 GPU7 的 worker 负责，这里只登记状态，
    # 好让 worker 一看到 best_params.json 出现就能领取。
    log(f"[final] holdout_validation=pending "
        f"seeds={list(HOLDOUT_SEEDS)} -> GPU7 (validate_holdout.py)")
    return True


def _sweep_tmp():
    """清理临时/缓存。**只在 TUNE_ROOT 子树内动手**，每一步都先断言路径在
    TUNE_ROOT 下 —— 规格里最硬的一条是"绝对不要删除或修改原来的
    C17+abmil / features / dataset / 原始 checkpoint / 原始实验结果"。"""
    root = TUNE_ROOT.resolve()
    removed = []

    def under_tune(p):
        return root == p or root in p.parents

    # 空的 trial 目录（失败/中断留下的）
    if TMP_ROOT.is_dir():
        for d in TMP_ROOT.iterdir():
            if d.is_dir() and d.name.startswith("trial_") and not any(d.iterdir()):
                if under_tune(d.resolve()):
                    d.rmdir()
                    removed.append(str(d))
    # __pycache__ / *.pyc
    for cache in list(TUNE_ROOT.rglob("__pycache__")) + list(TUNE_ROOT.rglob("*.pyc")):
        p = cache.resolve()
        if not under_tune(p):
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed.append(str(p))
        except OSError:
            pass
    return removed


def write_summary_csv():
    """所有组合跑完后汇总一张表（§7 允许的 summary.csv）。"""
    cols = ["combination", "seed", "best_trial", "best_epoch", "best_auc",
            "orig_auc_mean", "orig_auc_std", "orig_best_seed", "orig_best_auc",
            "delta_vs_orig_mean", "n_completed", "n_pruned", "n_failed",
            *SPACE.keys()]
    rows = []
    for combo, (slug, seed) in COMBOS.items():
        p = TUNE_ROOT / slug / "best_result.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        o = r.get("original_10runs") or {}
        rows.append({
            "combination": combo,
            "seed": seed,
            "best_trial": r["optuna"]["best_trial"],
            "best_epoch": r["optuna"]["best_epoch"],
            "best_auc": r["optuna"]["best_auc"],
            "orig_auc_mean": o.get("auc_mean"),
            "orig_auc_std": o.get("auc_std"),
            "orig_best_seed": o.get("best_seed"),
            "orig_best_auc": o.get("best_auc"),
            "delta_vs_orig_mean": (None if not o else
                                   r["optuna"]["best_auc"] - o["auc_mean"]),
            "n_completed": r["optuna"]["n_completed"],
            "n_pruned": r["optuna"]["n_pruned"],
            "n_failed": r["optuna"]["n_failed"],
            **{k: r["optuna"]["params"].get(k) for k in SPACE},
        })
    if not rows:
        return None
    out = TUNE_ROOT / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True, choices=list(COMBOS))
    ap.add_argument("--seed", type=int, default=None,
                    help="默认用该组合固定的 best seed；传了会警告")
    ap.add_argument("--max-trials", type=int, default=N_TRIALS,
                    help=f"该组合的**有效** trial 数（默认 {N_TRIALS}）："
                         f"COMPLETE/PRUNED 且配置互不重复才算，"
                         f"FAIL / 孤儿 RUNNING / 重复配置不计入")
    ap.add_argument("--trials-per-process", type=int, default=1,
                    help="本进程最多跑几条 trial（默认 1，跑完即退出、显存归还）")
    ap.add_argument("--max-epochs", type=int, default=SEARCH_EPOCHS)
    ap.add_argument("--patience", type=int, default=SEARCH_PATIENCE)
    ap.add_argument("--min-epochs", type=int, default=SEARCH_MIN_EPOCHS)
    ap.add_argument("--gpu", type=int, default=None,
                    help="物理卡号，只用于日志（真正的设备选择靠 CUDA_VISIBLE_DEVICES）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：只跑 1 条 trial，**不做 finalize、不删 study.db**")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打开 study 报告状态（验证 resume），不训练")
    ap.add_argument("--report-only", action="store_true",
                    help="不训练，只按 study 现有结果重写 best_* / summary.csv")
    ap.add_argument("--force", action="store_true",
                    help="组合已完成（best_result.json 在、study.db 不在）时也继续")
    args = ap.parse_args()

    combo = args.combo
    slug, fixed_seed = COMBOS[combo]
    seed = fixed_seed
    if args.seed is not None and args.seed != fixed_seed:
        print(f"[warn] --seed {args.seed} 覆盖了 {combo} 固定的搜索 seed "
              f"{fixed_seed}")
        seed = args.seed
    # hold-out seed 是验证专用，**绝不能**被搜索碰到 —— 一旦拿去搜索，
    # GPU7 的 paired comparison 就失去意义（同一份数据既调参又验证）。
    assert seed not in HOLDOUT_SEEDS, (
        f"搜索 seed {seed} 属于 hold-out 集 {HOLDOUT_SEEDS}，禁止用于搜索")
    if args.smoke:
        args.max_trials = max(args.max_trials, 1)
        args.trials_per_process = 1

    combo_dir = TUNE_ROOT / slug
    combo_dir.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    study_db = combo_dir / "study.db"
    done_marker = combo_dir / "best_result.json"

    # 搜索过程日志放 tmp/ 而不是组合目录 —— §7 规定每个组合最终只留
    # best_model.pt / best_params.json / best_result.json 三个文件。
    # tmp/ 里的东西在组合成功收尾时一并删掉；逐 trial 的明细本来就已经
    # 完整写进 best_result.json 的 trials 列表，不会丢。
    log_path = TMP_ROOT / f"{slug}_run.log"
    log_f = open(log_path, "a")

    def log(msg):
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    if done_marker.is_file() and not study_db.is_file() and not args.force:
        log(f"[skip ] {combo} 已完成（{done_marker} 在、study.db 已删）。"
            f"要重搜请先手动备份并删除 {done_marker}，或加 --force")
        return EXIT_FINISHED

    # report-only 必须在 create_study 之前拦：load_if_exists=True 会在
    # study.db 已被删掉时**新建一个空的**，那就白留一个空库。
    if args.report_only and not study_db.is_file():
        log(f"[err  ] --report-only 需要 {study_db}，但它不在"
            f"（组合已收尾、best_result.json 已定稿，无需重写）")
        return

    # ── Trial 0 = 当前 v3 实际配置 ──
    defaults, src = read_defaults(combo, seed)
    assert_in_space(defaults, "Trial 0 默认值")
    log(f"[trial0] 默认值来源: {src}")
    log(f"[trial0] {json.dumps(defaults)}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # 种子在 open_study 里由 study.db 的现有条数定下来（理由见 sampler_seed_for）
    study, sd = open_study(study_db, slug)
    n_seed, s_seed = sd["n_seed"], sd["s_seed"]

    # 孤儿 RUNNING 判 FAIL + startup 补队，全在 prepare_study 里（与 dry-run
    # 模拟器共用同一段逻辑）。
    prep = prepare_study(study, defaults, args.max_trials, log)
    n_rec = len(study.trials)
    n_val = prep["n_valid"]
    phys_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu))
    log(f"[study] combo={combo} search_seed={seed} gpu={phys_gpu} "
        f"storage={study_db}")
    log(f"[study] sampler seed={s_seed}（= {SAMPLER_SEED} + 已有 {n_seed} 条记录；"
        f"随 trial 号推进，保证 TPE 提案不重放）")
    log(f"[study] 协议 epochs={args.max_epochs} patience={args.patience} "
        f"min_epochs={args.min_epochs} T_max={args.max_epochs} "
        f"eta_min=lr*{ETA_MIN_RATIO}")
    log(f"[study] 有效 trial {n_val}/{args.max_trials}"
        f"（库内共 {n_rec} 条记录；COMPLETE/PRUNED 且配置互不重复才算有效）"
        f"，本次进程最多再跑 {args.trials_per_process} 条")

    if args.dry_run:
        b = best_complete(study)
        log(f"[dry  ] resume 正常：study 可读，库内 {n_rec} 条记录、有效 {n_val} 条，"
            f"下一条 number = {n_rec}，还差 {max(0, args.max_trials - n_val)} 条有效 trial")
        if prep["enqueued"]:
            log(f"[dry  ] 上面这几条 enqueue 已经写进 {study_db.name} —— 是幂等的："
                f"正式跑时 prepare_study 会看到队列已满，原样复用、不会重复入队")
        if b is not None:
            log(f"[dry  ] 当前最优 trial {b.number} value={float(b.value):.6f}")
        return

    if args.report_only:
        sync_best_and_clean(study, combo, log)
        if finalize(study, combo, seed, args, log):
            p = write_summary_csv()
            if p:
                log(f"[final] wrote {p}")
        return

    if n_val >= args.max_trials:
        log(f"[done ] 已有 {n_val} 条有效 trial（配置互不重复），直接收尾")
    else:
        objective = make_objective(combo, seed, args.max_epochs, args.patience,
                                   args.min_epochs, args.gpu, log)
        for _ in range(args.trials_per_process):
            if n_valid(study) >= args.max_trials:
                break
            try:
                # catch：重复配置的 DuplicateConfig 由 optuna 吞掉并标 FAIL
                # （_optimize.py:258-263 只在 not isinstance(err, catch) 时
                # 重新抛出）。该条不计入 n_valid，下一轮进程会补一条新的。
                study.optimize(objective, n_trials=1, catch=(DuplicateConfig,))
            except KeyboardInterrupt:
                log("[abort] 收到中断")
                break

    best = sync_best_and_clean(study, combo, log)
    finished = n_valid(study) >= args.max_trials

    if args.smoke:
        log("[smoke] 冒烟模式：不 finalize、不删 study.db")
        if best is not None:
            log(f"[smoke] 当前最优 trial {best.number} AUC={float(best.value):.4f} "
                f"best_epoch={best.user_attrs.get('best_epoch')}")
        log(f"[smoke] 剩余 {max(0, args.max_trials - n_valid(study))} 条有效 trial 待跑")
        return

    if not finished:
        log(f"[hold ] {n_valid(study)}/{args.max_trials} 条有效 trial 完成，"
            f"尚未收尾（等后续进程继续）")
        return 0

    if not finalize(study, combo, seed, args, log):
        return 0
    write_summary_csv()

    # §7：组合成功收尾、产物确认后才删 study.db。
    ok = all((combo_dir / f).is_file()
             for f in ("best_model.pt", "best_params.json", "best_result.json"))
    if ok:
        study_db.unlink()
        log(f"[clean] 产物齐全，已删除 {study_db}（断点续跑记录到这里为止）")
        # 组合目录最终只剩 §7 允许的三个文件
        log(f"[clean] {slug}/ = {sorted(p.name for p in combo_dir.iterdir())}")
        log_f.flush()
        log_f.close()
        log_path.unlink(missing_ok=True)
        _sweep_tmp()
    else:
        log(f"[clean] 产物不全 {sorted(p.name for p in combo_dir.iterdir())}，"
            f"保留 study.db")
    return EXIT_FINISHED


if __name__ == "__main__":
    # 退出码约定（run_optuna.sh 就靠它决定要不要继续下一条）：
    #   0  = 正常，但该组合还有 trial 没跑完 -> 接着跑
    #   10 = 该组合的 12 条已经跑满并收尾（或本来就已完成）-> 停这条链
    #   其他 = 出错
    sys.exit(main() or 0)
