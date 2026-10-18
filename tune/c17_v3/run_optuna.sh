#!/usr/bin/env bash
# C17 + ABMIL + v3 正式 Optuna —— 5 个组合 × 12 trials = 60 条 + GPU7 hold-out 验证。
#
# ┌───────────────── 固定 GPU 映射（用户 2026-09-18 指定）─────────────────┐
# │   GPU2  ->  HE+Ki67          搜索（12 trials，卡内串行）                │
# │   GPU3  ->  HE+PR            搜索                                       │
# │   GPU4  ->  HE+HER2+Ki67     搜索                                       │
# │   GPU5  ->  HE+PR+ER         搜索                                       │
# │   GPU6  ->  HE+PR+HER2+ER    搜索                                       │
# │   GPU7  ->  hold-out validation worker（不参与 Optuna study）           │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 结构
# ------
# * 一个组合一条链，一条链钉死一张卡，**链内 12 条 trial 串行**。
# * 每条 trial 是**一个独立进程**（--trials-per-process 1），进程退出即归还
#   显存；被剪枝的 trial 不会把 CUDA 上下文留给下一条。
# * 同一个 study 绝不跨 GPU 并发写 —— 每个组合只有一个链、只在一张卡上跑。
# * GPU7 的 worker 轮询等待：哪个组合的 best_params.json 落盘就验证哪个
#   （seeds 3611/6430/39998，与原始正式 baseline 做 paired comparison）。
#   **暂无可验证组合时它只等待，绝不碰 Optuna study。**
#
# 统一训练协议（搜索 / hold-out 验证 / 历史正式 run 三者同源）
# -------------------------------------------------------------
#   epochs=30  patience=15  min_epochs=10  T_max=30  eta_min=lr*0.01
#   搜索 seed 全部 = 42；hold-out seeds = 3611/6430/39998（搜索期禁用）。
#
# 断点续跑
# ---------
# 进度全在 <组合>/study.db 里。中断后原样重跑本脚本即可，已完成的 trial
# 不会重来，链会自己接着往下走。
#
# 磁盘
# ------
# 搜索期每个组合最多留一份 checkpoint（<组合>/best_model.pt）；非最优 trial
# 的临时目录当场删除；组合收尾后 study.db 与临时日志一并删除，组合目录最终
# 只剩 best_model.pt / best_params.json / best_result.json（+ holdout/ 指标）。
# 清理范围**严格限定在 tune/c17_v3/ 子树内**，永不触碰
# C17+abmil/ features/ dataset/ 与任何原始 checkpoint / 实验结果。
#
#   bash tune/c17_v3/run_optuna.sh --dry-run    # 只打印 GPU 映射，不跑
#   bash tune/c17_v3/run_optuna.sh              # 正式：5 搜索链 + GPU7 验证
#   bash tune/c17_v3/run_optuna.sh --no-validate   # 只搜索，不启 GPU7
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/cxl/miniconda3/envs/rrtmil/bin/python}"
TUNE_ROOT="$REPO_ROOT/tune/c17_v3"

# 兜底：确保 conda 自带的 libstdc++ 优先于系统那份。
# 不设的话，先 import torch 再 import optuna 会报
#   ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
# （conda 的 libicui18n.so.78 被系统 libstdc++ 接走）。两个子脚本的 import
# 顺序已经修好，这里再上一层保险，让行为不依赖 import 顺序。
CONDA_LIB="$(dirname "$(dirname "$PY")")/lib"
if [ -d "$CONDA_LIB" ]; then
    export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

MAX_TRIALS=12
VALIDATION_GPU=7
DRY_RUN=0
WITH_VALIDATION=1

while [ $# -gt 0 ]; do
    case "$1" in
        --max-trials)     MAX_TRIALS="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1;       shift   ;;
        --no-validate)    WITH_VALIDATION=0; shift ;;
        --validation-gpu) VALIDATION_GPU="$2"; shift 2 ;;
        -h|--help)        sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# 组合 -> 搜索 seed（统一 42，与 optuna_v3.py 的 SEARCH_SEED 保持一致）
search_seed_of () {
    case "$1" in
        "HE+Ki67"|"HE+PR"|"HE+HER2+Ki67"|"HE+PR+ER"|"HE+PR+HER2+ER") echo 42 ;;
        *) echo "" ;;
    esac
}
slug_of () {
    case "$1" in
        "HE+Ki67")       echo HE_Ki67       ;;
        "HE+PR")         echo HE_PR         ;;
        "HE+HER2+Ki67")  echo HE_HER2_Ki67  ;;
        "HE+PR+ER")      echo HE_PR_ER      ;;
        "HE+PR+HER2+ER") echo HE_PR_HER2_ER ;;
    esac
}
# 组合 -> 物理 GPU。**这就是那张固定映射表**，改这里就是改映射。
gpu_of () {
    case "$1" in
        "HE+Ki67")       echo 2 ;;
        "HE+PR")         echo 3 ;;
        "HE+HER2+Ki67")  echo 4 ;;
        "HE+PR+ER")      echo 5 ;;
        "HE+PR+HER2+ER") echo 6 ;;
    esac
}

COMBOS="HE+Ki67 HE+PR HE+HER2+Ki67 HE+PR+ER HE+PR+HER2+ER"

# ── 自检：映射必须是一一对应，且搜索卡不能撞上验证卡 ──────────────────
declare -A seen_gpu=()
for combo in $COMBOS; do
    g="$(gpu_of "$combo")"
    [ -n "$g" ] || { echo "组合 $combo 没有 GPU 映射" >&2; exit 2; }
    [ -n "$(search_seed_of "$combo")" ] || { echo "组合 $combo 没有搜索 seed" >&2; exit 2; }
    if [ -n "${seen_gpu[$g]:-}" ]; then
        echo "GPU$g 被 $combo 和 ${seen_gpu[$g]} 同时占用 —— 搜索卡必须一卡一组合" >&2
        exit 2
    fi
    seen_gpu[$g]="$combo"
    if [ "$g" = "$VALIDATION_GPU" ]; then
        echo "GPU$g 既是搜索卡又是验证卡 —— 必须分开" >&2
        exit 2
    fi
done

print_map () {
    echo "+------+------------------+-----------+--------------------------------+"
    printf '| %-4s | %-16s | %-9s | %s\n' "GPU" "combo" "searchseed" "role"
    echo "+------+------------------+-----------+--------------------------------+"
    for combo in $COMBOS; do
        printf '| %-4s | %-16s | %-9s | %s\n' \
            "$(gpu_of "$combo")" "$combo" "$(search_seed_of "$combo")" \
            "Optuna search (12 trials, serial)"
    done
    printf '| %-4s | %-16s | %-9s | %s\n' \
        "$VALIDATION_GPU" "(worker)" "-" "hold-out validation 3611/6430/39998"
    echo "+------+------------------+-----------+--------------------------------+"
    echo "protocol: epochs=30 patience=15 min_epochs=10 T_max=30 eta_min=lr*0.01"
}

echo "repo       : $REPO_ROOT"
echo "python     : $PY"
echo "max-trials : $MAX_TRIALS / 组合"
echo "validation : $([ $WITH_VALIDATION -eq 1 ] && echo "GPU$VALIDATION_GPU 启用" || echo 关闭)"
echo
print_map
echo

if [ $DRY_RUN -eq 1 ]; then
    echo "[dry-run] 映射自检通过：5 个组合分别绑定 GPU2/3/4/5/6，互不冲突，"
    echo "[dry-run] GPU$VALIDATION_GPU 只进 validation queue，不参与 Optuna study。"
    echo "[dry-run] 未启动任何训练。去掉 --dry-run 即为正式启动。"
    exit 0
fi

mkdir -p "$TUNE_ROOT/tmp"
DRIVER_LOG="$TUNE_ROOT/tmp/driver_$(date +%m%d%H%M).log"
echo "driver log : $DRIVER_LOG"
echo

# 一条搜索链：钉一张卡，串行跑完该组合的全部 trial。
run_chain () {
    local combo="$1" gpu="$2"
    local seed slug
    seed="$(search_seed_of "$combo")"
    slug="$(slug_of "$combo")"
    mkdir -p "$TUNE_ROOT/$slug"

    echo "[chain] START combo=$combo search_seed=$seed gpu=$gpu" | tee -a "$DRIVER_LOG"
    # 每轮最多 1 条 trial。循环上限 = MAX_TRIALS * 2：正常跑满 12 条**有效**
    # trial 后脚本以退出码 10（EXIT_FINISHED）结束并 break；多出来的 12 轮是给
    # 失败留的余量 —— 有效 trial 的口径是"COMPLETE/PRUNED 且配置互不重复"，
    # FAIL（含孤儿 RUNNING 被就地判 FAIL 的）、重复配置都不占预算，所以失败
    # 之后必须再开一轮把它补回来。上限只是防死循环，正常情况下用不到。
    #
    # **不把子进程输出收进变量**：trainer 的 tqdm 进度条每个 iteration 都刷
    # 一行，一条 trial 就是好几 MB，60 条会把 driver log 撑到上百 MB，与 §7
    # 的磁盘口径冲突。逐 trial 详情本来就有 tmp/<slug>_run.log 与 study.db 记着，
    # 这里只需要退出码。
    local attempt rc
    for attempt in $(seq 1 $((MAX_TRIALS * 2))); do
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
        "$PY" "$TUNE_ROOT/optuna_v3.py" \
            --combo "$combo" --seed "$seed" \
            --max-trials "$MAX_TRIALS" --trials-per-process 1 \
            --gpu "$gpu" >> "$DRIVER_LOG" 2>&1
        rc=$?

        if [ $rc -eq 10 ]; then
            echo "[chain] DONE  combo=$combo gpu=$gpu（$MAX_TRIALS 条有效 trial 跑满并收尾）" \
                | tee -a "$DRIVER_LOG"
            return 0
        fi
        if [ $rc -ne 0 ]; then
            # 单条 trial 失败不终止整条链：该 trial 记 FAIL，**不占** 12 条有效
            # 预算，下一轮会补跑一条新的。OOM 也一样 —— 不在这里改 batch_size /
            # max_patches / 模型参数，改卡请改 gpu_of()。
            echo "[chain] WARN  combo=$combo gpu=$gpu 第 $attempt 轮 rc=$rc（该 trial 记为 FAIL、不计入有效预算，继续补跑）" \
                | tee -a "$DRIVER_LOG"
        fi
    done
    echo "[chain] STOP  combo=$combo gpu=$gpu 达到循环上限 $((MAX_TRIALS * 2)) 轮，需人工检查 $TUNE_ROOT/$slug/study.db" \
        | tee -a "$DRIVER_LOG"
    return 1
}

pids=()
names=()

# GPU7：hold-out validation worker。**不参与 Optuna study**，只轮询等待
# best_params.json 落盘，出现一个就验一个。
if [ $WITH_VALIDATION -eq 1 ]; then
    echo "[valid] START gpu=$VALIDATION_GPU（hold-out validation worker，" \
         "等待搜索完成后领取）" | tee -a "$DRIVER_LOG"
    CUDA_VISIBLE_DEVICES="$VALIDATION_GPU" OMP_NUM_THREADS=4 \
    "$PY" "$TUNE_ROOT/validate_holdout.py" --gpu "$VALIDATION_GPU" \
        >> "$DRIVER_LOG" 2>&1 &
    pids+=($!); names+=("validate:GPU$VALIDATION_GPU")
fi

# GPU2/3/4/5/6：5 条搜索链同时启动
for combo in $COMBOS; do
    gpu="$(gpu_of "$combo")"
    run_chain "$combo" "$gpu" &
    pids+=($!); names+=("$combo:GPU$gpu")
done

fail=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        fail=$(( fail + 1 ))
        echo "[main ] ${names[$i]} 异常返回" | tee -a "$DRIVER_LOG"
    fi
done

echo
echo "=== 全部结束：${#pids[@]} 个后台任务，$fail 个异常返回 ==="
for combo in $COMBOS; do
    slug="$(slug_of "$combo")"
    echo "--- $combo (gpu=$(gpu_of "$combo"), $(slug)) ---"
    ls -la "$TUNE_ROOT/$slug" 2>/dev/null | tail -n +2
done
echo
echo "搜索汇总 : $TUNE_ROOT/summary.csv"
echo "验证汇总 : $TUNE_ROOT/summary_holdout.csv"
exit $(( fail > 0 ? 1 : 0 ))
