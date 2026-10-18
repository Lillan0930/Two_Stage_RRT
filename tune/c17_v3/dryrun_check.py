#!/usr/bin/env python
"""正式启动前的 dry-run 验证 —— 用户 2026-10-15 指定的 5 项检查。

    ① Trial 0 参数正确
    ② 连续 4 个 startup 配置互不相同
    ③ 同样 seed / 数据库，重启后采样可复现
    ④a TPE 段继续变化  ④b 提案由 value 决定  ④c startup 与 value 无关
    ④d **best/below 集连续多轮不动时，TPE 提案仍必须变化**
    ⑤ GPU 绑定正确

为什么 ③④ 必须跨进程验
------------------------
被修的缺陷就是"**每个新进程 RNG 重来**"：TPESampler 在 startup 窗口内把采样
委托给内部 RandomSampler，而 ``LazyRandomState.__init__`` 每次构造都
``rng.seed(seed)``。单进程里根本复现不出来 —— 必须像 driver 那样"一条 trial
一个进程"。所以本脚本把 driver 的行为如实模拟成**子进程序列**：

为什么 ④d 是这一轮新增的
------------------------
上一版 dry-run 的 ④a（TPE 段两两不同）**靠运气通过**：stub 值由参数哈希导出、
argmax 一直在换，于是永远走不到"below 集冻结"那条路。而真实跑里 argmax 一旦
稳定，``n_below = min(ceil(0.1n), 25)`` 在 n<=10 时恒为 1，below 集只剩 argmax
一个点 —— mpe_below 逐位不变，候选点云随常数种子逐轮重放，``argmax`` 于是每次
都挑中同一个下标，**提案锁死成不动点**。④d 就是把这个场景显式造出来：
``mode="frozen0"`` 让 trial 0 恒为 best，再要求 TPE 段 8 条仍两两不同。

``--legacy-seed`` 会把采样器退回旧的常数种子，用来证明 ④d 确实有牙
（旧行为下必须 FAIL）。**检查自己有牙**和**代码已修好**是两件事，都得验。

    python dryrun_check.py sim <db> <combo> <value-mode> <max-trials>

每个子进程做的是：打开 study → :func:`optuna_v3.prepare_study`（与正式跑
**同一段代码**，不是复制品）→ 用 stub objective 跑**恰好一条** trial。
stub 调 :func:`optuna_v3.sample_params` 和 :func:`optuna_v3.duplicate_of`
（同样是正式跑的那两个函数），然后返回一个由配置决定的确定性值。

**不训练、不碰 GPU、不写任何正式产物。** 所有 study.db 都在
``tune/c17_v3/tmp/dryrun/`` 下，跑完即删（``--keep`` 保留）。

    python tune/c17_v3/dryrun_check.py            # 全部检查
    python tune/c17_v3/dryrun_check.py --keep     # 保留中间 study.db
    python tune/c17_v3/dryrun_check.py --legacy-seed   # 额外跑一遍旧常数种子，
                                                       # ④d 应当 FAIL（证明检查有牙）
退出码 0 = 全部通过。
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import optuna                                                  # noqa: E402
import optuna_v3 as V                                          # noqa: E402

PY = sys.executable
SIM_COMBO = "HE+Ki67"          # 采样逻辑与组合无关，验一个即可
SIM_STEPS = V.N_TRIALS         # 12：正好覆盖 startup 段（4 条）+ TPE 段（8 条）
WORK = V.TMP_ROOT / "dryrun"

OK, NO = "PASS", "FAIL"


def say(ok):
    return "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"


# ── stub 目标值 ──────────────────────────────────────────────────────────

def stub_value(params, mode, number=0):
    """由配置决定的**确定性**目标值。

    绝不能用并列值 —— 那会让两份 history 等价，测不出 TPE 到底有没有在读它
    （2026-09-18 那次探针就踩过这个坑，得到的"提案相同"是假象）。

    ``mode="frozen0"`` 是给 ④d 造的：trial 0 恒拿 0.99 稳居第一，其余按 trial
    号严格递增地贴在 0.90 —— 于是 ``best`` 恒为 #0、below 集恒为 {#0}，
    **连续 12 轮一动不动**。值仍严格单调，所以 above 集的次序也是确定的。
    """
    if mode == "frozen0":
        return 0.99 if number == 0 else 0.90 + 1e-4 * number
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
    v = 0.85 + 0.10 * (int(h[:8], 16) / 0xFFFFFFFF)
    return 0.95 - (v - 0.85) if mode == "invert" else v


def make_study(db, combo=SIM_COMBO, legacy_seed=False):
    """建/载入 dry-run 用的 study —— **直接调 ``optuna_v3.open_study``**。

    2026-10-15 之前这里自己复制了一份 create_study，结果"每进程重放 RNG"这个
    缺陷恰好落在复制品之外，dry-run 根本照不到它。现在走正式代码路径：种子由
    study.db 里已有的记录条数决定（见 ``V.sampler_seed_for``），dry-run 验的
    就是正式跑的那段逻辑本身。

    ``legacy_seed=True`` 退回旧的常数种子 ``V.SAMPLER_SEED``；**只**用于证明
    ④d 有牙 —— 那是被修掉的行为，不是可选配置。
    """
    if legacy_seed:
        return optuna.create_study(
            study_name="dryrun_legacy",
            storage=f"sqlite:///{db}",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=V.SAMPLER_SEED,
                                               n_startup_trials=V.N_STARTUP,
                                               multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=4,
                                               n_warmup_steps=8,
                                               interval_steps=2),
            load_if_exists=True,
        )
    return V.open_study(db, V.COMBOS[combo][0])[0]


# ── 子进程 worker：模拟 driver 的一轮 ────────────────────────────────────

def sim_worker(db, combo, mode, max_trials, legacy_seed=False):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = make_study(db, combo, legacy_seed)
    defaults, _ = V.read_defaults(combo, V.COMBOS[combo][1])
    V.prepare_study(study, defaults, max_trials, lambda m: None)

    seen = {"value": None}

    def stub(trial):
        p = V.sample_params(trial)
        dup = V.duplicate_of(trial.study, p)
        seen["number"], seen["params"] = trial.number, p
        if dup is not None:
            seen["duplicate_of"] = dup
            raise V.DuplicateConfig(f"sim trial {trial.number} 与 #{dup} 重复")
        seen["value"] = stub_value(p, mode, trial.number)
        return seen["value"]

    study.optimize(stub, n_trials=1, catch=(V.DuplicateConfig,))
    t = next(x for x in study.trials if x.number == seen["number"])
    # 这一轮结束时谁是 best —— ④d 要拿它证明"below 集真的冻结了"，
    # 而不是假设它冻结了。
    try:
        best_number = study.best_trial.number
    except ValueError:                       # 一条 COMPLETE 都没有
        best_number = None
    print("SIM " + json.dumps({
        "number": t.number, "state": t.state.name,
        "params": V.trial_params_of(t),
        "duplicate_of": seen.get("duplicate_of"),
        "n_valid": V.n_valid(study),
        "value": t.value,                    # 库里记下的，不是我以为返回的
        "best_number": best_number,
    }), flush=True)


# ── 驱动器 ───────────────────────────────────────────────────────────────

def run_steps(db, mode, n, legacy_seed=False):
    """跑 n 轮，**每轮一个独立进程**（与 driver 的 --trials-per-process 1 一致）。"""
    argv = [PY, str(Path(__file__).resolve()), "sim",
            str(db), SIM_COMBO, mode, str(V.N_TRIALS)]
    if legacy_seed:
        argv.append("--legacy-seed")
    seq = []
    for i in range(n):
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"[sim] 第 {i} 轮子进程失败 rc={r.returncode}\n"
                             f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")
        hits = [l for l in r.stdout.splitlines() if l.startswith("SIM ")]
        if not hits:
            raise SystemExit(f"[sim] 第 {i} 轮没有 SIM 输出\n{r.stdout}\n{r.stderr}")
        seq.append(json.loads(hits[-1][4:]))
    return seq


def snap_db(src, dst):
    """把 study.db 连同可能存在的 -wal/-shm 一起复制 —— 相当于"进程被杀之后
    从同一份数据库重启"。"""
    for p in sorted(src.parent.glob(src.name + "*")):
        shutil.copy2(p, dst.parent / (dst.name + p.name[len(src.name):]))


def keys_of(seq):
    return [V.config_key(r["params"]) for r in seq]


def short(params):
    return (f"lr={params['learning_rate']:.6e} wd={params['weight_decay']:.3e} "
            f"do={params['dropout']:<5} rs={params['residual_scale']:<6}"
            f"tau={params['tau']:<5} mom={params['prototype_momentum']}")


def head(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ── ① Trial 0 参数正确 ───────────────────────────────────────────────────

def check_trial0():
    head("① Trial 0 = 当前正式基线参数（与历史 seed42 run 逐字段比）")
    bad = []
    for combo, (slug, seed) in V.COMBOS.items():
        params, src = V.read_defaults(combo, seed)
        try:
            V.assert_in_space(params, f"{combo} Trial 0")
        except AssertionError as e:
            bad.append(f"{combo}: {e}")
            print(f"  {combo:<15} {say(False)}  {e}")
            continue

        # 端到端：建一个空 study，走 prepare_study（正式跑的那段逻辑），
        # 跑一条 stub trial，看它**实际拿到**的参数是不是默认值。
        db = WORK / f"t0_{slug}.db"
        for p in db.parent.glob(db.name + "*"):
            p.unlink()
        study = make_study(db)
        V.prepare_study(study, params, V.N_TRIALS, lambda m: None)
        got = {}

        def stub(trial):
            got.update(V.sample_params(trial))
            return 0.0

        study.optimize(stub, n_trials=1, catch=(V.DuplicateConfig,))
        t0 = next(t for t in study.trials if t.number == 0)

        diff = [k for k in V.SPACE if repr(got[k]) != repr(params[k])]
        if diff:
            bad.append(f"{combo}: Trial 0 实采样值与基线不符 -> {diff}")
            print(f"  {combo:<15} {say(False)}  实采样与基线不符: {diff}")
            continue
        try:
            auc = json.loads((V.O.seed_dir(combo, seed) / "metrics.json")
                             .read_text())["AUC"]
            aucs = f"历史 seed{seed} AUC={auc:.10f}"
        except Exception:                                      # noqa: BLE001
            aucs = "历史 AUC 读不到"
        print(f"  {combo:<15} {say(True)}  {short(params)}")
        print(f"  {'':<15}        enqueue->实采样逐位一致(trial #{t0.number})  {aucs}")
        print(f"  {'':<15}        基线来源: {src}")

    return ("① Trial 0 参数正确", not bad, "; ".join(bad))


# ── ②③④ 采样：确定性 + 不重复 + TPE 正常 ────────────────────────────────

def check_sampling():
    head("②③④ startup 采样（跨进程模拟 driver：一条 trial 一个进程）")
    db_a, db_b, db_s, db_c = (WORK / f"{n}.db" for n in ("A", "B", "S", "C"))
    for db in (db_a, db_b, db_s, db_c):
        for p in db.parent.glob(db.name + "*"):
            p.unlink()

    print("  跑 db A：12 轮独立进程（startup 4 + TPE 8）…", flush=True)
    # 在第 3 条之后留一个快照，用来验"断点续跑"（③b）。
    a_head = run_steps(db_a, "normal", 3)
    snap_db(db_a, db_s)
    a_4th = run_steps(db_a, "normal", 1)
    a_rest = run_steps(db_a, "normal", SIM_STEPS - 4)
    seq_a = a_head + a_4th + a_rest

    print(f"  跑 db B：再来一遍（验可复现）…", flush=True)
    seq_b = run_steps(db_b, "normal", SIM_STEPS)

    print(f"  从 db A 第 3 条之后的快照重启一轮（验断点续跑）…", flush=True)
    s_one = run_steps(db_s, "normal", 1)[0]

    print(f"  跑 db C：目标值**全程取反**（验 TPE 的提案确实由 value 决定）…",
          flush=True)
    seq_c = run_steps(db_c, "invert", V.N_STARTUP + 1)

    print()
    print("  ── 12 条配置序列（db A）────────────────────────────────────────────")
    for r in seq_a:
        phase = "startup" if r["number"] < V.N_STARTUP else "TPE    "
        tag = "  <-- Trial0=基线" if r["number"] == 0 else ""
        print(f"   #{r['number']:<2} {phase} {r['state']:<8} {short(r['params'])}{tag}")

    fails = [r for r in seq_a + seq_b if r["state"] != "COMPLETE"]
    dupes = [r for r in seq_a + seq_b if r.get("duplicate_of") is not None]

    # ② startup 段（前 4 条）互不相同
    k = keys_of(seq_a)
    su = k[:V.N_STARTUP]
    ok2 = len(set(su)) == len(su)

    # ④ TPE 段（第 5 条起）继续变化：两两不同，且相邻不同
    tpe = k[V.N_STARTUP:]
    ok4a = len(set(tpe)) == len(tpe)
    # ④' TPE 的提案确实由 value 决定：把 history 的目标值**全部取反**，
    # below/above 的划分随之翻转，TPE 接手后的第一条提案必须改变。
    #
    # 机制说清楚（免得下次又设计错探针）：_ParzenEstimator 只用**参数**建密度、
    # 权重是均匀的，value 只通过"哪些 trial 落进 below/above"起作用。
    # 所以只翻一条 value、而那一条又恰好不是 argmax 时，划分不变 -> 提案本就
    # 不该变。那不是缺陷，是 TPE 的正常行为；探针必须翻到划分真的改变。
    kc = keys_of(seq_c)
    ok4b = kc[V.N_STARTUP] != k[V.N_STARTUP]
    # 附带：startup 段是 enqueue 出来的，与 value 无关，取反后应当逐位相同
    ok4s = kc[:V.N_STARTUP] == k[:V.N_STARTUP]

    # ③ 可复现：两个全新 db 的 12 条序列逐位相同；快照重启给同一条
    ok3a = keys_of(seq_b) == k
    ok3b = V.config_key(s_one["params"]) == k[3]
    ok3 = ok3a and ok3b

    print()
    print(f"  ② startup 4 条互不相同          : {say(ok2)}"
          f"  ({len(set(su))}/4 个不同配置)")
    print(f"  ③a 两个全新 db 的 12 条逐位相同  : {say(ok3a)}")
    print(f"  ③b 快照重启给出同一条配置        : {say(ok3b)}")
    print(f"  ④a TPE 段 8 条仍两两不同         : {say(ok4a)}"
          f"  ({len(set(tpe))}/8)")
    print(f"  ④b TPE 提案由 value 决定（取反后变样）: {say(ok4b)}")
    print(f"      A(正常) 第 {V.N_STARTUP} 条: {short(seq_a[V.N_STARTUP]['params'])}")
    print(f"      C(取反) 第 {V.N_STARTUP} 条: {short(seq_c[V.N_STARTUP]['params'])}")
    print(f"  ④c startup 段与 value 无关       : {say(ok4s)}")
    print(f"  附带 无 FAIL / 无重复配置        : {say(not fails and not dupes)}"
          f"  (FAIL {len(fails)}, 重复 {len(dupes)})")

    ok = ok2 and ok3 and ok4a and ok4b and ok4s and not fails and not dupes
    why = []
    if not ok2:
        why.append("startup 段出现重复配置")
    if not ok3a:
        why.append("两个全新 db 的序列不一致")
    if not ok3b:
        why.append("快照重启给出的配置与原来不同")
    if not ok4a:
        why.append("TPE 段出现重复配置")
    if not ok4b:
        why.append("目标值取反后 TPE 提案没变 —— TPE 没有在用 value 划分 below/above")
    if not ok4s:
        why.append("startup 段受了目标值影响（不该）")
    if fails or dupes:
        why.append(f"出现 FAIL({len(fails)}) 或重复({len(dupes)})")
    return ("②③④ startup 确定性 + 不重复 + TPE 正常", ok, "; ".join(why))


# ── ④d below 集冻结时提案仍须变化 ────────────────────────────────────────

def check_frozen(legacy=False):
    """④d：best/below 连续多轮一动不动时，TPE 后续提案仍应变化。

    这正是旧缺陷的致死场景 —— ``n_below = min(ceil(0.1n), 25)`` 在 n<=10 时
    恒为 1，below 集只剩 argmax；argmax 一稳，mpe_below 就逐位不变，配上每进程
    重放的常数种子，候选点云原样重演，``argmax`` 每次都挑中同一个下标，
    提案锁成不动点。所以这里**主动把 argmax 钉死在 #0**，再看 TPE 段会不会僵。

    同时打印 ``best=#?``：如果连"best 恒为 #0"这个前提都没成立，那 ④d 就是
    空跑一场，得当场说清楚，不能拿它冒充通过。
    """
    name = "④d′ 旧常数种子复现锁死（检查有牙？）" if legacy \
        else "④d best/below 冻结多轮后 TPE 提案仍变化"
    head(name + ("   —— 预期 FAIL" if legacy else ""))
    db = WORK / ("F_legacy.db" if legacy else "F.db")
    for p in db.parent.glob(db.name + "*"):
        p.unlink()

    print("  12 轮独立进程；目标值令 trial 0 恒为 best（below 集恒为 {#0}）…",
          flush=True)
    seq = run_steps(db, "frozen0", SIM_STEPS, legacy_seed=legacy)

    print()
    for r in seq:
        phase = "startup" if r["number"] < V.N_STARTUP else "TPE    "
        val = "  None  " if r["value"] is None else f"{r['value']:.6f}"
        print(f"   #{r['number']:<2} {phase} best=#{r['best_number']} "
              f"value={val}  {short(r['params'])}")

    frozen = [r["best_number"] for r in seq]
    pre = all(b == 0 for b in frozen)          # 前提：below 集真的冻结了
    tpe = keys_of(seq)[V.N_STARTUP:]
    n_distinct = len(set(tpe))
    ok = pre and n_distinct == len(tpe)

    print()
    print(f"  前提 best 恒为 #0（below 集冻结）: {say(pre)}"
          f"  ({sum(b == 0 for b in frozen)}/{len(frozen)} 轮)")
    print(f"  TPE 段 8 条两两不同             : {say(n_distinct == len(tpe))}"
          f"  ({n_distinct}/{len(tpe)})")
    if not pre:
        print("  ⚠ 前提没成立 —— 这一项等于空跑，不能算通过")

    why = []
    if not pre:
        why.append("best 没有恒定在 #0，场景没造出来")
    if n_distinct != len(tpe):
        why.append(f"TPE 段只有 {n_distinct}/{len(tpe)} 个不同配置 —— 提案锁死")
    return (name, ok, "; ".join(why))


# ── ⑤ GPU 绑定 ───────────────────────────────────────────────────────────

def check_gpu():
    head("⑤ GPU 绑定（直接跑 run_optuna.sh --dry-run，看它自己怎么认）")
    r = subprocess.run(["bash", str(HERE / "run_optuna.sh"), "--dry-run"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    out = r.stdout + r.stderr
    print("  " + "\n  ".join(l for l in out.splitlines()
                             if l.strip() and not l.startswith("repo ")
                             and not l.startswith("python ")
                             and not l.startswith("max-trials")
                             and not l.startswith("validation ")))

    want = {"HE+Ki67": "2", "HE+PR": "3", "HE+HER2+Ki67": "4",
            "HE+PR+ER": "5", "HE+PR+HER2+ER": "6"}
    bad = []
    if r.returncode != 0:
        bad.append(f"run_optuna.sh --dry-run rc={r.returncode}")
    for combo, gpu in want.items():
        # 表格行形如:  | 2    | HE+Ki67          | 42        | Optuna search ...
        hit = [l for l in out.splitlines()
               if combo in l and f"| {gpu} " in l and "Optuna search" in l]
        if not hit:
            bad.append(f"{combo} 没有绑到 GPU{gpu}")
    if "hold-out validation 3611/6430/39998" not in out:
        bad.append("GPU7 的 hold-out 角色没打印出来")
    for combo, gpu in want.items():
        print(f"  {combo:<15} -> GPU{gpu}   {say(not any(combo in b for b in bad))}")
    print(f"  GPU7 -> hold-out validation（不参与 Optuna study）   "
          f"{say('hold-out validation 3611/6430/39998' in out)}")
    return ("⑤ GPU 绑定正确", not bad, "; ".join(bad))


# ── 主流程 ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "sim"])
    ap.add_argument("db", nargs="?")
    ap.add_argument("combo", nargs="?")
    ap.add_argument("mode", nargs="?", default="normal")
    ap.add_argument("max_trials", nargs="?", type=int, default=V.N_TRIALS)
    ap.add_argument("--keep", action="store_true", help="保留中间 study.db")
    ap.add_argument("--legacy-seed", action="store_true",
                    help="把常数种子这个旧行为也跑一遍（④d 应当 FAIL）")
    args = ap.parse_args()

    if args.cmd == "sim":
        sim_worker(Path(args.db), args.combo, args.mode, args.max_trials,
                   args.legacy_seed)
        return 0

    WORK.mkdir(parents=True, exist_ok=True)
    verdicts = [check_trial0()]
    try:
        verdicts.append(check_sampling())
        verdicts.append(check_frozen(legacy=False))
        if args.legacy_seed:
            # 不进 verdicts：它**就该**失败。有牙与否只影响可信度，不影响放行。
            lname, lok, lwhy = check_frozen(legacy=True)
            print()
            print(f"  [{'仍会锁死（符合预期，检查有牙）' if not lok else '居然也通过了 —— 说明 ④d 照不到这个缺陷，不能当数'}]")
            if not lok:
                print(f"      失败原因: {lwhy}")
    finally:
        if not args.keep:
            shutil.rmtree(WORK, ignore_errors=True)
        else:
            print(f"\n[keep] 中间 study.db 留在 {WORK}")
    verdicts.append(check_gpu())

    head("汇总")
    for name, ok, why in verdicts:
        print(f"  [{say(ok)}] {name}" + ("" if ok else f"\n         —— {why}"))
    ok_all = all(v[1] for v in verdicts)
    print()
    print("全部通过。" if ok_all
          else "有检查未通过，**不要启动正式搜索**。")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
