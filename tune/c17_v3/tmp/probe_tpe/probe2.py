"""隔离探针 v2：历史直接从真实 HE+Ki67 study.db 取**全精度**值。"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent.parent/"scripts"))
import optuna
from optuna.distributions import FloatDistribution, CategoricalDistribution
optuna.logging.set_verbosity(optuna.logging.ERROR)
import optuna_v3 as V

real = optuna.load_study(study_name="c17_v3_HE_Ki67", storage=f"sqlite:///{HERE/'HE_Ki67'/'study.db'}")
HIST = [(t.value, dict(V.trial_params_of(t))) for t in sorted(real.trials, key=lambda x:x.number)[:8]]
T7 = V.config_key(HIST[7][1])
DISTS = {n: (FloatDistribution(s[1],s[2],log=s[3]) if s[0]=="float" else CategoricalDistribution(s[1]))
         for n,s in V.SPACE.items()}

def build(db, sd):
    p = Path(db)
    for f in p.parent.glob(p.name+"*"): f.unlink()
    st = optuna.create_study(study_name="probe", storage=f"sqlite:///{p}", direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sd, n_startup_trials=V.N_STARTUP, multivariate=True))
    for v, prm in HIST:
        st.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE,
            value=v, distributions=DISTS, params=prm))
    return st

def propose(db, sd):
    t = build(db, sd).ask()
    return V.sample_params(t)

base = Path("tmp/probe_tpe")
print(f"真实 #7 全精度: {[f'{k}={v!r}' for k,v in sorted(HIST[7][1].items())]}\n")

a = propose(base/"A.db", V.SAMPLER_SEED)
b = propose(base/"B.db", V.SAMPLER_SEED)          # 同 seed、同 history，两个独立 study
c = propose(base/"C.db", V.SAMPLER_SEED + 8)      # 每 trial 换 seed
d = propose(base/"D.db", V.SAMPLER_SEED + 9)

print(f"A (seed=2026)      lr={a['learning_rate']!r}\n                   wd={a['weight_decay']!r} "
      f"do={a['dropout']} rs={a['residual_scale']} tau={a['tau']} mom={a['prototype_momentum']}")
print(f"B (seed=2026)      lr={b['learning_rate']!r}   <-- 另一个独立 study")
print(f"C (seed=2026+8)    lr={c['learning_rate']!r}")
print(f"D (seed=2026+9)    lr={d['learning_rate']!r}\n")
print(f"① A 与 #7 逐位相同                  : {V.config_key(a)==T7}")
print(f"② A 与 B 逐位相同（RNG 重放）        : {V.config_key(a)==V.config_key(b)}")
print(f"③ C 与 #7 不同（换 seed 逃出不动点）  : {V.config_key(c)!=T7}")
print(f"④ D 与 C 不同（seed 随 trial 走）     : {V.config_key(d)!=V.config_key(c)}")
print(f"⑤ C/D 仍在搜索空间内                : ", end="")
try:
    V.assert_in_space(c, "C"); V.assert_in_space(d, "D"); print("是")
except AssertionError as e: print(f"否: {e}")
