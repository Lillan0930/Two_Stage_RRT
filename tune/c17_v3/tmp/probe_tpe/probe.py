"""隔离探针：复现 HE+Ki67 的不动点，并验证「每 trial 换 seed」能否破解。
不碰 GPU、不碰正式 study。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE.parent.parent/"scripts"))
import optuna
from optuna.distributions import FloatDistribution, CategoricalDistribution
optuna.logging.set_verbosity(optuna.logging.WARNING)
import optuna_v3 as V

HIST = [  # HE+Ki67 的 #0..#7，逐位照抄 study.db
 (0.9407238904, 2.420000000000e-04, 7.590000000000e-06, 0.05, 0.1,   0.2,  0.99 ),
 (0.9261652007, 7.407200227376e-05, 6.699208178750e-06, 0.35, 0.2,   0.3,  0.995),
 (0.9297644323, 1.523592848195e-04, 3.829392623696e-05, 0.25, 0.05,  0.15, 0.995),
 (0.9348599737, 1.551324624212e-04, 2.268455918431e-05, 0.1,  0.05,  0.2,  0.98 ),
 (0.9006066121, 1.438397880779e-04, 2.086219006069e-06, 0.05, 0.05,  0.2,  0.99 ),
 (0.9410676372, 2.420357261899e-04, 2.035375483160e-05, 0.15, 0.075, 0.2,  0.99 ),
 (0.9403599232, 2.420451101999e-04, 3.602990930365e-05, 0.15, 0.075, 0.2,  0.99 ),
 (0.9420281064, 2.920284813305e-04, 1.782647785721e-06, 0.15, 0.2,   0.2,  0.98 ),
]
T7 = V.config_key({"learning_rate":HIST[7][1],"weight_decay":HIST[7][2],"dropout":HIST[7][3],
                   "residual_scale":HIST[7][4],"tau":HIST[7][5],"prototype_momentum":HIST[7][6]})
DISTS = {n: (FloatDistribution(s[1],s[2],log=s[3]) if s[0]=="float" else CategoricalDistribution(s[1]))
         for n,s in V.SPACE.items()}

def build(db, seed_of_sampler):
    p = Path(db)
    for f in p.parent.glob(p.name+"*"): f.unlink()
    st = optuna.create_study(study_name="probe", storage=f"sqlite:///{p}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed_of_sampler,
                                           n_startup_trials=V.N_STARTUP, multivariate=True))
    for v, lr, wd, do, rs, tau, mom in HIST:
        st.add_trial(optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE, value=v, distributions=DISTS,
            params={"learning_rate":lr,"weight_decay":wd,"dropout":do,
                    "residual_scale":rs,"tau":tau,"prototype_momentum":mom}))
    return st

def propose(st):
    t = st.ask()
    return V.sample_params(t), t.number

base = Path("tmp/probe_tpe")
print(f"固定点（#7）: {tuple(x[1] for x in T7)}\n")
for label, seed_of_sampler in [("现在：seed=2026 恒定", V.SAMPLER_SEED),
                               ("修复：seed=2026+下一个 trial 号", V.SAMPLER_SEED + 8),
                               ("修复：seed=2026+9", V.SAMPLER_SEED + 9)]:
    p, num = propose(build(base/f"s{seed_of_sampler}.db", seed_of_sampler))
    k = V.config_key(p)
    print(f"{label}")
    print(f"   → lr={p['learning_rate']:.12e} wd={p['weight_decay']:.12e} do={p['dropout']} "
          f"rs={p['residual_scale']} tau={p['tau']} mom={p['prototype_momentum']}")
    print(f"   {'*** 与 #7 逐位相同（不动点）***' if k==T7 else '与 #7 不同 —— 逃出不动点'}\n")
