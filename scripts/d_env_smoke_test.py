import numpy as np
from rlfplan.openkbp_case import OpenKBPCase
from rlfplan.env_openkbp_grouped import OpenKBPGroupedEnv

ROOT = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
CASE = "pt_241"

case = OpenKBPCase.load(ROOT, CASE)
env = OpenKBPGroupedEnv(case, K=64, max_steps=20)

obs, info = env.reset(seed=0)
print("reset obs:", obs, "calib_gain:", info.get("calibration_gain"))

ret = 0.0
for i in range(20):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    ret += r
    if i < 3 or i == 19:
        bs = info.get("brainstem_mean", float("nan"))
        bs_ref = info.get("brainstem_ref_mean", float("nan"))
        sc = info.get("spinalcord_mean", float("nan"))
        sc_ref = info.get("spinalcord_ref_mean", float("nan"))
        bs_excess = max(0.0, bs - bs_ref) if np.isfinite(bs) and np.isfinite(bs_ref) else float("nan")
        sc_excess = max(0.0, sc - sc_ref) if np.isfinite(sc) and np.isfinite(sc_ref) else float("nan")

        print(
            f"step {i+1:02d} r={r:.4f} obs={obs} "
            f"err_norm={info['err_norm']:.4f} oar_norm={info['oar_pen_norm']:.4f} "
            f"BS={bs:.3f}/{bs_ref:.3f} SC={sc:.3f}/{sc_ref:.3f}"
        )

    if term or trunc:
        break

print("episode return:", ret)
print("done:", term, trunc)
