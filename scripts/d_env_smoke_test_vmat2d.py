import numpy as np
from rlfplan.openkbp_case import OpenKBPCase
from rlfplan.env_openkbp_vmat2d import OpenKBPVMAT2DEnv

ROOT = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
CASE = "pt_241"

case = OpenKBPCase.load(ROOT, CASE)
env = OpenKBPVMAT2DEnv(case, K=64, max_steps=50, seed=0)

obs, info = env.reset(seed=0)
print("obs shape:", obs.shape, "dtype:", obs.dtype)
print("frame1 min/max:", obs[..., 0].min(), obs[..., 0].max())
print("frame2 min/max:", obs[..., 1].min(), obs[..., 1].max())
print("action_space:", env.action_space)
print("reset info:", {k: info[k] for k in ["case_id", "cp_idx", "theta_deg", "d0", "x1_mm", "x2_mm", "calibration_gain"]})

ret = 0.0
for i in range(10):
    a = env.action_space.sample()
    obs, r, term, trunc, inf = env.step(a)
    ret += r
    if i < 3 or i == 9:
        print(
            f"step {i+1:02d} a={a:02d} r={r:.4f} "
            f"cp={inf['cp_idx']:02d} theta={inf['theta_deg']:.2f} "
            f"err={inf['err_norm']:.4f} oar={inf['oar_pen_norm']:.4f} "
            f"d0={inf['d0']} x1={inf['x1_mm']} x2={inf['x2_mm']}"
        )
    if term or trunc:
        break

print("episode return:", ret, "done:", term, trunc)
