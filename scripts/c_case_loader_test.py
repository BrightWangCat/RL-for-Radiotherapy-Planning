# scripts/c_case_loader_test.py
import numpy as np
from rlfplan.openkbp_case import OpenKBPCase

ROOT = "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans"
CASE = "pt_241"

case = OpenKBPCase.load(ROOT, CASE)

print("case:", case.case_id)
print("A shape:", case.A.shape, "nnz:", case.A.nnz, "dtype:", case.A.dtype)
print("dose_ref:", case.dose_ref.shape, "min/max/mean:", float(case.dose_ref.min()), float(case.dose_ref.max()), float(case.dose_ref.mean()))

# structure report
for k in sorted(case.struct_masks.keys()):
    print(f"{k:12s} in_possible={case.struct_in_possible_counts[k]:6d} / global={case.struct_global_counts[k]:6d}")

# simple numeric check: a single sparse matvec (random beamlets) — 只跑一次
rng = np.random.default_rng(0)
w = rng.random(case.n_beamlets, dtype=np.float32)
d = case.A @ w
print("matvec dose:", d.shape, "min/max/mean:", float(d.min()), float(d.max()), float(d.mean()))

# MSE vs reference (purely sanity; not clinical)
mse = float(np.mean((d - case.dose_ref) ** 2))
mae = float(np.mean(np.abs(d - case.dose_ref)))
print("sanity error vs ref dose: MSE=", mse, " MAE=", mae)

# Ensure PTVs exist and have nonzero voxels
for ptv in ["PTV56","PTV63","PTV70"]:
    if ptv in case.struct_masks:
        assert case.struct_masks[ptv].sum() > 0, f"{ptv} empty in possible-dose space"
print("OK")
