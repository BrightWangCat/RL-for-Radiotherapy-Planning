from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess.state_frames_openkbp.openkbp_io import load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--case", type=str, default="pt_241")
    ap.add_argument("--outdir", type=str, default="preprocess_outputs/step2_normalized_dose")
    ap.add_argument("--z", type=int, default=None)
    args = ap.parse_args()

    case_dir = Path(args.root) / args.case
    data = load_case(case_dir, load_dij=False)

    if data.dose is None:
        raise RuntimeError("dose.csv not found / dose is None for this case.")

    dose = data.dose.astype(np.float32, copy=False)

    if data.possible_dose_mask is None:
        pmask = np.ones_like(dose, dtype=bool)
    else:
        pmask = data.possible_dose_mask.astype(bool, copy=False)

    if "PTV70" not in data.structures:
        raise RuntimeError("PTV70 not found in structures. Available: " + ",".join(sorted(data.structures.keys())))
    ptv70 = data.structures["PTV70"].astype(bool, copy=False)
    if ptv70.sum() == 0:
        raise RuntimeError("PTV70 mask is empty (sum==0).")

    dmax_ptv = float(dose[ptv70].max())
    r = 0.926 * dmax_ptv
    dose_norm = dose / (r + 1e-8)

    # mask 外归零（与后续状态构造一致）
    dose_norm = dose_norm * pmask

    z = args.z if args.z is not None else dose.shape[2] // 2

    print(f"[INFO] case={args.case}")
    print(f"[INFO] dose max overall={dose.max():.6f} Gy, dmax_ptv70={dmax_ptv:.6f} Gy")
    print(f"[INFO] r=0.926*dmax_ptv70={r:.6f}")
    print(f"[INFO] dose_norm max in ptv70={float(dose_norm[ptv70].max()):.6f} (target ~1.08)")
    print(f"[INFO] dose_norm min/max overall={float(dose_norm.min()):.6f}/{float(dose_norm.max()):.6f}")

    outdir = Path(args.outdir) / args.case
    outdir.mkdir(parents=True, exist_ok=True)

    # 可视化：normalized dose
    plt.figure()
    plt.imshow(dose_norm[:, :, z].T, origin="lower")
    plt.colorbar()
    plt.title(f"Normalized Dose mid-slice z={z}")
    plt.tight_layout()
    out_png = outdir / f"normalized_dose_z{z}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    # 同时画一张 PTV70 叠加轮廓，便于 sanity check
    plt.figure()
    plt.imshow(dose_norm[:, :, z].T, origin="lower")
    plt.contour(ptv70[:, :, z].T.astype(np.uint8), levels=[0.5])
    plt.colorbar()
    plt.title(f"Normalized Dose + PTV70 contour z={z}")
    plt.tight_layout()
    out_png2 = outdir / f"normalized_dose_with_ptv70_z{z}.png"
    plt.savefig(out_png2, dpi=200)
    plt.close()

    print(f"[INFO] wrote: {out_png}")
    print(f"[INFO] wrote: {out_png2}")


if __name__ == "__main__":
    main()
