from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess.state_frames_openkbp.openkbp_io import load_case


def pick_nonempty_z(mask3d: np.ndarray) -> int:
    """Pick a z slice with maximum mask area (avoid empty slices)."""
    areas = mask3d.reshape(mask3d.shape[0] * mask3d.shape[1], mask3d.shape[2]).sum(axis=0)
    return int(np.argmax(areas))


def safe_imshow_with_colorbar(img2d: np.ndarray, title: str, out_png: Path, contour: np.ndarray | None = None):
    plt.figure()
    im = plt.imshow(img2d.T, origin="lower")
    if contour is not None:
        plt.contour(contour.T.astype(np.uint8), levels=[0.5])
    plt.title(title)
    plt.tight_layout()

    # only add colorbar if range is not degenerate
    vmin = float(np.nanmin(img2d))
    vmax = float(np.nanmax(img2d))
    if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 1e-8:
        plt.colorbar(im)
    else:
        # no colorbar; still save the figure
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


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
    dose_norm = dose_norm * pmask  # mask外归零

    # choose z
    if args.z is not None:
        z = args.z
    else:
        # pick a slice where mask is most present
        z = pick_nonempty_z(pmask)

    pmask_count = int(pmask[:, :, z].sum())
    ptv_count = int(ptv70[:, :, z].sum())
    slice_min = float(np.nanmin(dose_norm[:, :, z]))
    slice_max = float(np.nanmax(dose_norm[:, :, z]))

    print(f"[INFO] case={args.case}")
    print(f"[INFO] dose max overall={dose.max():.6f} Gy, dmax_ptv70={dmax_ptv:.6f} Gy")
    print(f"[INFO] r=0.926*dmax_ptv70={r:.6f}")
    print(f"[INFO] dose_norm max in ptv70={float(dose_norm[ptv70].max()):.6f} (target ~1.08)")
    print(f"[INFO] dose_norm min/max overall={float(dose_norm.min()):.6f}/{float(dose_norm.max()):.6f}")
    print(f"[INFO] chosen z={z}, pmask_count={pmask_count}, ptv70_count={ptv_count}, slice_min/max={slice_min:.6f}/{slice_max:.6f}")

    outdir = Path(args.outdir) / args.case
    outdir.mkdir(parents=True, exist_ok=True)

    safe_imshow_with_colorbar(
        dose_norm[:, :, z],
        title=f"Normalized Dose z={z}",
        out_png=outdir / f"normalized_dose_z{z}.png",
        contour=None,
    )
    safe_imshow_with_colorbar(
        dose_norm[:, :, z],
        title=f"Normalized Dose + PTV70 contour z={z}",
        out_png=outdir / f"normalized_dose_with_ptv70_z{z}.png",
        contour=ptv70[:, :, z],
    )

    print(f"[INFO] wrote PNGs to: {outdir}")


if __name__ == "__main__":
    main()
