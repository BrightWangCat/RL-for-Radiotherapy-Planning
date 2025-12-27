from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess.state_frames_openkbp.openkbp_io import list_case_dirs, load_case


def _pick_mid_slice(vol: np.ndarray) -> int:
    return vol.shape[2] // 2


def _save_slice(img2d: np.ndarray, title: str, out_png: Path):
    plt.figure()
    plt.imshow(img2d.T, origin="lower")  # transpose to show x-y in common orientation
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_mask(mask2d: np.ndarray, title: str, out_png: Path):
    plt.figure()
    plt.imshow(mask2d.T.astype(np.uint8), origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Dataset root, e.g. .../open-kbp-opt-data/reference-plans")
    ap.add_argument("--case", type=str, default="",
                    help="Case folder name. If empty, use the first case.")
    ap.add_argument("--outdir", type=str, default="preprocess_outputs/step1_inspect",
                    help="Where to write png outputs")
    ap.add_argument("--load-dij", action="store_true",
                    help="Also load dij.npz (can be big).")
    args = ap.parse_args()

    root = Path(args.root)
    case_dirs = list_case_dirs(root)
    if len(case_dirs) == 0:
        raise RuntimeError(f"No case directories found under: {root}")

    if args.case:
        case_dir = root / args.case
    else:
        case_dir = case_dirs[0]

    print(f"[INFO] root      = {root}")
    print(f"[INFO] case_dir  = {case_dir}")
    print(f"[INFO] num_cases = {len(case_dirs)}")
    print(f"[INFO] first_5_cases = {[p.name for p in case_dirs[:5]]}")

    data = load_case(case_dir, load_dij=args.load_dij)

    print("\n=== Basic tensors ===")
    print(f"ct.shape = {data.ct.shape}, dtype={data.ct.dtype}, min/max={data.ct.min():.3f}/{data.ct.max():.3f}")
    print(f"voxel_dimensions raw = {data.voxel_dimensions.tolist()} (len={len(data.voxel_dimensions)})")

    if data.dose is None:
        print("dose: None (dose.csv not found)")
    else:
        print(f"dose.shape = {data.dose.shape}, dtype={data.dose.dtype}, min/max={data.dose.min():.6f}/{data.dose.max():.6f}")

    if data.possible_dose_mask is None:
        print("possible_dose_mask: None (possible_dose_mask.csv not found)")
    else:
        u = np.unique(data.possible_dose_mask.astype(np.uint8))
        print(f"possible_dose_mask.shape = {data.possible_dose_mask.shape}, unique={u.tolist()}, "
              f"nonzero={int(data.possible_dose_mask.sum())}")

    print("\n=== Structures (masks) ===")
    print(f"num_structures = {len(data.structures)}")
    for k in sorted(data.structures.keys()):
        m = data.structures[k]
        print(f"{k:20s} sum={int(m.sum())}")

    print("\n=== Beamlets / Dij ===")
    if data.beamlet_indices is None:
        print("beamlet_indices: None (beamlet_indices.csv not found)")
    else:
        print(f"beamlet_indices.shape = {data.beamlet_indices.shape}")
        print("beamlet_indices.head():")
        print(data.beamlet_indices.head())

    if args.load_dij:
        dij = data.dij
        print(f"dij type={type(dij)} shape={dij.shape} nnz={getattr(dij,'nnz','?')}")

    # --- Visualization ---
    outdir = Path(args.outdir) / case_dir.name
    z = _pick_mid_slice(data.ct)

    _save_slice(data.ct[:, :, z], f"CT mid-slice z={z}", outdir / "ct_mid.png")

    if data.dose is not None:
        _save_slice(data.dose[:, :, z], f"Dose mid-slice z={z}", outdir / "dose_mid.png")

    # Prefer to visualize high-dose PTV if exists
    for ptv_name in ["PTV70", "PTV63", "PTV56"]:
        if ptv_name in data.structures:
            _save_mask(data.structures[ptv_name][:, :, z], f"{ptv_name} mask mid-slice z={z}", outdir / f"{ptv_name}_mid.png")
            break

    print(f"\n[INFO] Wrote PNGs to: {outdir}")


if __name__ == "__main__":
    main()
