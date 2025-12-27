from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import rotate, binary_dilation

from preprocess.state_frames_openkbp.openkbp_io import load_case


def pick_z_by_mask_area(mask3d: np.ndarray) -> int:
    """Pick z where mask area is maximal."""
    areas = mask3d.reshape(mask3d.shape[0] * mask3d.shape[1], mask3d.shape[2]).sum(axis=0)
    return int(np.argmax(areas))


def safe_save(img2d: np.ndarray, title: str, out_png: Path, vmin=None, vmax=None, contour=None):
    plt.figure()
    im = plt.imshow(img2d.T, origin="lower", vmin=vmin, vmax=vmax)
    if contour is not None:
        plt.contour(contour.T.astype(np.uint8), levels=[0.5])
    plt.title(title)
    plt.tight_layout()
    # colorbar only if non-degenerate
    mn = float(np.nanmin(img2d))
    mx = float(np.nanmax(img2d))
    if np.isfinite(mn) and np.isfinite(mx) and (mx - mn) > 1e-8:
        plt.colorbar(im)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def build_objective_map(
    dose_shape,
    r: float,
    structures: dict[str, np.ndarray],
    defaults_gy: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      obj_map: float32, same shape as dose, objective value (normalized) for voxels that have objectives
      obj_mask: bool, objective-defined voxels (where obj_map is meaningful)
      ptv_union: bool, union of all PTVs (for boundary encoding)
    """
    obj_map = np.zeros(dose_shape, dtype=np.float32)
    obj_mask = np.zeros(dose_shape, dtype=bool)

    # PTV priority: high dose overrides lower dose if overlap
    ptv_specs = [("PTV70", 70.0), ("PTV63", 63.0), ("PTV56", 56.0)]
    ptv_union = np.zeros(dose_shape, dtype=bool)

    for name, presc in ptv_specs:
        if name not in structures:
            continue
        m = structures[name].astype(bool, copy=False)
        if m.sum() == 0:
            continue
        ptv_union |= m
        obj_map[m] = presc / (r + 1e-8)   # normalized objective value
        obj_mask |= m

    # OARs: use constant per-voxel objective value (normalized)
    # NOTE: This matches the paper’s concept of “planning objective value for each voxel”. :contentReference[oaicite:3]{index=3}
    for organ, lim_gy in defaults_gy.items():
        if organ not in structures:
            continue
        m = structures[organ].astype(bool, copy=False)
        if m.sum() == 0:
            continue
        obj_map[m] = lim_gy / (r + 1e-8)
        obj_mask |= m

    return obj_map, obj_mask, ptv_union


def boundary_ring(mask2d: np.ndarray, iters: int = 1) -> np.ndarray:
    """One-pixel-ish ring around mask, via dilation - mask."""
    dil = binary_dilation(mask2d, iterations=iters)
    ring = np.logical_and(dil, ~mask2d)
    return ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--case", type=str, default="pt_241")
    ap.add_argument("--outdir", type=str, default="preprocess_outputs/step3_state_frame1")
    ap.add_argument("--theta_deg", type=float, default=None, help="gantry angle in degrees; if None, use first angle index")
    ap.add_argument("--z", type=int, default=None, help="slice index; if None, pick z where PTV union area is maximal")
    args = ap.parse_args()

    case_dir = Path(args.root) / args.case
    data = load_case(case_dir, load_dij=False)

    if data.dose is None:
        raise RuntimeError("dose is None")

    dose = data.dose.astype(np.float32, copy=False)
    pmask = data.possible_dose_mask.astype(bool, copy=False) if data.possible_dose_mask is not None else np.ones_like(dose, dtype=bool)

    # normalize using PTV70 max dose (as you已验证正确)
    ptv70 = data.structures.get("PTV70", None)
    if ptv70 is None or ptv70.sum() == 0:
        raise RuntimeError("PTV70 missing/empty; later we can add fallback to PTV63/PTV56")
    dmax_ptv = float(dose[ptv70.astype(bool)].max())
    r = 0.926 * dmax_ptv  # paper definition :contentReference[oaicite:4]{index=4}
    dose_norm = (dose / (r + 1e-8)) * pmask

    # ---- Objective Map defaults (temporary) ----
    # These are common H&N constraints; we will replace with official open-kbp-opt objectives once you grep them from repo.
    defaults_gy = {
        "Brainstem": 54.0,
        "SpinalCord": 45.0,
        "LeftParotid": 26.0,
        "RightParotid": 26.0,
    }

    obj_map, obj_mask, ptv_union = build_objective_map(
        dose_shape=dose.shape,
        r=r,
        structures=data.structures,
        defaults_gy=defaults_gy,
    )

    # Apply objective mask rule from paper: voxels without objectives -> 0 :contentReference[oaicite:5]{index=5}
    frame1_raw = (dose_norm - obj_map) * obj_mask

    # choose z for visualization: prioritize PTV union
    if args.z is not None:
        z = args.z
    else:
        if ptv_union.sum() > 0:
            z = pick_z_by_mask_area(ptv_union.astype(np.uint8))
        else:
            z = pick_z_by_mask_area(pmask.astype(np.uint8))

    # choose theta
    if args.theta_deg is not None:
        theta = float(args.theta_deg)
    else:
        # beamlet_indices.angle in your case looks like an index (1,2,...) not degrees
        # For visualization we can start with 0 degrees; later we will map index->degrees once we inspect unique angles.
        theta = 0.0

    # rotate by -theta (paper) :contentReference[oaicite:6]{index=6}
    frame1_rot = rotate(frame1_raw[:, :, z], angle=-theta, reshape=False, order=1, mode="constant", cval=0.0)
    ptv_rot = rotate(ptv_union[:, :, z].astype(np.float32), angle=-theta, reshape=False, order=0, mode="constant", cval=0.0) > 0.5

    # rescale to uint8: (x + 1) * 50  (paper) :contentReference[oaicite:7]{index=7}
    frame1_u8 = np.clip((frame1_rot + 1.0) * 50.0, 0, 255).astype(np.uint8)

    # boundary voxels around PTV set to 255 :contentReference[oaicite:8]{index=8}
    ring = boundary_ring(ptv_rot, iters=1)
    frame1_u8[ring] = 255

    outdir = Path(args.outdir) / args.case
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] case={args.case}, shape={dose.shape}")
    print(f"[INFO] dmax_ptv70={dmax_ptv:.6f} Gy, r={r:.6f}, dose_norm_max_ptv70={float(dose_norm[ptv70.astype(bool)].max()):.6f}")
    print(f"[INFO] z={z}, ptv_union_count(z)={int(ptv_union[:, :, z].sum())}, obj_mask_count(z)={int(obj_mask[:, :, z].sum())}")
    print(f"[INFO] theta_deg={theta}")

    # Visualizations (three required + one extra)
    safe_save(dose_norm[:, :, z], f"Normalized Dose z={z}", outdir / f"1_normdose_z{z}.png", vmin=0, vmax=1.08)
    safe_save(obj_map[:, :, z], f"Objective Map z={z}", outdir / f"2_objmap_z{z}.png")
    safe_save(frame1_rot, f"StateFrame1 raw (rot -{theta}°) z={z}", outdir / f"3_stateframe1_raw_z{z}_th{int(theta)}.png")
    safe_save(frame1_u8.astype(np.float32), f"StateFrame1 uint8 z={z} th={int(theta)} (boundary=255)", outdir / f"4_stateframe1_u8_z{z}_th{int(theta)}.png")

    print(f"[INFO] wrote PNGs to {outdir}")


if __name__ == "__main__":
    main()
