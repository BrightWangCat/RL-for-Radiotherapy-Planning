from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess.state_frames_openkbp.openkbp_io import (
    _read_index_and_optional_value,
    _infer_shape_from_max_index,
)

def dense_from_idx_val(idx: np.ndarray, val: np.ndarray, shape, order: str) -> np.ndarray:
    nvox = int(np.prod(shape))
    out = np.zeros(nvox, dtype=np.float32)
    out[idx] = val.astype(np.float32, copy=False)
    return out.reshape(shape, order=order)

def mask_from_idx(idx: np.ndarray, shape, order: str) -> np.ndarray:
    nvox = int(np.prod(shape))
    out = np.zeros(nvox, dtype=np.uint8)
    out[idx] = 1
    return out.reshape(shape, order=order).astype(bool)

def save_img(img2d: np.ndarray, title: str, path: Path):
    plt.figure()
    plt.imshow(img2d.T, origin="lower")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="preprocess_outputs/step1b_order_check")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    outdir = Path(args.outdir) / case_dir.name

    ct_idx, ct_val = _read_index_and_optional_value(case_dir / "ct.csv")
    dose_idx, dose_val = _read_index_and_optional_value(case_dir / "dose.csv")
    pm_idx, _ = _read_index_and_optional_value(case_dir / "possible_dose_mask.csv")
    ptv_idx, _ = _read_index_and_optional_value(case_dir / "PTV70.csv")

    max_idx = int(max(ct_idx.max(), dose_idx.max(), pm_idx.max(), ptv_idx.max()))
    shape = _infer_shape_from_max_index(max_idx)
    z = shape[2] // 2

    for order in ["C", "F"]:
        ct = dense_from_idx_val(ct_idx, ct_val, shape, order)
        dose = dense_from_idx_val(dose_idx, dose_val, shape, order)
        pm = mask_from_idx(pm_idx, shape, order)
        ptv = mask_from_idx(ptv_idx, shape, order)

        save_img(ct[:, :, z],   f"CT mid z={z} order={order}",   outdir / f"ct_mid_{order}.png")
        save_img(dose[:, :, z], f"Dose mid z={z} order={order}", outdir / f"dose_mid_{order}.png")
        save_img(pm[:, :, z].astype(np.uint8),  f"PossibleMask mid z={z} order={order}", outdir / f"pm_mid_{order}.png")
        save_img(ptv[:, :, z].astype(np.uint8), f"PTV70 mid z={z} order={order}", outdir / f"ptv70_mid_{order}.png")

    print(f"[INFO] shape={shape}, wrote to {outdir}")

if __name__ == "__main__":
    main()
