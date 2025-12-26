#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Normalize dose so that max dose in PTV equals 1.08 (paper definition).

Usage:
  python step1_normalize_dose.py \
    --case_dir /fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241 \
    --out /fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans/pt_241/dose_norm_1p08.npy
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
import numpy as np
from array import array




def read_numeric_csv_flat(path: Path, dtype=np.float32) -> np.ndarray:
    """
    Robust numeric CSV reader:
    - tolerates header lines like ",data"
    - tolerates trailing commas / empty fields
    - tolerates index,value (uses last numeric field in the line)
    Returns 1D float array.
    """
    vals = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if not parts:
                continue

            x = None
            # scan from right to left, pick the last numeric field
            for p in reversed(parts):
                try:
                    x = float(p)
                    break
                except ValueError:
                    continue

            if x is None:
                skipped += 1
                continue

            vals.append(x)

    if len(vals) == 0:
        raise RuntimeError(f"No numeric data parsed from {path}. Skipped={skipped} lines.")

    arr = np.asarray(vals, dtype=dtype).ravel()
    # optional: print skip info (kept minimal)
    if skipped > 0:
        print(f"[read_numeric_csv_flat] skipped {skipped} non-numeric/header lines in {path.name}")
    return arr


def read_index_csv(path: Path) -> np.ndarray:
    """
    Robust index CSV reader:
    - tolerates header lines like ",data"
    - tolerates trailing commas / empty fields
    - tolerates one or multiple integer fields per line
    Returns 1D int64 array.
    """
    idx_list = []
    skipped = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if not parts:
                continue

            line_has_int = False
            for p in parts:
                try:
                    # some files may contain "123.0"
                    v = int(float(p))
                    idx_list.append(v)
                    line_has_int = True
                except ValueError:
                    continue

            if not line_has_int:
                skipped += 1

    if len(idx_list) == 0:
        raise RuntimeError(f"No indices parsed from {path}. Skipped={skipped} lines.")

    if skipped > 0:
        print(f"[read_index_csv] skipped {skipped} non-numeric/header lines in {path.name}")
    return np.asarray(idx_list, dtype=np.int64)



def infer_zero_based(indices: np.ndarray, n_vox: int) -> tuple[np.ndarray, str]:
    """
    Heuristic to determine whether indices are 0-based or 1-based and convert to 0-based.
    Returns (converted_indices, note_string).
    """
    if indices.size == 0:
        return indices, "empty indices"

    mn = int(indices.min())
    mx = int(indices.max())

    # If already looks 0-based
    if mn == 0 and mx <= n_vox - 1:
        return indices, "indices appear 0-based (min=0)"

    # If looks 1-based (common in some exports)
    if mn >= 1 and mx <= n_vox:
        # strong signal: mx==n_vox or mn==1
        if mx == n_vox or mn == 1:
            return indices - 1, "indices converted from 1-based to 0-based"
        # ambiguous but still plausible
        return indices - 1, "indices likely 1-based; converted to 0-based (heuristic)"

    # Otherwise do not convert; but warn.
    return indices, f"WARNING: indices range [{mn},{mx}] not compatible with n_vox={n_vox}; left unchanged"


def collect_ptv_indices(case_dir: Path) -> tuple[np.ndarray, list[Path], list[str]]:
    ptv_files = sorted([p for p in case_dir.glob("PTV*.csv") if p.is_file()])
    if not ptv_files:
        raise FileNotFoundError(f"No PTV*.csv found under {case_dir}")

    return ptv_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    out_path = Path(args.out)

    dose_path = case_dir / "dose.csv"
    if not dose_path.exists():
        raise FileNotFoundError(f"Missing dose.csv: {dose_path}")

    dose = read_numeric_csv_flat(dose_path, dtype=np.float32)
    n_vox = int(dose.size)

    ptv_files = collect_ptv_indices(case_dir)

    ptv_indices_all = []
    notes = []
    for f in ptv_files:
        idx = read_index_csv(f)
        idx0, note = infer_zero_based(idx, n_vox)
        # filter any out-of-range safely
        idx0 = idx0[(idx0 >= 0) & (idx0 < n_vox)]
        idx0 = np.unique(idx0)
        ptv_indices_all.append(idx0)
        notes.append(f"{f.name}: {note}, kept={idx0.size}")

    ptv_idx = np.unique(np.concatenate(ptv_indices_all, axis=0))
    if ptv_idx.size == 0:
        raise RuntimeError("PTV index union is empty after filtering; cannot normalize.")

    d_ptv_max = float(np.max(dose[ptv_idx]))
    if not np.isfinite(d_ptv_max) or d_ptv_max <= 0:
        raise RuntimeError(f"Invalid d_ptv_max={d_ptv_max}")

    scale = 1.08 / d_ptv_max
    dose_norm = dose * np.float32(scale)

    # Verification numbers
    dnorm_ptv_max = float(np.max(dose_norm[ptv_idx]))
    dnorm_min = float(np.min(dose_norm))
    dnorm_max = float(np.max(dose_norm))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), dose_norm.astype(np.float32, copy=False))

    print("=== Step1: Normalized Dose (max PTV -> 1.08) ===")
    print(f"case_dir: {case_dir}")
    print(f"dose.csv voxels: N={n_vox}")
    print(f"dose raw min/max: {float(dose.min()):.6g} / {float(dose.max()):.6g}")
    print(f"PTV files: {[p.name for p in ptv_files]}")
    for s in notes:
        print("  -", s)
    print(f"PTV union voxels: {ptv_idx.size}")
    print(f"D_PTVmax (raw): {d_ptv_max:.6g}")
    print(f"scale = 1.08 / D_PTVmax = {scale:.9g}")
    print(f"dose_norm min/max: {dnorm_min:.6g} / {dnorm_max:.6g}")
    print(f"max(dose_norm[PTV]) should be ~1.08 => {dnorm_ptv_max:.6g}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
