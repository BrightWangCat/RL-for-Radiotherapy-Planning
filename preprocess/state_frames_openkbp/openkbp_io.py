from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except Exception:
    sparse = None


OPENKBP_EXPECTED_SHAPE = (128, 128, 128)  # per OpenKBP dataset description (axial 128x128, volume 128 slices)


def _read_csv_flat(path: Path, dtype=np.float32) -> np.ndarray:
    """
    Read OpenKBP-style CSV which is typically a single column (no header) flattened vector.
    Robust to accidental headers.
    """
    df = pd.read_csv(path, header=None)
    arr = df.values.reshape(-1).astype(dtype, copy=False)
    return arr


def list_case_dirs(root: Path) -> List[Path]:
    root = Path(root)
    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda p: p.name)
    return case_dirs


def infer_and_reshape(vec: np.ndarray, shape=OPENKBP_EXPECTED_SHAPE) -> np.ndarray:
    expected = int(np.prod(shape))
    if vec.size != expected:
        raise ValueError(
            f"Vector length mismatch: got {vec.size}, expected {expected} for shape={shape}. "
            f"Please confirm dataset preprocessing / shape."
        )
    return vec.reshape(shape)


@dataclass
class OpenKBPCasedata:
    case_dir: Path
    ct: np.ndarray                      # (128,128,128), float
    dose: Optional[np.ndarray]          # (128,128,128), float or None
    possible_dose_mask: Optional[np.ndarray]  # (128,128,128), bool or None
    voxel_dimensions: np.ndarray        # raw numbers from voxel_dimensions.csv (often spacing)
    structures: Dict[str, np.ndarray]   # name -> (128,128,128) bool
    beamlet_indices: Optional[pd.DataFrame]   # may be None
    dij: Optional[object]              # scipy sparse matrix if loaded


def load_case(case_dir: Path, load_dij: bool = False) -> OpenKBPCasedata:
    case_dir = Path(case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    # Mandatory-ish
    ct_path = case_dir / "ct.csv"
    voxdim_path = case_dir / "voxel_dimensions.csv"

    ct_flat = _read_csv_flat(ct_path, dtype=np.float32)
    ct = infer_and_reshape(ct_flat)

    voxdim = _read_csv_flat(voxdim_path, dtype=np.float32)

    # Optional files (some datasets can omit depending on split)
    dose = None
    dose_path = case_dir / "dose.csv"
    if dose_path.exists():
        dose_flat = _read_csv_flat(dose_path, dtype=np.float32)
        dose = infer_and_reshape(dose_flat)

    possible_mask = None
    pm_path = case_dir / "possible_dose_mask.csv"
    if pm_path.exists():
        pm_flat = _read_csv_flat(pm_path, dtype=np.float32)
        pm = infer_and_reshape(pm_flat)
        possible_mask = pm > 0.5

    beamlet_indices = None
    bi_path = case_dir / "beamlet_indices.csv"
    if bi_path.exists():
        beamlet_indices = pd.read_csv(bi_path)

    # Structures: all csv except known ones
    known = {"ct.csv", "dose.csv", "possible_dose_mask.csv", "voxel_dimensions.csv", "beamlet_indices.csv"}
    structures: Dict[str, np.ndarray] = {}
    for csv_path in sorted(case_dir.glob("*.csv")):
        if csv_path.name in known:
            continue
        name = csv_path.stem
        mask_flat = _read_csv_flat(csv_path, dtype=np.float32)
        mask = infer_and_reshape(mask_flat) > 0.5
        structures[name] = mask

    dij_obj = None
    if load_dij:
        if sparse is None:
            raise RuntimeError("scipy is required to load dij.npz, but scipy import failed.")
        dij_path = case_dir / "dij.npz"
        if not dij_path.exists():
            raise FileNotFoundError(dij_path)
        dij_obj = sparse.load_npz(dij_path)  # typically csr_matrix

    return OpenKBPCasedata(
        case_dir=case_dir,
        ct=ct,
        dose=dose,
        possible_dose_mask=possible_mask,
        voxel_dimensions=voxdim,
        structures=structures,
        beamlet_indices=beamlet_indices,
        dij=dij_obj,
    )
