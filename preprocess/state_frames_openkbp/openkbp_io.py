from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import sparse
except Exception:
    sparse = None


XY = 128 * 128


def _read_index_and_optional_value(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read OpenKBP-Opt sparse CSVs:
      - ct.csv / dose.csv: (index, value) with header like ",data"
      - masks (structures / possible_dose_mask): (index, <empty>) with same header

    Returns:
      idx: int64, shape (N,)
      val: float32, shape (N,) or None if value column is empty
    """
    df = pd.read_csv(path, low_memory=False)
    num = df.apply(pd.to_numeric, errors="coerce")

    # choose index column: integer-like, most non-NaN
    def is_int_like(s: pd.Series) -> bool:
        s2 = s.dropna()
        if s2.empty:
            return False
        a = s2.to_numpy()
        return np.all(np.isclose(a, np.round(a)))

    idx_col = None
    best_cnt = -1
    for c in num.columns:
        s = num[c]
        cnt = int(s.notna().sum())
        if cnt <= 0:
            continue
        if not is_int_like(s):
            continue
        if cnt > best_cnt:
            best_cnt = cnt
            idx_col = c

    if idx_col is None:
        raise ValueError(f"No integer-like index column found in {path}")

    # value column: among remaining columns pick one with most numeric entries
    other_cols = [c for c in num.columns if c != idx_col]
    val_col = None
    val_cnt = 0
    for c in other_cols:
        cnt = int(num[c].notna().sum())
        if cnt > val_cnt:
            val_cnt = cnt
            val_col = c

    idx = num[idx_col].dropna().to_numpy(dtype=np.int64)

    val = None
    if val_col is not None and val_cnt > 0:
        # align pairs (drop rows where either is NaN)
        pair = num[[idx_col, val_col]].dropna()
        idx = pair[idx_col].to_numpy(dtype=np.int64)
        val = pair[val_col].to_numpy(dtype=np.float32)

    # normalize to 0-based if it looks 1-based (no zeros, min>=1)
    if idx.size > 0 and idx.min() >= 1 and not np.any(idx == 0):
        idx = idx - 1

    return idx, val


def _infer_shape_from_max_index(max_idx: int) -> Tuple[int, int, int]:
    """
    Infer (128,128,z) from maximum voxel index present in sparse files.
    We set nvox = XY * ceil((max_idx+1)/XY) to make it reshaped cleanly.
    """
    n = int(max_idx) + 1
    z = int(np.ceil(n / XY))
    return (128, 128, z)


def _dense_mask_from_indices(idx: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    nvox = int(np.prod(shape))
    m = np.zeros(nvox, dtype=bool)
    if idx.size > 0:
        if idx.max() >= nvox:
            raise ValueError(f"Index out of range: idx.max={int(idx.max())} >= nvox={nvox}")
        m[idx] = True
    return m.reshape(shape, order="F")


def _dense_values_from_index_value(idx: np.ndarray, val: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    nvox = int(np.prod(shape))
    out = np.zeros(nvox, dtype=np.float32)
    if idx.size > 0:
        if idx.max() >= nvox:
            raise ValueError(f"Index out of range: idx.max={int(idx.max())} >= nvox={nvox}")
        out[idx] = val.astype(np.float32, copy=False)
    return out.reshape(shape, order="F")


@dataclass
class OpenKBPCasedata:
    case_dir: Path
    vol_shape: Tuple[int, int, int]          # (128,128,z)
    ct: np.ndarray                            # float32 dense
    dose: Optional[np.ndarray]                # float32 dense or None
    possible_dose_mask: Optional[np.ndarray]  # bool dense or None
    voxel_dimensions: np.ndarray
    structures: Dict[str, np.ndarray]         # bool dense masks
    beamlet_indices: Optional[pd.DataFrame]
    dij: Optional[object]


def list_case_dirs(root: Path) -> List[Path]:
    root = Path(root)
    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted(case_dirs, key=lambda p: p.name)


def load_case(case_dir: Path, load_dij: bool = False) -> OpenKBPCasedata:
    case_dir = Path(case_dir)

    ct_path = case_dir / "ct.csv"
    dose_path = case_dir / "dose.csv"
    pm_path = case_dir / "possible_dose_mask.csv"
    voxdim_path = case_dir / "voxel_dimensions.csv"
    bi_path = case_dir / "beamlet_indices.csv"

    # 1) Read sparse sources FIRST to determine a safe global volume size
    ct_idx, ct_val = _read_index_and_optional_value(ct_path)
    if ct_val is None:
        raise ValueError(f"ct.csv should have (index,value) but value column seems empty: {ct_path}")
    max_idx = int(ct_idx.max()) if ct_idx.size > 0 else 0

    pm_idx = None
    if pm_path.exists():
        pm_idx, _ = _read_index_and_optional_value(pm_path)
        if pm_idx.size > 0:
            max_idx = max(max_idx, int(pm_idx.max()))

    dose_idx = None
    dose_val = None
    if dose_path.exists():
        dose_idx, dose_val = _read_index_and_optional_value(dose_path)
        if dose_val is None:
            raise ValueError(f"dose.csv should have (index,value) but value column seems empty: {dose_path}")
        if dose_idx.size > 0:
            max_idx = max(max_idx, int(dose_idx.max()))

    # 2) Infer shape from the maximum index across ct/dose/mask
    vol_shape = _infer_shape_from_max_index(max_idx)

    # 3) Densify CT / Dose / possible mask into that shape
    ct = _dense_values_from_index_value(ct_idx, ct_val, vol_shape)

    dose = None
    if dose_idx is not None and dose_val is not None:
        dose = _dense_values_from_index_value(dose_idx, dose_val, vol_shape)

    possible_mask = None
    if pm_idx is not None:
        possible_mask = _dense_mask_from_indices(pm_idx, vol_shape)

    # 4) voxel_dimensions (small dense list)
    vdf = pd.read_csv(voxdim_path, low_memory=False)
    vnum = vdf.apply(pd.to_numeric, errors="coerce")
    voxel_dimensions = vnum.stack().dropna().to_numpy(dtype=np.float32)

    # 5) beamlet_indices (table)
    beamlet_indices = None
    if bi_path.exists():
        beamlet_indices = pd.read_csv(bi_path, low_memory=False)

    # 6) Structures: indices-only masks (fit into the inferred shape)
    known = {
        "ct.csv", "dose.csv", "possible_dose_mask.csv",
        "voxel_dimensions.csv", "beamlet_indices.csv"
    }
    structures: Dict[str, np.ndarray] = {}
    for csv_path in sorted(case_dir.glob("*.csv")):
        if csv_path.name in known:
            continue
        name = csv_path.stem
        s_idx, _ = _read_index_and_optional_value(csv_path)
        structures[name] = _dense_mask_from_indices(s_idx, vol_shape)

    # 7) dij
    dij_obj = None
    if load_dij:
        if sparse is None:
            raise RuntimeError("scipy is required to load dij.npz, but scipy import failed.")
        dij_path = case_dir / "dij.npz"
        if not dij_path.exists():
            raise FileNotFoundError(dij_path)
        dij_obj = sparse.load_npz(dij_path)

    return OpenKBPCasedata(
        case_dir=case_dir,
        vol_shape=vol_shape,
        ct=ct.astype(np.float32, copy=False),
        dose=None if dose is None else dose.astype(np.float32, copy=False),
        possible_dose_mask=possible_mask,
        voxel_dimensions=voxel_dimensions,
        structures=structures,
        beamlet_indices=beamlet_indices,
        dij=dij_obj,
    )
