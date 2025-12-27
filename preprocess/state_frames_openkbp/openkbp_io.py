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


def _read_numeric_series(path: Path, dtype=np.float32) -> np.ndarray:
    """
    Robust CSV reader for OpenKBP/OpenKBP-Opt style files.

    Handles:
      - Optional header (e.g., column name 'data')
      - Extra index columns
      - Mixed types (e.g., first row 'data')
    Strategy:
      - read with pandas
      - choose the column with most numeric entries (prefer column named 'data' if exists)
      - coerce to numeric, drop NaNs
    """
    df = pd.read_csv(path, low_memory=False)

    # Prefer explicit 'data' column if present
    if "data" in df.columns:
        ser = df["data"]
        num = pd.to_numeric(ser, errors="coerce").dropna().to_numpy(dtype=dtype)
        return num

    # Otherwise choose the column that yields the most numeric entries
    best = None
    best_count = -1
    for c in df.columns:
        num_c = pd.to_numeric(df[c], errors="coerce")
        cnt = int(num_c.notna().sum())
        if cnt > best_count:
            best_count = cnt
            best = num_c

    if best is None or best_count <= 0:
        raise ValueError(f"No numeric data detected in {path}")

    return best.dropna().to_numpy(dtype=dtype)


def _infer_volume_shape_from_ct_len(nvox: int) -> Tuple[int, int, int]:
    """
    In OpenKBP-style data, x=y=128 in the downsampled axial plane.
    z can vary (often 128) depending on dataset preprocessing.
    """
    xy = 128 * 128
    if nvox % xy != 0:
        raise ValueError(
            f"Cannot infer (128,128,z) volume shape because ct length={nvox} "
            f"is not divisible by 128*128={xy}. "
            f"Please run `head -n 5 ct.csv` and `python -c 'import pandas as pd; print(pd.read_csv(\"ct.csv\").head())'` "
            f"and share the output."
        )
    z = nvox // xy
    return (128, 128, z)


def _to_dense_mask(mask_data: np.ndarray, nvox: int) -> np.ndarray:
    """
    Convert mask representation to dense boolean mask of length nvox.

    If length == nvox => treat as dense 0/1 (or floats), threshold at 0.5
    Else => treat as sparse indices (int), set those indices True
    """
    if mask_data.size == nvox:
        return (mask_data > 0.5)

    # sparse indices
    idx = mask_data.astype(np.int64, copy=False)
    if idx.size == 0:
        return np.zeros(nvox, dtype=bool)

    # sanity check: indices should be within [0, nvox-1]
    mx = int(idx.max())
    mn = int(idx.min())
    if mn < 0 or mx >= nvox:
        raise ValueError(
            f"Mask indices out of range: min={mn}, max={mx}, but nvox={nvox}. "
            f"This suggests indices may be 1-based or the file format is different. "
            f"Please run `head -n 10 {Path(mask_data).__str__()}` (or head on the mask csv) and share."
        )

    dense = np.zeros(nvox, dtype=bool)
    dense[idx] = True
    return dense


def _to_dense_values(values_data: np.ndarray, nvox: int, fill_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert dose/ct-like data to dense float array of length nvox.

    If length == nvox => already dense.
    Else if fill_mask is provided and values length == fill_mask.sum() => fill into zeros at mask locations.
    """
    if values_data.size == nvox:
        return values_data.astype(np.float32, copy=False)

    if fill_mask is not None:
        k = int(fill_mask.sum())
        if values_data.size == k:
            out = np.zeros(nvox, dtype=np.float32)
            out[fill_mask] = values_data.astype(np.float32, copy=False)
            return out

    raise ValueError(
        f"Cannot densify values: values_len={values_data.size}, nvox={nvox}, "
        f"mask_sum={(int(fill_mask.sum()) if fill_mask is not None else None)}. "
        f"Likely file is stored as (index,value) pairs or another format. "
        f"Please share first 5 lines via `head -n 5 <file>`."
    )


def list_case_dirs(root: Path) -> List[Path]:
    root = Path(root)
    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    return sorted(case_dirs, key=lambda p: p.name)


@dataclass
class OpenKBPCasedata:
    case_dir: Path
    vol_shape: Tuple[int, int, int]          # (128,128,z)
    ct: np.ndarray                            # (128,128,z), float32
    dose: Optional[np.ndarray]                # (128,128,z), float32 or None
    possible_dose_mask: Optional[np.ndarray]  # (128,128,z), bool or None
    voxel_dimensions: np.ndarray              # numeric entries from voxel_dimensions.csv
    structures: Dict[str, np.ndarray]         # name -> (128,128,z) bool
    beamlet_indices: Optional[pd.DataFrame]
    dij: Optional[object]                     # scipy sparse matrix if loaded


def load_case(case_dir: Path, load_dij: bool = False) -> OpenKBPCasedata:
    case_dir = Path(case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)

    ct_path = case_dir / "ct.csv"
    voxdim_path = case_dir / "voxel_dimensions.csv"

    ct_vec_raw = _read_numeric_series(ct_path, dtype=np.float32)
    nvox = int(ct_vec_raw.size)
    vol_shape = _infer_volume_shape_from_ct_len(nvox)

    voxel_dimensions = _read_numeric_series(voxdim_path, dtype=np.float32)

    # possible_dose_mask can be dense or sparse indices
    possible_mask = None
    pm_path = case_dir / "possible_dose_mask.csv"
    if pm_path.exists():
        pm_raw = _read_numeric_series(pm_path, dtype=np.float32)
        pm_dense_1d = _to_dense_mask(pm_raw, nvox)
        possible_mask = pm_dense_1d.reshape(vol_shape)

    # densify ct (some variants store ct only on feasible mask; handle both)
    ct_dense_1d = _to_dense_values(ct_vec_raw, nvox, fill_mask=(possible_mask.reshape(-1) if possible_mask is not None else None))
    ct = ct_dense_1d.reshape(vol_shape)

    # dose
    dose = None
    dose_path = case_dir / "dose.csv"
    if dose_path.exists():
        dose_raw = _read_numeric_series(dose_path, dtype=np.float32)
        fill_mask_1d = (possible_mask.reshape(-1) if possible_mask is not None else None)
        dose_dense_1d = _to_dense_values(dose_raw, nvox, fill_mask=fill_mask_1d)
        dose = dose_dense_1d.reshape(vol_shape)

    # beamlet_indices
    beamlet_indices = None
    bi_path = case_dir / "beamlet_indices.csv"
    if bi_path.exists():
        beamlet_indices = pd.read_csv(bi_path, low_memory=False)

    # structures: treat as dense if length==nvox else sparse indices
    known = {"ct.csv", "dose.csv", "possible_dose_mask.csv", "voxel_dimensions.csv", "beamlet_indices.csv"}
    structures: Dict[str, np.ndarray] = {}
    for csv_path in sorted(case_dir.glob("*.csv")):
        if csv_path.name in known:
            continue
        name = csv_path.stem
        raw = _read_numeric_series(csv_path, dtype=np.float32)
        dense_1d = _to_dense_mask(raw, nvox)
        structures[name] = dense_1d.reshape(vol_shape)

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
