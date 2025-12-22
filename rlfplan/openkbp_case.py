# rlfplan/openkbp_case.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import scipy.sparse as sp

# ---- CSV readers (OpenKBP sparse-vector style) ----
def read_index_only_csv(path: str | Path) -> np.ndarray:
    """CSV with header ',data' and rows like '515752,' or '515752'."""
    path = Path(path)
    idx = []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "data" in s and "," in s:
                continue
            tok = s.split(",")[0].strip()
            if tok == "":
                continue
            try:
                idx.append(int(float(tok)))
            except ValueError:
                continue
    if not idx:
        raise ValueError(f"No indices parsed from {path}")
    return np.asarray(idx, dtype=np.int64)

def read_index_value_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """CSV with header ',data' and rows like '515752,19.808'."""
    path = Path(path)
    idx, val = [], []
    with path.open("r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "data" in s and "," in s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2 or parts[0] == "" or parts[1] == "":
                continue
            try:
                idx.append(int(float(parts[0])))
                val.append(float(parts[1]))
            except ValueError:
                continue
    if not idx:
        raise ValueError(f"No index,value parsed from {path}")
    return np.asarray(idx, dtype=np.int64), np.asarray(val, dtype=np.float32)

# ---- Core case container ----
DEFAULT_STRUCTS = [
    "PTV56", "PTV63", "PTV70",
    "Brainstem", "SpinalCord", "LeftParotid", "RightParotid",
]

@dataclass
class OpenKBPCase:
    case_id: str
    case_dir: Path

    # possible-dose subspace
    possible_idx_global: np.ndarray          # (nV,)
    dose_ref: np.ndarray                     # (nV,) Gy, aligned to possible_idx_global
    A: sp.csr_matrix                         # (nV, nB) float32 sparse influence matrix in possible-dose subspace

    # structures in possible-dose subspace: bool masks over nV
    struct_masks: Dict[str, np.ndarray]      # name -> (nV,) bool
    struct_global_counts: Dict[str, int]     # original size in global indices
    struct_in_possible_counts: Dict[str, int]

    @property
    def n_vox(self) -> int:
        return int(self.dose_ref.shape[0])

    @property
    def n_beamlets(self) -> int:
        return int(self.A.shape[1])

    @property
    def full_voxels(self) -> int:
        return int(self.A_full_rows)

    # store for reporting
    A_full_rows: int = 0

    @staticmethod
    def load(reference_plans_root: str | Path, case_id: str, structs: Optional[list[str]] = None) -> "OpenKBPCase":
        root = Path(reference_plans_root)
        case_dir = root / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)

        # Load possible indices + reference dose (must match order)
        possible_idx = read_index_only_csv(case_dir / "possible_dose_mask.csv")
        dose_idx, dose_val = read_index_value_csv(case_dir / "dose.csv")

        if possible_idx.shape[0] != dose_idx.shape[0] or not np.all(possible_idx == dose_idx):
            raise ValueError("possible_dose_mask indices and dose.csv indices are not identical (same order required).")

        # Load dij and slice to possible-dose subspace
        dij = sp.load_npz(case_dir / "dij.npz").tocsr()
        A_full_rows = dij.shape[0]
        # Convert to float32 to reduce memory pressure
        if dij.data.dtype != np.float32:
            dij.data = dij.data.astype(np.float32, copy=False)

        A = dij[possible_idx, :]  # CSR slice
        A.eliminate_zeros()

        # Build global->local mapping (vectorized, fast, moderate memory)
        global2local = np.full((A_full_rows,), -1, dtype=np.int32)
        global2local[possible_idx] = np.arange(possible_idx.shape[0], dtype=np.int32)

        # Load structures
        structs = structs or DEFAULT_STRUCTS
        struct_masks: Dict[str, np.ndarray] = {}
        global_counts: Dict[str, int] = {}
        in_possible_counts: Dict[str, int] = {}

        nV = possible_idx.shape[0]
        for name in structs:
            p = case_dir / f"{name}.csv"
            if not p.exists():
                continue
            sidx_global = read_index_only_csv(p)
            global_counts[name] = int(sidx_global.shape[0])
            sidx_local = global2local[sidx_global]
            sidx_local = sidx_local[sidx_local >= 0]
            in_possible_counts[name] = int(sidx_local.shape[0])

            m = np.zeros((nV,), dtype=bool)
            m[sidx_local] = True
            struct_masks[name] = m

        return OpenKBPCase(
            case_id=case_id,
            case_dir=case_dir,
            possible_idx_global=possible_idx,
            dose_ref=dose_val,
            A=A,
            struct_masks=struct_masks,
            struct_global_counts=global_counts,
            struct_in_possible_counts=in_possible_counts,
            A_full_rows=A_full_rows,
        )
