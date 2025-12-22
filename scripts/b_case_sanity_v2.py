#!/usr/bin/env python3
import os, sys
import numpy as np

def read_index_only_csv(path: str) -> np.ndarray:
    """Read csv of form: header ',data' then rows like '515752,' or '515752'."""
    idx = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # skip header lines
            if "data" in s and "," in s:
                continue
            # take token before comma (or whole line)
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

def read_index_value_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read csv rows like '515752,19.808' (with header ',data')."""
    idx = []
    val = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "data" in s and "," in s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 2:
                continue
            if parts[0] == "" or parts[1] == "":
                continue
            try:
                idx.append(int(float(parts[0])))
                val.append(float(parts[1]))
            except ValueError:
                continue
    if not idx:
        raise ValueError(f"No index,value parsed from {path}")
    return np.asarray(idx, dtype=np.int64), np.asarray(val, dtype=np.float32)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/b_case_sanity_v2.py <reference_plans_root> [case_id]")
        sys.exit(2)
    root = sys.argv[1]
    case_id = sys.argv[2] if len(sys.argv) >= 3 else None

    if case_id is None:
        case_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not case_dirs:
            raise RuntimeError(f"No case directories under {root}")
        case_id = case_dirs[0]

    case_dir = os.path.join(root, case_id)
    print(f"Selected case: {case_dir}")

    # load dij
    import scipy.sparse as sp
    dij = sp.load_npz(os.path.join(case_dir, "dij.npz")).tocsr()
    print(f"dij: shape={dij.shape}, nnz={dij.nnz}, dtype={dij.dtype}")

    # possible indices + dose
    poss_idx = read_index_only_csv(os.path.join(case_dir, "possible_dose_mask.csv"))
    dose_idx, dose_val = read_index_value_csv(os.path.join(case_dir, "dose.csv"))

    print(f"possible indices: n={poss_idx.size}, min={poss_idx.min()}, max={poss_idx.max()}")
    print(f"dose: n={dose_val.size}, min={dose_val.min():.4g}, max={dose_val.max():.4g}, mean={dose_val.mean():.4g}")

    # alignment checks
    same_order = (poss_idx.size == dose_idx.size) and np.all(poss_idx == dose_idx)
    same_set = (poss_idx.size == dose_idx.size) and (np.array_equal(np.sort(poss_idx), np.sort(dose_idx)))
    print(f"possible_idx == dose_idx (same order): {same_order}")
    print(f"possible_idx and dose_idx (same set):  {same_set}")

    # core consistency
    print("\n=== CONSISTENCY CHECKS ===")
    print(f"dij rows (full voxels): {dij.shape[0]}  (expect 128^3=2097152 for H&N OpenKBP)")
    print(f"possible voxels:        {poss_idx.size}")

    # slice dij to possible voxels
    A = dij[poss_idx, :]
    print(f"A = dij[possible_idx,:] shape={A.shape}, nnz={A.nnz}")

    # structure indices and intersection with possible voxels
    struct_files = [
        "PTV56.csv","PTV63.csv","PTV70.csv",
        "Brainstem.csv","SpinalCord.csv","LeftParotid.csv","RightParotid.csv",
    ]
    poss_set = set(poss_idx.tolist())
    print("\n=== STRUCTURES (index-list) ===")
    for fn in struct_files:
        p = os.path.join(case_dir, fn)
        sidx = read_index_only_csv(p)
        inter = sum((i in poss_set) for i in sidx.tolist())
        print(f"{fn:14s} n={sidx.size:6d}  in_possible={inter:6d}  (ratio={inter/sidx.size:.3f})")

if __name__ == "__main__":
    main()
