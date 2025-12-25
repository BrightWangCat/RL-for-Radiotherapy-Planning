# rlfplan/env_openkbp_vmat2d.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.sparse as sp

try:
    from scipy.ndimage import rotate as nd_rotate
    from scipy.ndimage import binary_dilation as nd_binary_dilation
except Exception:  # pragma: no cover
    nd_rotate = None
    nd_binary_dilation = None

from rlfplan.openkbp_case import OpenKBPCase


# --- Paper-aligned discrete action table (Table I) ---
# index: 0..14 corresponds to action 1..15 in paper
# (dx1_mm, dx2_mm, dd0) where dd0 is % of baseline dose rate (i.e., +/-20 or 0)
ACTIONS_15 = [
    (0,   0,   0),     # 1
    (+5, +5,  0),      # 2
    (+5, -5,  0),      # 3
    (-5, +5,  0),      # 4
    (-5, -5,  0),      # 5
    (0,   0,  +20),    # 6
    (+5, +5, +20),     # 7
    (+5, -5, +20),     # 8
    (-5, +5, +20),     # 9
    (-5, -5, +20),     # 10
    (0,   0,  -20),    # 11
    (+5, +5, -20),     # 12
    (+5, -5, -20),     # 13
    (-5, +5, -20),     # 14
    (-5, -5, -20),     # 15
]


@dataclass
class GroupedDoseModel:
    # FULL (possibly multi-slice) arrays in "possible-dose space"
    B_full: np.ndarray                 # (nV, K) float32 dense
    ptv_mask_full: np.ndarray          # (nV,) bool
    brainstem_mask_full: Optional[np.ndarray]
    spinalcord_mask_full: Optional[np.ndarray]

    # reference stats (full, may be overwritten by slice stats)
    ptv_ref_mean_full: float
    brainstem_ref_mean_full: float
    spinalcord_ref_mean_full: float

    gain: float                        # calibration gain applied to B_full

    # IMPORTANT: to align with paper leaf indexing semantics, we DO NOT reorder bins.
    # Leaf indices are interpreted as lateral bins 0..K-1.
    order: np.ndarray                  # identity 0..K-1


class OpenKBPVMAT2DEnv(gym.Env):
    """
    Paper-aligned IO for VMAT MPO (2D) with PPO as optimizer.

    Observation: uint8 (96,96,2)
      - frame1: (normalized dose - objective map), mask non-objective voxels=0,
                rotate by -theta (gantry frame), rescale to uint8 via (+1)*50,
                and set PTV boundary ring to 255.
      - frame2: machine parameter sinogram: y=relative theta, x=[d0|x1|x2],
                mark with 200/150/100; d0 spans left 30 pixels.

    Action: Discrete(15) Table I (leaf +/-5mm/0 and dose rate +/-20/0)
    Gantry increment: 3.75° => 96 control points
    Max steps: up to 2 gantry rotations (192) per paper.

    Notes:
      - If case contains multiple slices stacked in voxel order (96*96 per slice),
        we can sample a slice at reset (paper: random slice each iteration).
      - Reward is negative paper-like cost g(si). Termination uses paper early stop.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        case: Optional[OpenKBPCase] = None,
        *,
        root: Optional[str] = None,
        case_id: Optional[str] = None,

        K: int = 96,                  # paper naturally aligns with 96 lateral bins
        n_cps: int = 96,
        max_steps: int = 192,

        # paper constraints / parameters
        d0_min: int = 20,
        d0_max: int = 600,
        leaf_step_mm: int = 5,
        ptv_margin_mm: int = 10,      # 1.0 cm margin for conformal init/limits

        init_d0: int = 100,
        init_mode: str = "conformal",  # "conformal" or "constant"
        sample_slices: bool = True,    # paper: random slice each iteration

        seed: int = 0,
    ):
        super().__init__()

        self.root = root or os.environ.get("OPENKBP_ROOT", "")
        self.case_id = case_id or os.environ.get("OPENKBP_CASE", "")

        if case is None:
            if not self.root or not self.case_id:
                raise ValueError("Provide (case) or set OPENKBP_ROOT and OPENKBP_CASE.")
            case = OpenKBPCase.load(self.root, self.case_id)

        self.case: OpenKBPCase = case
        if not self.case_id:
            inferred = getattr(case, "case_id", None) or getattr(case, "patient_id", None)
            if inferred is not None:
                self.case_id = str(inferred)

        self.K = int(K)
        self.n_cps = int(n_cps)
        self.max_steps = int(max_steps)

        self.d0_min = int(d0_min)
        self.d0_max = int(d0_max)
        self.leaf_step_mm = int(leaf_step_mm)
        self.ptv_margin_mm = int(ptv_margin_mm)
        self.init_d0 = int(init_d0)
        self.init_mode = str(init_mode).lower()
        if self.init_mode not in ("conformal", "constant"):
            raise ValueError("init_mode must be 'conformal' or 'constant'")
        self.sample_slices = bool(sample_slices)

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        # Build grouped model (identity leaf indexing)
        self.model = self._build_grouped_model(self.case, self.K)

        # Full objective vector (per-voxel objective value). We will slice it.
        self._obj_full = self._build_objective_vec_full()

        # Slice system (active 2D view: exactly 96*96 voxels)
        self._slice_len = 96 * 96
        self._slice_candidates = self._discover_slice_candidates()
        self._active_slice_idx = 0
        self._active_offset = 0

        # active slice views (set in reset)
        self.B = None                   # (96*96, K)
        self.ptv_mask = None            # (96*96,)
        self.bs_mask = None             # (96*96,) or None
        self.sc_mask = None             # (96*96,) or None
        self.obj_vec = None             # (96*96,)

        # PTV boundary ring in patient frame for active slice: (96,96) bool
        self._ptv_ring_img0 = None

        # per-CP allowed aperture open-limits (paper: within 1cm of target edge)
        self._x1_min_allowed = np.zeros((self.n_cps,), dtype=np.int32)
        self._x2_max_allowed = np.zeros((self.n_cps,), dtype=np.int32)

        # plan state across CPs
        self.x1_idx = np.zeros((self.n_cps,), dtype=np.int32)
        self.x2_idx = np.zeros((self.n_cps,), dtype=np.int32)
        self.d0 = np.zeros((self.n_cps,), dtype=np.int32)

        # cost normalization term gOAR(s0)
        self._goar0 = 1.0

        # counters
        self.t = 0
        self.cp_idx = 0

        # spaces
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 2), dtype=np.uint8)

        # --- fast plan caches (on K bins) ---
        self._mask_cp = np.zeros((self.n_cps, self.K), dtype=np.float32)
        self._scale_cp = np.zeros((self.n_cps,), dtype=np.float32)
        self._sum_masked = np.zeros((self.K,), dtype=np.float32)

    # -------------------- model + objectives --------------------
    @staticmethod
    def _build_grouped_model(case: OpenKBPCase, K: int) -> GroupedDoseModel:
        """
        Build dense grouped influence matrix B_full from sparse A.
        Leaf indexing is identity 0..K-1 to align with paper semantics.
        """
        A: sp.csr_matrix = case.A  # (nV, nB)
        nV, nB = A.shape
        K = min(int(K), int(nB))
        if K <= 0:
            raise ValueError("K must be >= 1")

        edges = np.linspace(0, nB, K + 1, dtype=np.int32)
        B = np.zeros((nV, K), dtype=np.float32)

        Acsc = A.tocsc()
        for g in range(K):
            c0, c1 = int(edges[g]), int(edges[g + 1])
            if c1 <= c0:
                continue
            Bg = Acsc[:, c0:c1].sum(axis=1)
            B[:, g] = np.asarray(Bg).reshape(-1).astype(np.float32)

        ptv = case.struct_masks.get("PTV70", None)
        if ptv is None or int(ptv.sum()) == 0:
            raise ValueError("PTV70 mask missing/empty in possible-dose space.")

        brainstem = case.struct_masks.get("Brainstem", None)
        if brainstem is not None and int(brainstem.sum()) == 0:
            brainstem = None

        spinalcord = case.struct_masks.get("SpinalCord", None)
        if spinalcord is not None and int(spinalcord.sum()) == 0:
            spinalcord = None

        dose_ref = case.dose_ref
        ptv_ref_mean = float(dose_ref[ptv].mean())
        bs_ref_mean = float(dose_ref[brainstem].mean()) if brainstem is not None else 0.0
        sc_ref_mean = float(dose_ref[spinalcord].mean()) if spinalcord is not None else 0.0

        # Calibration gain so that "fully open" roughly matches reference magnitude.
        ones = np.ones((K,), dtype=np.float32)
        ptv_mean_per_unit = float((B @ ones)[ptv].mean())
        if ptv_mean_per_unit <= 1e-8:
            raise ValueError("Calibration failed: ptv_mean_per_unit is ~0.")
        gain = float(ptv_ref_mean / ptv_mean_per_unit)
        B *= np.float32(gain)

        order = np.arange(K, dtype=np.int32)  # identity order (paper-aligned leaf bins)

        return GroupedDoseModel(
            B_full=B,
            ptv_mask_full=ptv.astype(bool),
            brainstem_mask_full=brainstem.astype(bool) if brainstem is not None else None,
            spinalcord_mask_full=spinalcord.astype(bool) if spinalcord is not None else None,
            ptv_ref_mean_full=ptv_ref_mean,
            brainstem_ref_mean_full=bs_ref_mean,
            spinalcord_ref_mean_full=sc_ref_mean,
            gain=gain,
            order=order,
        )

    def _build_objective_vec_full(self) -> np.ndarray:
        """
        Per-voxel objective value in the same dose units as dose_ref.
        This is a simplification relative to the paper's Table II DVH objectives,
        but keeps the "objective map" concept aligned structurally.
        """
        nV = int(self.model.B_full.shape[0])
        obj = np.zeros((nV,), dtype=np.float32)

        obj[self.model.ptv_mask_full] = np.float32(self.model.ptv_ref_mean_full)
        if self.model.brainstem_mask_full is not None:
            obj[self.model.brainstem_mask_full] = np.float32(self.model.brainstem_ref_mean_full)
        if self.model.spinalcord_mask_full is not None:
            obj[self.model.spinalcord_mask_full] = np.float32(self.model.spinalcord_ref_mean_full)
        return obj

    # -------------------- slice handling (paper: random slice each iteration) --------------------
    def _discover_slice_candidates(self) -> np.ndarray:
        """
        If nV is multiple of 96*96, treat as stacked slices and select those with PTV present.
        Otherwise, treat as single slice (offset=0).
        """
        nV = int(self.model.B_full.shape[0])
        H = self._slice_len
        if nV < H:
            raise ValueError(f"nV={nV} < 96*96={H}. Cannot form paper-aligned 96x96 state.")
        if nV % H != 0:
            # Not perfectly stacked; fallback to single slice at offset 0 (still deterministic).
            return np.array([0], dtype=np.int32)

        nS = nV // H
        ptv = self.model.ptv_mask_full
        cand = []
        for s in range(nS):
            off = s * H
            if np.any(ptv[off:off + H]):
                cand.append(s)
        if len(cand) == 0:
            # if no ptv slice found, keep slice 0
            cand = [0]
        return np.array(cand, dtype=np.int32)

    def _set_active_slice(self, slice_idx: int):
        """
        Activate one 96x96 slice view by slicing B_full and masks/objectives.
        """
        nV = int(self.model.B_full.shape[0])
        H = self._slice_len

        if (nV % H) == 0:
            off = int(slice_idx) * H
        else:
            off = 0

        self._active_slice_idx = int(slice_idx)
        self._active_offset = int(off)

        self.B = self.model.B_full[off:off + H, :].astype(np.float32, copy=False)
        self.ptv_mask = self.model.ptv_mask_full[off:off + H].astype(bool, copy=False)
        self.bs_mask = self.model.brainstem_mask_full[off:off + H].astype(bool, copy=False) if self.model.brainstem_mask_full is not None else None
        self.sc_mask = self.model.spinalcord_mask_full[off:off + H].astype(bool, copy=False) if self.model.spinalcord_mask_full is not None else None

        self.obj_vec = self._obj_full[off:off + H].astype(np.float32, copy=False)

        # Build PTV boundary ring (patient frame)
        ptv_img = self.ptv_mask.reshape(96, 96)
        self._ptv_ring_img0 = self._ptv_boundary_ring(ptv_img)

    # -------------------- morphology / rotation helpers --------------------
    @staticmethod
    def _dilate3x3(mask: np.ndarray) -> np.ndarray:
        """
        Fallback binary dilation (3x3) if scipy.ndimage.binary_dilation is unavailable.
        """
        m = mask.astype(np.uint8)
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        out = np.zeros_like(m)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out = np.maximum(out, p[1 + dy:1 + dy + m.shape[0], 1 + dx:1 + dx + m.shape[1]])
        return out.astype(bool)

    def _ptv_boundary_ring(self, ptv_img: np.ndarray) -> np.ndarray:
        """
        Paper: voxels surrounding PTV set to 255 to indicate boundary.
        We implement a 1-pixel ring outside PTV via dilation(ptv) & ~ptv.
        """
        if nd_binary_dilation is not None:
            dil = nd_binary_dilation(ptv_img.astype(bool), iterations=1)
        else:
            dil = self._dilate3x3(ptv_img.astype(bool))
        ring = np.logical_and(dil, ~ptv_img.astype(bool))
        return ring.astype(bool)

    def _rotate_2d(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        if nd_rotate is None:
            return img
        # nearest-neighbor rotation to keep mask crisp
        return nd_rotate(img, angle=angle_deg, reshape=False, order=0, mode="constant", cval=0)

    # -------------------- plan mask + caches --------------------
    def _mask_from_x1x2(self, x1: int, x2: int) -> np.ndarray:
        """
        Aperture mask on K lateral bins. Identity indexing (paper-aligned).
        """
        x1 = int(np.clip(x1, 0, self.K - 2))
        x2 = int(np.clip(x2, x1 + 1, self.K - 1))
        m = np.zeros((self.K,), dtype=np.float32)
        m[x1:x2 + 1] = 1.0
        return m

    def _rebuild_plan_cache(self):
        self._scale_cp[:] = (self.d0.astype(np.float32) / 100.0)
        self._sum_masked[:] = 0.0
        for cp in range(self.n_cps):
            m = self._mask_from_x1x2(int(self.x1_idx[cp]), int(self.x2_idx[cp]))
            self._mask_cp[cp] = m
            self._sum_masked += m * float(self._scale_cp[cp])

    def _effective_weights(self) -> np.ndarray:
        # mean over CPs
        return (self._sum_masked / float(self.n_cps)).astype(np.float32)

    def _dose(self) -> np.ndarray:
        w = self._effective_weights()
        # active 2D slice dose
        return self.B @ w

    # -------------------- paper-aligned state frames --------------------
    def _frame1(self, dose: np.ndarray, theta_deg: float) -> np.ndarray:
        """
        Paper frame1:
          - normalize dose so max PTV dose = 1.08
          - subtract objective map (normalized by same scale)
          - mask voxels without objective to 0
          - rotate by -theta (gantry frame)
          - rescale to uint8: (x + 1) * 50
          - set PTV boundary ring to 255
        """
        dose = np.asarray(dose, dtype=np.float32).reshape(-1)
        obj = np.asarray(self.obj_vec, dtype=np.float32).reshape(-1)

        dose_img = dose.reshape(96, 96)
        obj_img = obj.reshape(96, 96)

        # normalization: max PTV dose == 1.08
        ptv_vals = dose[self.ptv_mask]
        ptv_max = float(np.max(ptv_vals)) if ptv_vals.size else float(np.max(dose))
        ptv_max = max(ptv_max, 1e-6)
        scale = ptv_max / 1.08

        dose_n = dose_img / scale
        obj_n = obj_img / scale
        diff = dose_n - obj_n

        # uint8 rescale (paper)
        frame = (diff + 1.0) * 50.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        # mask voxels with no objective
        frame[obj_img == 0.0] = 0

        # rotate into gantry frame
        frame_rot = self._rotate_2d(frame, angle_deg=-theta_deg).astype(np.uint8)

        # rotate boundary ring and set to 255
        if self._ptv_ring_img0 is not None:
            ring_u8 = (self._ptv_ring_img0.astype(np.uint8) * 255)
            ring_rot = self._rotate_2d(ring_u8, angle_deg=-theta_deg)
            frame_rot[ring_rot > 0] = 255

        return frame_rot

    def _frame2_sinogram(self) -> np.ndarray:
        """
        Paper frame2:
          - y-axis: relative theta
          - x-axis: [d0 | x1 | x2]
          - mark d0=200, x1=150, x2=100
          - d0 spans left 30 pixels over [d0_min, d0_max]
          - background 0
        """
        H, W = 96, 96
        img = np.zeros((H, W), dtype=np.uint8)

        denom_d0 = max(1, (self.d0_max - self.d0_min))
        denom_x = max(1, (self.K - 1))

        # fixed segmentation: d0=30 cols, x1=33 cols, x2=33 cols
        for cp in range(self.n_cps):
            r = int(cp % H)

            d0 = int(self.d0[cp])
            d0_col = int(np.clip(round((d0 - self.d0_min) / denom_d0 * 29), 0, 29))
            img[r, d0_col] = 200

            x1 = int(self.x1_idx[cp])
            x1_col = 30 + int(np.clip(round(x1 / denom_x * 32), 0, 32))
            img[r, x1_col] = 150

            x2 = int(self.x2_idx[cp])
            x2_col = 63 + int(np.clip(round(x2 / denom_x * 32), 0, 32))
            img[r, x2_col] = 100

        # relative-theta alignment: keep current cp near middle row (48)
        shift = 48 - int(self.cp_idx % H)
        img = np.roll(img, shift=shift, axis=0)
        return img

    def _obs(self, dose: np.ndarray) -> np.ndarray:
        theta_deg = float(self.cp_idx) * (360.0 / float(self.n_cps))
        f1 = self._frame1(dose, theta_deg=theta_deg)
        f2 = self._frame2_sinogram()
        return np.stack([f1, f2], axis=-1).astype(np.uint8)

    # -------------------- paper-aligned conformal init + constraints --------------------
    def _idx_from_col(self, col: int) -> int:
        # map image column 0..95 to leaf index 0..K-1
        return int(np.clip(round(float(col) / 95.0 * float(self.K - 1)), 0, self.K - 1))

    def _compute_conformal_aperture_limits(self):
        """
        For each CP, rotate PTV mask into gantry frame and compute x1/x2 edges
        with 1cm margin. Also set open-limit constraints (paper: within 1cm of target edge).
        """
        margin_steps = int(round(float(self.ptv_margin_mm) / float(self.leaf_step_mm)))
        ptv_img0 = self.ptv_mask.reshape(96, 96).astype(bool)

        for cp in range(self.n_cps):
            theta = float(cp) * (360.0 / float(self.n_cps))
            ptv_rot = self._rotate_2d(ptv_img0.astype(np.uint8), angle_deg=-theta) > 0

            cols = np.where(ptv_rot.any(axis=0))[0]
            if cols.size == 0:
                # fallback to center small window
                c0, c1 = 45, 50
            else:
                c0, c1 = int(cols.min()), int(cols.max())

            x1_edge = self._idx_from_col(c0)
            x2_edge = self._idx_from_col(c1)

            x1_min = int(np.clip(x1_edge - margin_steps, 0, self.K - 2))
            x2_max = int(np.clip(x2_edge + margin_steps, 1, self.K - 1))
            if x2_max <= x1_min:
                x2_max = int(np.clip(x1_min + 1, 1, self.K - 1))

            self._x1_min_allowed[cp] = x1_min
            self._x2_max_allowed[cp] = x2_max

    def _init_plan_conformal_arc(self):
        """
        Initialize plan as conformal arc (2D approximation):
          - for each CP, set x1/x2 to PTV edge + 1cm margin (in gantry frame)
          - constant dose rate init_d0
        """
        self._compute_conformal_aperture_limits()
        for cp in range(self.n_cps):
            self.x1_idx[cp] = int(self._x1_min_allowed[cp])
            self.x2_idx[cp] = int(self._x2_max_allowed[cp])
        self.d0[:] = int(np.clip(self.init_d0, self.d0_min, self.d0_max))

    def _init_plan_constant(self):
        """
        Constant parameter init (paper Appendix-style): fixed aperture and d0.
        """
        # allow full range (no "within margin" constraint)
        self._x1_min_allowed[:] = 0
        self._x2_max_allowed[:] = self.K - 1

        center = self.K // 2
        half_width = max(2, int(round(16 / float(self.leaf_step_mm))))  # ~1.6cm each side as default
        x1 = int(np.clip(center - half_width, 0, self.K - 2))
        x2 = int(np.clip(center + half_width, x1 + 1, self.K - 1))

        self.x1_idx[:] = x1
        self.x2_idx[:] = x2
        self.d0[:] = int(np.clip(self.init_d0, self.d0_min, self.d0_max))

    def _clip_aperture_to_limits(self, cp: int):
        """
        Paper constraint: leaf positions remain within 1cm of target edge (open limits).
        We enforce:
          x1 >= x1_min_allowed[cp]
          x2 <= x2_max_allowed[cp]
          and x2 > x1
        """
        cp = int(cp)
        x1_min = int(self._x1_min_allowed[cp])
        x2_max = int(self._x2_max_allowed[cp])

        x1 = int(self.x1_idx[cp])
        x2 = int(self.x2_idx[cp])

        x1 = int(np.clip(x1, x1_min, self.K - 2))
        x2 = int(np.clip(x2, 1, x2_max))

        if x2 <= x1:
            x2 = int(np.clip(x1 + 1, 1, x2_max))

        self.x1_idx[cp] = x1
        self.x2_idx[cp] = x2

    # -------------------- paper-like cost + stopping --------------------
    def _compute_goar_terms(self, dose_n: np.ndarray, obj_n: np.ndarray, ri: float) -> Tuple[float, Dict[str, float]]:
        """
        Paper gOAR is sum over OAR metrics max(Dk_i/ri - Dk_obj, 0).
        Here we use mean dose as metric for Brainstem/SpinalCord (dataset dependent),
        but keep the paper formula form.
        """
        terms: Dict[str, float] = {}
        goar = 0.0

        if self.bs_mask is not None and np.any(self.bs_mask):
            dk_i = float(dose_n[self.bs_mask].mean())
            dk_obj = float(obj_n[self.bs_mask].mean())  # constant within mask
            t = max(dk_i / max(ri, 1e-6) - dk_obj, 0.0)
            terms["brainstem"] = t
            goar += t

        if self.sc_mask is not None and np.any(self.sc_mask):
            dk_i = float(dose_n[self.sc_mask].mean())
            dk_obj = float(obj_n[self.sc_mask].mean())
            t = max(dk_i / max(ri, 1e-6) - dk_obj, 0.0)
            terms["spinalcord"] = t
            goar += t

        return float(goar), terms

    def _compute_cost(self, dose: np.ndarray) -> Tuple[float, float, float, float, bool, Dict[str, Any]]:
        """
        Returns:
          g_total, gPTV, gOAR, gOAR_ratio, eps_stop, extra_dict
        """
        dose = np.asarray(dose, dtype=np.float32).reshape(-1)
        obj = np.asarray(self.obj_vec, dtype=np.float32).reshape(-1)

        # normalize so max PTV dose = 1.08 (paper)
        ptv_vals = dose[self.ptv_mask]
        ptv_max = float(np.max(ptv_vals)) if ptv_vals.size else float(np.max(dose))
        ptv_max = max(ptv_max, 1e-6)
        scale = ptv_max / 1.08

        dose_n = dose / scale
        obj_n = obj / scale

        # prescription level ri = 0.926 * DPTVmax (paper) -> ~1.0 when DPTVmax=1.08
        dptv_max_n = float(np.max(dose_n[self.ptv_mask])) if np.any(self.ptv_mask) else float(np.max(dose_n))
        ri = 0.926 * dptv_max_n

        # gPTV = 1 - V_PTV_r
        v_ptv_r = float(np.mean(dose_n[self.ptv_mask] >= ri)) if np.any(self.ptv_mask) else 0.0
        gptv = 1.0 - v_ptv_r

        # gOAR (paper form, metric simplified)
        goar, goar_terms = self._compute_goar_terms(dose_n, obj_n, ri)

        # normalize gOAR by gOAR(s0), denom >= 0.05, ratio <= 1 (paper)
        denom = max(float(self._goar0), 0.05)
        goar_ratio = min(goar / denom, 1.0)

        # total cost g(si) = 0.5*gPTV + goar/goar0 (paper Eq.1 form)
        g_total = 0.5 * gptv + goar_ratio

        # early stopping criterion: gPTV<0.05 & gOAR<0.05 (paper)
        eps_stop = (gptv < 0.05) and (goar < 0.05)

        extra = {
            "scale_ptvmax_to_1p08": float(scale),
            "dptv_max_norm": float(dptv_max_n),
            "ri_norm": float(ri),
            "v_ptv_ri": float(v_ptv_r),
            "goar_terms": goar_terms,
        }
        return float(g_total), float(gptv), float(goar), float(goar_ratio), bool(eps_stop), extra

    # -------------------- Gym API --------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)

        options = options or {}

        # Optional: switch case on reset (multi-case training)
        new_case_id = options.get("case_id", None)
        if new_case_id is not None and str(new_case_id) != str(self.case_id):
            if not self.root:
                raise ValueError("To switch case_id via options, set OPENKBP_ROOT or pass root=.")
            self.case_id = str(new_case_id)
            self.case = OpenKBPCase.load(self.root, self.case_id)

            # rebuild model/objectives/slice candidates
            self.model = self._build_grouped_model(self.case, self.K)
            self._obj_full = self._build_objective_vec_full()
            self._slice_candidates = self._discover_slice_candidates()

        # choose active slice (paper: random training slice)
        if self.sample_slices and self._slice_candidates.size > 0:
            sidx = int(self.rng.choice(self._slice_candidates))
        else:
            sidx = int(self._slice_candidates[0]) if self._slice_candidates.size > 0 else 0
        self._set_active_slice(sidx)

        # init plan
        if self.init_mode == "conformal":
            self._init_plan_conformal_arc()
        else:
            self._init_plan_constant()

        self.t = 0
        self.cp_idx = 0

        # build caches for baseline plan
        self._rebuild_plan_cache()

        # baseline dose + baseline gOAR(s0) for normalization
        dose0 = self._dose()
        g0, gptv0, goar0, goar_ratio0, eps0, extra0 = self._compute_cost(dose0)
        self._goar0 = float(goar0)

        obs = self._obs(dose0)

        info = {
            "case_id": self.case_id,
            "slice_idx": int(self._active_slice_idx),
            "slice_offset": int(self._active_offset),
            "calibration_gain": float(self.model.gain),

            "cp_idx": int(self.cp_idx),
            "theta_deg": 0.0,

            # applied (at cp=0)
            "d0": int(self.d0[self.cp_idx]),
            "x1_mm": int(self.x1_idx[self.cp_idx]) * self.leaf_step_mm,
            "x2_mm": int(self.x2_idx[self.cp_idx]) * self.leaf_step_mm,

            # paper cost components at s0
            "g_total": float(g0),
            "g_ptv": float(gptv0),
            "g_oar": float(goar0),
            "g_oar_ratio": float(goar_ratio0),
            "eps_stop": bool(eps0),
            **extra0,
        }
        return obs, info

    def step(self, action: int):
        self.t += 1

        a = int(action)
        if a < 0 or a >= 15:
            raise ValueError(f"action must be in [0,14], got {a}")

        dx1_mm, dx2_mm, dd0 = ACTIONS_15[a]

        # apply to CURRENT control point (paper: replace current CP)
        cp_applied = int(self.cp_idx)
        theta_applied_deg = float(cp_applied) * (360.0 / float(self.n_cps))

        # old contribution for incremental update
        old_scale = float(self._scale_cp[cp_applied])
        old_mask = self._mask_cp[cp_applied]
        old_term = old_mask * np.float32(old_scale)

        # convert mm to index steps
        dx1 = int(round(dx1_mm / float(self.leaf_step_mm)))
        dx2 = int(round(dx2_mm / float(self.leaf_step_mm)))

        # update parameters
        self.x1_idx[cp_applied] = int(self.x1_idx[cp_applied]) + dx1
        self.x2_idx[cp_applied] = int(self.x2_idx[cp_applied]) + dx2
        self.d0[cp_applied] = int(self.d0[cp_applied]) + int(dd0)

        # constraints (paper): d0 range; x1/x2 do not cross; within margin of PTV edge
        self.d0[cp_applied] = int(np.clip(self.d0[cp_applied], self.d0_min, self.d0_max))
        self.x1_idx[cp_applied] = int(np.clip(self.x1_idx[cp_applied], 0, self.K - 2))
        self.x2_idx[cp_applied] = int(np.clip(self.x2_idx[cp_applied], 1, self.K - 1))
        if self.x2_idx[cp_applied] <= self.x1_idx[cp_applied]:
            self.x2_idx[cp_applied] = int(np.clip(self.x1_idx[cp_applied] + 1, 1, self.K - 1))

        # paper "within 1 cm of target edge" open-limits
        self._clip_aperture_to_limits(cp_applied)

        # new mask & scale
        new_scale = float(self.d0[cp_applied]) / 100.0
        new_mask = self._mask_from_x1x2(int(self.x1_idx[cp_applied]), int(self.x2_idx[cp_applied]))
        new_term = new_mask * np.float32(new_scale)

        # write caches + incremental update
        self._scale_cp[cp_applied] = np.float32(new_scale)
        self._mask_cp[cp_applied] = new_mask
        self._sum_masked += (new_term - old_term)

        # advance gantry / next CP
        self.cp_idx = (self.cp_idx + 1) % self.n_cps
        theta_deg = float(self.cp_idx) * (360.0 / float(self.n_cps))

        # dose + paper-like cost
        dose = self._dose()
        g_total, gptv, goar, goar_ratio, eps_stop, extra = self._compute_cost(dose)

        # PPO reward: maximize -> negative cost
        reward = -g_total

        terminated = bool(eps_stop)
        truncated = bool(self.t >= self.max_steps)

        obs = self._obs(dose)

        info: Dict[str, Any] = {
            "case_id": self.case_id,
            "slice_idx": int(self._active_slice_idx),
            "slice_offset": int(self._active_offset),

            "cp_idx": int(self.cp_idx),
            "theta_deg": float(theta_deg),

            "cp_applied": int(cp_applied),
            "theta_applied_deg": float(theta_applied_deg),
            "action_index": int(a),
            "dx1_mm": int(dx1_mm),
            "dx2_mm": int(dx2_mm),
            "dd0": int(dd0),

            # applied values (paper semantics)
            "d0": int(self.d0[cp_applied]),
            "x1_mm": int(self.x1_idx[cp_applied]) * self.leaf_step_mm,
            "x2_mm": int(self.x2_idx[cp_applied]) * self.leaf_step_mm,

            # paper cost components
            "g_total": float(g_total),
            "g_ptv": float(gptv),
            "g_oar": float(goar),
            "g_oar_ratio": float(goar_ratio),
            "eps_stop": bool(eps_stop),

            # compatibility fields (older scripts)
            "err_norm": float(g_total),
            "oar_pen_norm": float(goar_ratio),

            "calibration_gain": float(self.model.gain),
            **extra,
        }

        return obs, float(reward), terminated, truncated, info
