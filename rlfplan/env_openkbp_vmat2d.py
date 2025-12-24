# rlfplan/env_openkbp_vmat2d.py
from __future__ import annotations

import os
import zlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.sparse as sp

try:
    from scipy.ndimage import rotate as nd_rotate
except Exception:  # pragma: no cover
    nd_rotate = None

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
    B: np.ndarray                    # (nV, K) float32 dense
    ptv70_mask: np.ndarray           # (nV,) bool
    brainstem_mask: Optional[np.ndarray]    # (nV,) bool or None
    spinalcord_mask: Optional[np.ndarray]   # (nV,) bool or None
    ptv70_ref_mean: float
    brainstem_ref_mean: float
    spinalcord_ref_mean: float
    gain: float                      # calibration gain applied to B
    order: np.ndarray                # (K,) int, sorted indices by PTV/OAR ratio desc


class OpenKBPVMAT2DEnv(gym.Env):
    """
    Paper-aligned IO (approximate VMAT dynamics) + fast incremental dose cache.

    Observation: uint8 (96,96,2)
      - frame1: (normalized dose - objective map) rotated by -theta
      - frame2: machine parameter sinogram

    Action: Discrete(15) exactly Table I in paper
    Gantry increment: 3.75° => 96 control points

    Performance note:
      - We maintain per-control-point mask & scale caches and a running sum over CPs.
      - Each step updates only the modified CP contribution (O(K) instead of O(n_cps*K)).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        case: Optional[OpenKBPCase] = None,
        *,
        root: Optional[str] = None,
        case_id: Optional[str] = None,
        K: int = 64,
        n_cps: int = 96,
        max_steps: int = 192,
        oar_lambda: float = 0.02,
        seed: int = 0,
        d0_min: int = 20,
        d0_max: int = 600,
        leaf_step_mm: int = 5,
        init_leaf_half_width: int = 8,
        init_d0: int = 100,
        calibrate_init: bool = True,
        init_scale_clip: Tuple[float, float] = (0.2, 6.0),
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
        self.oar_lambda = float(oar_lambda)

        self.base_seed = int(seed)
        self.rng = np.random.default_rng(self.base_seed)

        self.d0_min = int(d0_min)
        self.d0_max = int(d0_max)
        self.leaf_step_mm = int(leaf_step_mm)

        self.init_leaf_half_width = int(init_leaf_half_width)
        self.init_d0 = int(init_d0)

        self.calibrate_init = bool(calibrate_init)
        self.init_scale_clip = (float(init_scale_clip[0]), float(init_scale_clip[1]))

        self.model = self._build_grouped_model(self.case, self.K)

        # spaces
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=255, shape=(96, 96, 2), dtype=np.uint8)

        # plan state across CPs (stored in ordered-axis indices)
        self.x1_idx = np.zeros((self.n_cps,), dtype=np.int32)
        self.x2_idx = np.zeros((self.n_cps,), dtype=np.int32)
        self.d0 = np.zeros((self.n_cps,), dtype=np.int32)

        # base group weights
        self.s0 = np.ones((self.K,), dtype=np.float32)

        # counters
        self.t = 0
        self.cp_idx = 0

        # objective vector in possible-dose space
        self._obj_vec = self._build_objective_vec()

        # stable 96x96 index map
        self._img_index_map = self._build_img_index_map()

        # --- fast dose caches ---
        # mask_cp: (n_cps, K) float32 0/1 in ORIGINAL group index space
        # scale_cp: (n_cps,) float32 = d0/100
        # sum_masked: (K,) float32 = sum_cp( mask_cp[cp] * scale_cp[cp] )
        self._mask_cp = np.zeros((self.n_cps, self.K), dtype=np.float32)
        self._scale_cp = np.zeros((self.n_cps,), dtype=np.float32)
        self._sum_masked = np.zeros((self.K,), dtype=np.float32)

    # -------------------- model + objectives --------------------
    @staticmethod
    def _build_grouped_model(case: OpenKBPCase, K: int) -> GroupedDoseModel:
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

        ptv70 = case.struct_masks.get("PTV70", None)
        if ptv70 is None or int(ptv70.sum()) == 0:
            raise ValueError("PTV70 mask missing/empty in possible-dose space.")

        brainstem = case.struct_masks.get("Brainstem", None)
        if brainstem is not None and int(brainstem.sum()) == 0:
            brainstem = None

        spinalcord = case.struct_masks.get("SpinalCord", None)
        if spinalcord is not None and int(spinalcord.sum()) == 0:
            spinalcord = None

        dose_ref = case.dose_ref
        ptv70_ref_mean = float(dose_ref[ptv70].mean())

        brainstem_ref_mean = float(dose_ref[brainstem].mean()) if brainstem is not None else 0.0
        spinalcord_ref_mean = float(dose_ref[spinalcord].mean()) if spinalcord is not None else 0.0

        # Calibration so that s~=1 yields PTV70 mean near reference when fully open
        ones = np.ones((K,), dtype=np.float32)
        ptv_mean_per_unit = float((B @ ones)[ptv70].mean())
        if ptv_mean_per_unit <= 1e-8:
            raise ValueError("Calibration failed: ptv_mean_per_unit is ~0.")
        gain = float(ptv70_ref_mean / ptv_mean_per_unit)
        B *= np.float32(gain)

        # Heuristic order by PTV/OAR ratio
        ptv_gain = np.asarray(B[ptv70].mean(axis=0)).reshape(-1)
        oar_mask = None
        if brainstem is not None and spinalcord is not None:
            oar_mask = (brainstem | spinalcord)
        elif brainstem is not None:
            oar_mask = brainstem
        elif spinalcord is not None:
            oar_mask = spinalcord

        if oar_mask is None:
            order = np.arange(K, dtype=np.int32)
        else:
            oar_gain = np.asarray(B[oar_mask].mean(axis=0)).reshape(-1)
            ratio = ptv_gain / (oar_gain + 1e-6)
            order = np.argsort(-ratio).astype(np.int32)

        return GroupedDoseModel(
            B=B,
            ptv70_mask=ptv70,
            brainstem_mask=brainstem,
            spinalcord_mask=spinalcord,
            ptv70_ref_mean=ptv70_ref_mean,
            brainstem_ref_mean=brainstem_ref_mean,
            spinalcord_ref_mean=spinalcord_ref_mean,
            gain=gain,
            order=order,
        )

    def _build_objective_vec(self) -> np.ndarray:
        nV = int(self.model.B.shape[0])
        obj = np.zeros((nV,), dtype=np.float32)

        obj[self.model.ptv70_mask] = np.float32(self.model.ptv70_ref_mean)
        if self.model.brainstem_mask is not None:
            obj[self.model.brainstem_mask] = np.float32(self.model.brainstem_ref_mean)
        if self.model.spinalcord_mask is not None:
            obj[self.model.spinalcord_mask] = np.float32(self.model.spinalcord_ref_mean)

        return obj

    # -------------------- stable 96x96 mapping in possible-dose space --------------------
    def _case_seed(self) -> int:
        cid = (self.case_id or "").encode("utf-8")
        h = zlib.adler32(cid) & 0xFFFFFFFF
        return (self.base_seed ^ h) & 0xFFFFFFFF

    def _build_img_index_map(self) -> np.ndarray:
        nV = int(self.model.B.shape[0])
        obj_mask = (self._obj_vec != 0.0)
        obj_idx = np.flatnonzero(obj_mask)
        bg_idx = np.flatnonzero(~obj_mask)

        H = 96 * 96
        rng = np.random.default_rng(self._case_seed())

        if obj_idx.size >= H:
            sel = rng.choice(obj_idx, size=H, replace=False)
            sel = np.sort(sel)
            return sel.astype(np.int32)

        need = H - obj_idx.size
        if bg_idx.size == 0:
            pad = rng.choice(obj_idx, size=need, replace=True) if obj_idx.size > 0 else np.zeros((need,), dtype=np.int64)
        else:
            pad = rng.choice(bg_idx, size=need, replace=(bg_idx.size < need))

        idx = np.concatenate([np.sort(obj_idx), np.sort(pad)])
        rng.shuffle(idx)
        if idx.size != H:
            raise RuntimeError(f"img_index_map size mismatch: {idx.size} vs {H}")
        if np.any(idx < 0) or np.any(idx >= nV):
            raise RuntimeError("img_index_map contains out-of-range indices.")
        return idx.astype(np.int32)

    def _vec_to_96x96(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size != int(self.model.B.shape[0]):
            raise ValueError(f"Expected v size {int(self.model.B.shape[0])}, got {v.size}")
        if self._img_index_map is None or self._img_index_map.size != 96 * 96:
            self._img_index_map = self._build_img_index_map()
        return v[self._img_index_map].reshape(96, 96)

    # -------------------- aperture mask helper (ordered axis -> original group index) --------------------
    def _mask_from_x1x2(self, x1: int, x2: int) -> np.ndarray:
        x1 = int(np.clip(x1, 0, self.K - 2))
        x2 = int(np.clip(x2, x1 + 1, self.K - 1))

        # window positions in ordered axis
        open_pos = np.arange(x1, x2 + 1, dtype=np.int32)
        # map to original group indices
        open_group_indices = self.model.order[open_pos]

        m = np.zeros((self.K,), dtype=np.float32)
        m[open_group_indices] = 1.0
        return m

    # -------------------- fast cache rebuild --------------------
    def _rebuild_plan_cache(self):
        self._scale_cp[:] = (self.d0.astype(np.float32) / 100.0)
        # if all CP share same x1/x2, build one mask and broadcast
        if np.all(self.x1_idx == self.x1_idx[0]) and np.all(self.x2_idx == self.x2_idx[0]):
            m = self._mask_from_x1x2(int(self.x1_idx[0]), int(self.x2_idx[0]))
            self._mask_cp[:] = m[None, :]
            self._sum_masked[:] = m * float(self._scale_cp.sum())
        else:
            self._sum_masked[:] = 0.0
            for cp in range(self.n_cps):
                m = self._mask_from_x1x2(int(self.x1_idx[cp]), int(self.x2_idx[cp]))
                self._mask_cp[cp] = m
                self._sum_masked += m * float(self._scale_cp[cp])

    # -------------------- plan -> dose (fast) --------------------
    def _effective_weights(self) -> np.ndarray:
        # mean over CPs, then apply s0
        return (self._sum_masked / float(self.n_cps)) * self.s0

    def _dose(self) -> np.ndarray:
        w = self._effective_weights()
        return self.model.B @ w

    # -------------------- images --------------------
    def _rotate_2d(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        if nd_rotate is None:
            return img
        return nd_rotate(img, angle=angle_deg, reshape=False, order=0, mode="constant", cval=0)

    def _frame1(self, dose: np.ndarray, theta_deg: float) -> np.ndarray:
        dose_img = self._vec_to_96x96(dose)
        obj_img = self._vec_to_96x96(self._obj_vec)

        ptv_vals = dose[self.model.ptv70_mask]
        ptv_max = float(np.max(ptv_vals)) if ptv_vals.size else float(np.max(dose))
        ptv_max = max(ptv_max, 1e-6)
        scale = ptv_max / 1.08  # paper-like normalization

        dose_n = dose_img / scale
        obj_n = obj_img / scale
        diff = dose_n - obj_n

        frame = (diff + 1.0) * 50.0
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)

        frame[obj_img == 0.0] = 0
        frame = self._rotate_2d(frame, angle_deg=-theta_deg)
        return frame.astype(np.uint8)

    def _frame2_sinogram(self) -> np.ndarray:
        H, W = 96, 96
        img = np.zeros((H, W), dtype=np.uint8)

        denom_d0 = max(1, (self.d0_max - self.d0_min))
        denom_x = max(1, (self.K - 1))

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

        shift = 48 - int(self.cp_idx % H)
        img = np.roll(img, shift=shift, axis=0)
        return img

    def _obs(self, dose: np.ndarray) -> np.ndarray:
        theta_deg = float(self.cp_idx) * (360.0 / float(self.n_cps))
        f1 = self._frame1(dose, theta_deg=theta_deg)
        f2 = self._frame2_sinogram()
        return np.stack([f1, f2], axis=-1).astype(np.uint8)

    # -------------------- reset init calibration --------------------
    def _calibrate_init_to_ptv(self) -> Dict[str, float]:
        dose0 = self._dose()
        ptv_mean0 = float(dose0[self.model.ptv70_mask].mean())
        ref = float(self.model.ptv70_ref_mean)

        eps = 1e-6
        scale = ref / max(ptv_mean0, eps)
        lo, hi = self.init_scale_clip
        scale_applied = float(np.clip(scale, lo, hi))

        self.s0 *= np.float32(scale_applied)

        dose1 = self._dose()
        ptv_mean1 = float(dose1[self.model.ptv70_mask].mean())

        return {
            "init_ptv_mean_before": ptv_mean0,
            "init_ptv_mean_after": ptv_mean1,
            "init_scale_raw": float(scale),
            "init_scale_applied": float(scale_applied),
        }

    # -------------------- Gym API --------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.base_seed = int(seed)
            self.rng = np.random.default_rng(self.base_seed)

        options = options or {}

        # Optional: switch case on reset (for multi-case training)
        new_case_id = options.get("case_id", None)
        if new_case_id is not None and str(new_case_id) != str(self.case_id):
            if not self.root:
                raise ValueError("To switch case_id via options, set OPENKBP_ROOT or pass root=.")
            self.case_id = str(new_case_id)
            self.case = OpenKBPCase.load(self.root, self.case_id)
            self.model = self._build_grouped_model(self.case, self.K)
            self._obj_vec = self._build_objective_vec()
            self._img_index_map = self._build_img_index_map()

        # random base weights around [0.5, 1.0]
        self.s0 = (0.5 + self.rng.random(self.K, dtype=np.float32) * 0.5).astype(np.float32)

        # Initialize plan: centered window + constant d0
        center = self.K // 2
        x1 = int(np.clip(center - self.init_leaf_half_width, 0, self.K - 2))
        x2 = int(np.clip(center + self.init_leaf_half_width, x1 + 1, self.K - 1))

        self.x1_idx[:] = x1
        self.x2_idx[:] = x2
        self.d0[:] = int(np.clip(self.init_d0, self.d0_min, self.d0_max))

        self.t = 0
        self.cp_idx = 0

        # build fast caches for this baseline plan
        self._rebuild_plan_cache()

        calib_info = {}
        if self.calibrate_init:
            calib_info = self._calibrate_init_to_ptv()

        dose = self._dose()
        obs = self._obs(dose)

        info = {
            "case_id": self.case_id,
            "ptv70_ref_mean": float(self.model.ptv70_ref_mean),
            "brainstem_ref_mean": float(self.model.brainstem_ref_mean),
            "spinalcord_ref_mean": float(self.model.spinalcord_ref_mean),
            "calibration_gain": float(self.model.gain),
            "cp_idx": int(self.cp_idx),
            "theta_deg": 0.0,

            "cp_applied": int(self.cp_idx),
            "theta_applied_deg": 0.0,
            "d0": int(self.d0[self.cp_idx]),
            "x1_mm": int(self.x1_idx[self.cp_idx]) * self.leaf_step_mm,
            "x2_mm": int(self.x2_idx[self.cp_idx]) * self.leaf_step_mm,

            "ptv70_mean": float(dose[self.model.ptv70_mask].mean()),
            **calib_info,
        }
        return obs, info

    def step(self, action: int):
        self.t += 1

        a = int(action)
        if a < 0 or a >= 15:
            raise ValueError(f"action must be in [0,14], got {a}")

        dx1_mm, dx2_mm, dd0 = ACTIONS_15[a]

        # apply to CURRENT control point
        cp_applied = int(self.cp_idx)
        theta_applied_deg = float(cp_applied) * (360.0 / float(self.n_cps))

        # old contribution for incremental update
        old_scale = float(self._scale_cp[cp_applied])
        old_mask = self._mask_cp[cp_applied]  # view
        old_term = old_mask * np.float32(old_scale)

        dx1 = int(round(dx1_mm / float(self.leaf_step_mm)))
        dx2 = int(round(dx2_mm / float(self.leaf_step_mm)))

        self.x1_idx[cp_applied] = int(self.x1_idx[cp_applied]) + dx1
        self.x2_idx[cp_applied] = int(self.x2_idx[cp_applied]) + dx2
        self.d0[cp_applied] = int(self.d0[cp_applied]) + int(dd0)

        # constraints on applied cp
        self.d0[cp_applied] = int(np.clip(self.d0[cp_applied], self.d0_min, self.d0_max))
        self.x1_idx[cp_applied] = int(np.clip(self.x1_idx[cp_applied], 0, self.K - 2))
        self.x2_idx[cp_applied] = int(np.clip(self.x2_idx[cp_applied], 1, self.K - 1))
        if self.x2_idx[cp_applied] <= self.x1_idx[cp_applied]:
            self.x2_idx[cp_applied] = int(np.clip(self.x1_idx[cp_applied] + 1, 1, self.K - 1))

        # new mask & scale
        new_scale = float(self.d0[cp_applied]) / 100.0
        new_mask = self._mask_from_x1x2(int(self.x1_idx[cp_applied]), int(self.x2_idx[cp_applied]))
        new_term = new_mask * np.float32(new_scale)

        # write caches
        self._scale_cp[cp_applied] = np.float32(new_scale)
        self._mask_cp[cp_applied] = new_mask

        # incremental update of sum_masked
        self._sum_masked += (new_term - old_term)

        # snapshot applied values for logging
        d0_applied = int(self.d0[cp_applied])
        x1_applied_mm = int(self.x1_idx[cp_applied]) * self.leaf_step_mm
        x2_applied_mm = int(self.x2_idx[cp_applied]) * self.leaf_step_mm

        # advance gantry / control point
        self.cp_idx = (self.cp_idx + 1) % self.n_cps
        theta_deg = float(self.cp_idx) * (360.0 / float(self.n_cps))

        # snapshot next cp values (optional)
        d0_next = int(self.d0[self.cp_idx])
        x1_next_mm = int(self.x1_idx[self.cp_idx]) * self.leaf_step_mm
        x2_next_mm = int(self.x2_idx[self.cp_idx]) * self.leaf_step_mm

        dose = self._dose()

        # metrics (mean-based proxy)
        ptv70_mean = float(dose[self.model.ptv70_mask].mean())
        ref = float(self.model.ptv70_ref_mean) + 1e-6

        base_err = abs(ptv70_mean - float(self.model.ptv70_ref_mean)) / ref
        overdose = max(0.0, ptv70_mean - float(self.model.ptv70_ref_mean)) / ref
        err = float(base_err + overdose)

        bs_mean = float(dose[self.model.brainstem_mask].mean()) if self.model.brainstem_mask is not None else 0.0
        sc_mean = float(dose[self.model.spinalcord_mask].mean()) if self.model.spinalcord_mask is not None else 0.0

        bs_excess = max(0.0, bs_mean - float(self.model.brainstem_ref_mean)) if self.model.brainstem_mask is not None else 0.0
        sc_excess = max(0.0, sc_mean - float(self.model.spinalcord_ref_mean)) if self.model.spinalcord_mask is not None else 0.0

        oar_pen = float((bs_excess + sc_excess) / ref)

        reward = -err - self.oar_lambda * oar_pen

        terminated = (err < 0.05) and (oar_pen < 0.05)
        truncated = (self.t >= self.max_steps)

        obs = self._obs(dose)

        info: Dict[str, Any] = {
            "case_id": self.case_id,

            "cp_idx": int(self.cp_idx),
            "theta_deg": float(theta_deg),

            "cp_applied": int(cp_applied),
            "theta_applied_deg": float(theta_applied_deg),
            "action_index": int(a),
            "dx1_mm": int(dx1_mm),
            "dx2_mm": int(dx2_mm),
            "dd0": int(dd0),

            # legacy keys reflect APPLIED values
            "d0": int(d0_applied),
            "x1_mm": int(x1_applied_mm),
            "x2_mm": int(x2_applied_mm),

            "d0_next": int(d0_next),
            "x1_next_mm": int(x1_next_mm),
            "x2_next_mm": int(x2_next_mm),

            "ptv70_mean": float(ptv70_mean),
            "ptv70_ref_mean": float(self.model.ptv70_ref_mean),
            "brainstem_mean": float(bs_mean),
            "spinalcord_mean": float(sc_mean),
            "brainstem_ref_mean": float(self.model.brainstem_ref_mean),
            "spinalcord_ref_mean": float(self.model.spinalcord_ref_mean),
            "err_norm": float(err),
            "oar_pen_norm": float(oar_pen),
            "calibration_gain": float(self.model.gain),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info
