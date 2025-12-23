# rlfplan/env_openkbp_grouped.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.sparse as sp

from rlfplan.openkbp_case import OpenKBPCase


@dataclass
class GroupedDoseModel:
    B: np.ndarray               # (nV, K) float32 dense
    ptv70_mask: np.ndarray      # (nV,) bool
    brainstem_mask: Optional[np.ndarray]   # (nV,) bool or None
    spinalcord_mask: Optional[np.ndarray]  # (nV,) bool or None
    ptv70_ref_mean: float       # float
    gain: float                 # calibration gain applied to B
    brainstem_ref_mean: float   # reference mean in (possible-dose ∩ structure)
    spinalcord_ref_mean: float  # reference mean in (possible-dose ∩ structure)


class OpenKBPGroupedEnv(gym.Env):
    """
    Minimal single-case environment (Grouped beamlets -> low-dim control).

    Action (Box): delta on K group weights in [-1, 1]
    Internal state: s in R^K, constrained s >= 0
    Dose model: dose = B @ s
    Observation: [ptv70_mean, brainstem_mean, spinalcord_mean, step_fraction]

    Reward (normalized):
      - PTV: symmetric mean-dose error + extra overdose penalty
      - OAR: hinge penalty only for exceeding reference-plan OAR mean
      reward = -err_norm - oar_lambda * oar_pen_norm

    Notes:
      - B is calibrated so that s ~ 1 yields PTV70 mean near reference.
      - OAR reference means are computed on the same voxel subset (possible-dose space).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        case: OpenKBPCase,
        K: int = 64,
        max_steps: int = 50,
        step_scale: float = 0.05,
        oar_lambda: float = 0.02,
        seed: int = 0,
    ):
        super().__init__()
        self.case = case
        self.K = int(K)
        self.max_steps = int(max_steps)
        self.step_scale = float(step_scale)
        self.oar_lambda = float(oar_lambda)
        self.rng = np.random.default_rng(seed)

        self.model = self._build_grouped_model(case, self.K)

        # Action: continuous deltas in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.K,), dtype=np.float32
        )

        # Observation: 4 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.s = np.zeros((self.K,), dtype=np.float32)
        self.t = 0

    @staticmethod
    def _build_grouped_model(case: OpenKBPCase, K: int) -> GroupedDoseModel:
        A: sp.csr_matrix = case.A  # (nV, nB) possible-dose subspace
        nV, nB = A.shape
        K = min(int(K), int(nB))
        if K <= 0:
            raise ValueError("K must be >= 1")

        # contiguous block grouping
        edges = np.linspace(0, nB, K + 1, dtype=np.int32)

        # Build dense B (nV, K) by summing column blocks
        B = np.zeros((nV, K), dtype=np.float32)

        # Use CSC for efficient column slicing
        Acsc = A.tocsc()
        for g in range(K):
            c0, c1 = int(edges[g]), int(edges[g + 1])
            if c1 <= c0:
                continue
            Bg = Acsc[:, c0:c1].sum(axis=1)  # (nV, 1)
            B[:, g] = np.asarray(Bg).reshape(-1).astype(np.float32)

        # Masks (in possible-dose space)
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

        brainstem_ref_mean = 0.0
        if brainstem is not None:
            brainstem_ref_mean = float(dose_ref[brainstem].mean())

        spinalcord_ref_mean = 0.0
        if spinalcord is not None:
            spinalcord_ref_mean = float(dose_ref[spinalcord].mean())

        # ---- Calibration: scale B so that s~=1 yields PTV70 mean near reference ----
        ones = np.ones((K,), dtype=np.float32)
        ptv_mean_per_unit = float((B @ ones)[ptv70].mean())
        if ptv_mean_per_unit <= 1e-8:
            raise ValueError("Calibration failed: ptv_mean_per_unit is ~0.")
        gain = float(ptv70_ref_mean / ptv_mean_per_unit)
        B *= np.float32(gain)

        return GroupedDoseModel(
            B=B,
            ptv70_mask=ptv70,
            brainstem_mask=brainstem,
            spinalcord_mask=spinalcord,
            ptv70_ref_mean=ptv70_ref_mean,
            gain=gain,
            brainstem_ref_mean=brainstem_ref_mean,
            spinalcord_ref_mean=spinalcord_ref_mean,
        )

    def _dose(self) -> np.ndarray:
        return self.model.B @ self.s

    def _obs_from_dose(self, d: np.ndarray) -> np.ndarray:
        ptv70_mean = float(d[self.model.ptv70_mask].mean())

        bs_mean = 0.0
        if self.model.brainstem_mask is not None:
            bs_mean = float(d[self.model.brainstem_mask].mean())

        sc_mean = 0.0
        if self.model.spinalcord_mask is not None:
            sc_mean = float(d[self.model.spinalcord_mask].mean())

        step_frac = float(self.t / self.max_steps)
        return np.asarray([ptv70_mean, bs_mean, sc_mean, step_frac], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Start around ~[0.5, 1.0] so that calibrated B yields reasonable dose magnitudes
        self.s = (0.5 + self.rng.random(self.K, dtype=np.float32) * 0.5).astype(np.float32)
        self.t = 0

        d = self._dose()
        obs = self._obs_from_dose(d)
        info = {
            "ptv70_ref_mean": self.model.ptv70_ref_mean,
            "brainstem_ref_mean": self.model.brainstem_ref_mean,
            "spinalcord_ref_mean": self.model.spinalcord_ref_mean,
            "calibration_gain": self.model.gain,
        }
        return obs, info

    def step(self, action: np.ndarray):
        self.t += 1

        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)

        # Update weights, enforce non-negativity
        self.s = np.maximum(0.0, self.s + self.step_scale * a)

        d = self._dose()
        ptv70_mean = float(d[self.model.ptv70_mask].mean())

        ref = float(self.model.ptv70_ref_mean) + 1e-6

        # ---- PTV error (normalized) ----
        base_err = abs(ptv70_mean - self.model.ptv70_ref_mean) / ref

        # Extra penalty for overdose to suppress PTV overshoot
        overdose = max(0.0, ptv70_mean - self.model.ptv70_ref_mean) / ref

        # err_norm: symmetric + extra overdose
        err = float(base_err + overdose)

        # ---- OAR hinge penalty: only punish exceeding reference mean ----
        bs_mean = 0.0
        if self.model.brainstem_mask is not None:
            bs_mean = float(d[self.model.brainstem_mask].mean())

        sc_mean = 0.0
        if self.model.spinalcord_mask is not None:
            sc_mean = float(d[self.model.spinalcord_mask].mean())

        bs_excess = 0.0
        if self.model.brainstem_mask is not None:
            bs_excess = max(0.0, bs_mean - float(self.model.brainstem_ref_mean))

        sc_excess = 0.0
        if self.model.spinalcord_mask is not None:
            sc_excess = max(0.0, sc_mean - float(self.model.spinalcord_ref_mean))

        oar_pen = float((bs_excess + sc_excess) / ref)

        reward = -err - self.oar_lambda * oar_pen

        terminated = False
        truncated = (self.t >= self.max_steps)

        obs = self._obs_from_dose(d)
        info = {
            "ptv70_mean": ptv70_mean,
            "ptv70_ref_mean": self.model.ptv70_ref_mean,
            "err_norm": float(err),
            "oar_pen_norm": float(oar_pen),
            "calibration_gain": self.model.gain,
            "brainstem_mean": float(bs_mean),
            "spinalcord_mean": float(sc_mean),
            "brainstem_ref_mean": float(self.model.brainstem_ref_mean),
            "spinalcord_ref_mean": float(self.model.spinalcord_ref_mean),
        }
        return obs, float(reward), terminated, truncated, info
