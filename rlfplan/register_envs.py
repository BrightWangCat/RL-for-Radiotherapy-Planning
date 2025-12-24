# rlfplan/register_envs.py
from __future__ import annotations

import os
from typing import Optional, Callable

import gymnasium as gym


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if v is None or str(v).strip() == "":
        raise ValueError(
            f"Missing required env var: {name}. "
            f"Set it before running (e.g., export {name}=...)."
        )
    return str(v)


def _get_int(name: str, default: int) -> int:
    v = os.environ.get(name, None)
    return int(v) if v is not None and str(v).strip() != "" else int(default)


def _get_float(name: str, default: float) -> float:
    v = os.environ.get(name, None)
    return float(v) if v is not None and str(v).strip() != "" else float(default)


def _safe_register(id: str, entry_point: Callable):
    # Avoid duplicate registration if module is imported multiple times
    try:
        if id in gym.registry:
            return
    except Exception:
        # older gymnasium versions may not support "in" cleanly
        pass
    try:
        gym.register(id=id, entry_point=entry_point)
    except Exception:
        # if already registered, ignore
        pass


# --------------------------
# OpenKBP Grouped (baseline)
# --------------------------
def make_openkbp_grouped_env(**_kwargs):
    from rlfplan.openkbp_case import OpenKBPCase
    from rlfplan.env_openkbp_grouped import OpenKBPGroupedEnv

    root = _get_env("OPENKBP_ROOT")
    case_id = _get_env("OPENKBP_CASE")

    K = _get_int("OPENKBP_K", 64)
    max_steps = _get_int("OPENKBP_MAX_STEPS", 50)
    step_scale = _get_float("OPENKBP_STEP_SCALE", 0.05)
    oar_lambda = _get_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _get_int("OPENKBP_SEED", 0)

    case = OpenKBPCase.load(root, case_id)
    env = OpenKBPGroupedEnv(
        case=case,
        K=K,
        max_steps=max_steps,
        step_scale=step_scale,
        oar_lambda=oar_lambda,
        seed=seed,
    )
    return env


# --------------------------
# OpenKBP VMAT2D (paper IO)
# --------------------------
def make_openkbp_vmat2d_env(**_kwargs):
    # This env reads root/case from OPENKBP_ROOT/OPENKBP_CASE if not passed.
    from rlfplan.env_openkbp_vmat2d import OpenKBPVMAT2DEnv

    # required
    root = _get_env("OPENKBP_ROOT")
    case_id = _get_env("OPENKBP_CASE")

    # optional knobs
    K = _get_int("OPENKBP_K", 64)
    n_cps = _get_int("OPENKBP_N_CPS", 96)
    max_steps = _get_int("OPENKBP_MAX_STEPS", 192)
    oar_lambda = _get_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _get_int("OPENKBP_SEED", 0)

    d0_min = _get_int("OPENKBP_D0_MIN", 20)
    d0_max = _get_int("OPENKBP_D0_MAX", 600)
    init_d0 = _get_int("OPENKBP_INIT_D0", 100)
    init_leaf_half_width = _get_int("OPENKBP_INIT_LEAF_HALF_WIDTH", 8)

    env = OpenKBPVMAT2DEnv(
        root=root,
        case_id=case_id,
        K=K,
        n_cps=n_cps,
        max_steps=max_steps,
        oar_lambda=oar_lambda,
        seed=seed,
        d0_min=d0_min,
        d0_max=d0_max,
        init_d0=init_d0,
        init_leaf_half_width=init_leaf_half_width,
    )
    return env


# Register IDs used by training/eval scripts
_safe_register("OpenKBPGrouped-v0", make_openkbp_grouped_env)
_safe_register("OpenKBPVMAT2D-v0", make_openkbp_vmat2d_env)
