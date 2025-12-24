# rlfplan/register_envs.py
from __future__ import annotations

import os
from typing import Optional

import gymnasium as gym


_REGISTERED = False


def _getenv(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name, default)
    if v is None:
        raise KeyError(f"Missing required env var: {name}")
    return v


def _getenv_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _getenv_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _getenv_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _make_openkbp_grouped_env():
    """
    Env vars (typical):
      OPENKBP_ROOT (required)
      OPENKBP_CASE (required)
      OPENKBP_K (default 64)
      OPENKBP_MAX_STEPS (default 50)
      OPENKBP_STEP_SCALE (default 0.05)
      OPENKBP_OAR_LAMBDA (default 0.02)
      OPENKBP_SEED (default 0)
    """
    from rlfplan.openkbp_case import OpenKBPCase
    from rlfplan.env_openkbp_grouped import OpenKBPGroupedEnv

    root = _getenv("OPENKBP_ROOT")
    case_id = _getenv("OPENKBP_CASE")
    case = OpenKBPCase(root=root, case_id=case_id)

    K = _getenv_int("OPENKBP_K", 64)
    max_steps = _getenv_int("OPENKBP_MAX_STEPS", 50)
    step_scale = _getenv_float("OPENKBP_STEP_SCALE", 0.05)
    oar_lambda = _getenv_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _getenv_int("OPENKBP_SEED", 0)

    return OpenKBPGroupedEnv(
        case=case,
        K=K,
        max_steps=max_steps,
        step_scale=step_scale,
        oar_lambda=oar_lambda,
        seed=seed,
    )


def _make_openkbp_vmat2d_env():
    """
    Env vars (typical):
      OPENKBP_ROOT (required)
      OPENKBP_CASE (required)   # used only for initial load; reset(options={'case_id':...}) can switch case
      OPENKBP_MAX_STEPS (default 192)  # paper upper bound【Hrinivich&Lee 2020】
      OPENKBP_OAR_LAMBDA (default 0.02)
      OPENKBP_SEED (default 0)

      Optional init knobs (if your env_openkbp_vmat2d.py supports them):
      OPENKBP_INIT_D0 (default 100)
      OPENKBP_INIT_X1_MM (default 120)
      OPENKBP_INIT_X2_MM (default 200)
      OPENKBP_CALIBRATE_INIT (default 1)
    """
    from rlfplan.openkbp_case import OpenKBPCase
    from rlfplan.env_openkbp_vmat2d import OpenKBPVMAT2DEnv

    root = _getenv("OPENKBP_ROOT")
    case_id = _getenv("OPENKBP_CASE")
    case = OpenKBPCase(root=root, case_id=case_id)

    max_steps = _getenv_int("OPENKBP_MAX_STEPS", 192)
    oar_lambda = _getenv_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _getenv_int("OPENKBP_SEED", 0)

    init_d0 = _getenv_int("OPENKBP_INIT_D0", 100)
    init_x1 = _getenv_int("OPENKBP_INIT_X1_MM", 120)
    init_x2 = _getenv_int("OPENKBP_INIT_X2_MM", 200)
    calibrate_init = _getenv_bool("OPENKBP_CALIBRATE_INIT", True)

    return OpenKBPVMAT2DEnv(
        case=case,
        max_steps=max_steps,
        oar_lambda=oar_lambda,
        seed=seed,
        init_d0=init_d0,
        init_x1_mm=init_x1,
        init_x2_mm=init_x2,
        calibrate_init=calibrate_init,
    )


def register_all():
    global _REGISTERED
    if _REGISTERED:
        return

    # Grouped continuous baseline
    gym.register(
        id="OpenKBPGrouped-v0",
        entry_point=_make_openkbp_grouped_env,
    )

    # VMAT2D discrete 15-actions env
    gym.register(
        id="OpenKBPVMAT2D-v0",
        entry_point=_make_openkbp_vmat2d_env,
    )

    _REGISTERED = True


# Side-effect registration (so "import rlfplan.register_envs" is sufficient)
register_all()
