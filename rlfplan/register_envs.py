# rlfplan/register_envs.py
from __future__ import annotations

import os
import inspect
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


def _load_openkbp_case(root: str, case_id: str):
    """
    Compatible loader for your project’s OpenKBPCase API:
      - Prefer OpenKBPCase.load(root, case_id)
      - Fallback to positional ctor OpenKBPCase(root, case_id)
    """
    from rlfplan.openkbp_case import OpenKBPCase

    if hasattr(OpenKBPCase, "load") and callable(getattr(OpenKBPCase, "load")):
        return OpenKBPCase.load(root, case_id)

    # positional fallback (avoid keyword args that may not exist)
    try:
        return OpenKBPCase(root, case_id)
    except TypeError as e:
        raise TypeError(
            "OpenKBPCase loader failed. Expected OpenKBPCase.load(root, case_id) "
            "or OpenKBPCase(root, case_id) to work."
        ) from e


def _make_openkbp_grouped_env():
    """
    Env vars:
      OPENKBP_ROOT (required)
      OPENKBP_CASE (required)
      OPENKBP_K (default 64)
      OPENKBP_MAX_STEPS (default 50)
      OPENKBP_STEP_SCALE (default 0.05)
      OPENKBP_OAR_LAMBDA (default 0.02)
      OPENKBP_SEED (default 0)
    """
    from rlfplan.env_openkbp_grouped import OpenKBPGroupedEnv

    root = _getenv("OPENKBP_ROOT")
    case_id = _getenv("OPENKBP_CASE")
    case = _load_openkbp_case(root, case_id)

    K = _getenv_int("OPENKBP_K", 64)
    max_steps = _getenv_int("OPENKBP_MAX_STEPS", 50)
    step_scale = _getenv_float("OPENKBP_STEP_SCALE", 0.05)
    oar_lambda = _getenv_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _getenv_int("OPENKBP_SEED", 0)

    # keep ctor call minimal & compatible
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
    Env vars:
      OPENKBP_ROOT (required)
      OPENKBP_CASE (required)   # only initial load; reset(options={'case_id':...}) can switch
      OPENKBP_MAX_STEPS (default 192)
      OPENKBP_OAR_LAMBDA (default 0.02)
      OPENKBP_SEED (default 0)

      Optional init knobs (we auto-adapt to whatever your env supports):
      OPENKBP_INIT_D0 (default 100)
      OPENKBP_INIT_LEAF_HALF_WIDTH (default 8)
      OPENKBP_CALIBRATE_INIT (default 1)
    """
    from rlfplan.env_openkbp_vmat2d import OpenKBPVMAT2DEnv

    root = _getenv("OPENKBP_ROOT")
    case_id = _getenv("OPENKBP_CASE")
    case = _load_openkbp_case(root, case_id)

    max_steps = _getenv_int("OPENKBP_MAX_STEPS", 192)
    oar_lambda = _getenv_float("OPENKBP_OAR_LAMBDA", 0.02)
    seed = _getenv_int("OPENKBP_SEED", 0)

    init_d0 = _getenv_int("OPENKBP_INIT_D0", 100)
    init_leaf_half_width = _getenv_int("OPENKBP_INIT_LEAF_HALF_WIDTH", 8)
    calibrate_init = _getenv_bool("OPENKBP_CALIBRATE_INIT", True)

    # Build kwargs based on actual env signature (avoids mismatch across your iterations)
    sig = inspect.signature(OpenKBPVMAT2DEnv.__init__)
    params = sig.parameters

    kwargs = {}
    # required
    if "case" in params:
        kwargs["case"] = case
    else:
        # extremely unlikely; but keep explicit error
        raise TypeError("OpenKBPVMAT2DEnv.__init__ does not accept 'case' argument.")

    # common knobs
    if "max_steps" in params:
        kwargs["max_steps"] = max_steps
    if "oar_lambda" in params:
        kwargs["oar_lambda"] = oar_lambda
    if "seed" in params:
        kwargs["seed"] = seed

    # init knobs (choose supported names)
    if "init_d0" in params:
        kwargs["init_d0"] = init_d0
    if "init_leaf_half_width" in params:
        kwargs["init_leaf_half_width"] = init_leaf_half_width
    if "calibrate_init" in params:
        kwargs["calibrate_init"] = calibrate_init

    # If your env still uses old names, support them too
    if "init_d0_rate" in params and "init_d0" not in kwargs:
        kwargs["init_d0_rate"] = init_d0

    return OpenKBPVMAT2DEnv(**kwargs)


def register_all():
    global _REGISTERED
    if _REGISTERED:
        return

    gym.register(
        id="OpenKBPGrouped-v0",
        entry_point=_make_openkbp_grouped_env,
    )

    gym.register(
        id="OpenKBPVMAT2D-v0",
        entry_point=_make_openkbp_vmat2d_env,
    )

    _REGISTERED = True


# side-effect registration
register_all()
