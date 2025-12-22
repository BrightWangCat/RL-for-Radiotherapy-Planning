# rlfplan/register_envs.py
from __future__ import annotations

import os
from typing import Any, Dict

import gymnasium as gym
from gymnasium.envs.registration import register

from rlfplan.openkbp_case import OpenKBPCase
from rlfplan.env_openkbp_grouped import OpenKBPGroupedEnv

_CASE_CACHE: Dict[str, OpenKBPCase] = {}

def _make_openkbp_grouped_env(**kwargs: Any):
    # Read config from environment variables (so gym.make() needs no kwargs)
    root = os.environ.get(
        "OPENKBP_ROOT",
        "/fs/scratch/PCON0023/mingshiw/PlanData/open-kbp-opt-data/reference-plans",
    )
    case_id = os.environ.get("OPENKBP_CASE", "pt_241")
    K = int(os.environ.get("OPENKBP_K", "64"))
    max_steps = int(os.environ.get("OPENKBP_MAX_STEPS", "50"))
    step_scale = float(os.environ.get("OPENKBP_STEP_SCALE", "0.05"))
    oar_lambda = float(os.environ.get("OPENKBP_OAR_LAMBDA", "0.02"))
    seed = int(os.environ.get("OPENKBP_SEED", "0"))

    case = _CASE_CACHE.get(case_id)
    if case is None:
        case = OpenKBPCase.load(root, case_id)
        _CASE_CACHE[case_id] = case

    return OpenKBPGroupedEnv(
        case=case,
        K=K,
        max_steps=max_steps,
        step_scale=step_scale,
        oar_lambda=oar_lambda,
        seed=seed,
    )

def register_all():
    # Safe re-register
    try:
        register(id="OpenKBPGrouped-v0", entry_point=_make_openkbp_grouped_env)
    except Exception:
        pass

# Register on import
register_all()
