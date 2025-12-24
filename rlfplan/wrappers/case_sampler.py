# rlfplan/wrappers/case_sampler.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import gymnasium as gym
import numpy as np


def load_case_list(path: str) -> List[str]:
    """
    Load case ids from a text file: one case id per line.
    Lines starting with '#' are ignored.
    """
    if not path:
        return []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"case list file not found: {path}")
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    if not out:
        raise ValueError(f"case list file is empty: {path}")
    return out


@dataclass
class CaseSamplerConfig:
    case_ids: Sequence[str]
    mode: str = "random"  # "random" | "round_robin"
    seed: int = 0


class CaseSamplerWrapper(gym.Wrapper):
    """
    Ensure every reset selects a case_id and passes it to env.reset(options={'case_id': ...}).

    This is important with VectorEnv autoreset: the vector wrapper will call env.reset() internally,
    and we still want per-episode case sampling to happen.
    """

    def __init__(self, env: gym.Env, cfg: CaseSamplerConfig):
        super().__init__(env)
        if len(cfg.case_ids) <= 0:
            raise ValueError("CaseSamplerWrapper requires non-empty case_ids")
        if cfg.mode not in ("random", "round_robin"):
            raise ValueError(f"Unknown mode: {cfg.mode}")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._rr_i = 0
        self.last_case_id: Optional[str] = None

    def _pick_case(self) -> str:
        if self.cfg.mode == "random":
            return str(self.cfg.case_ids[int(self.rng.integers(0, len(self.cfg.case_ids)))])
        # round_robin
        cid = str(self.cfg.case_ids[self._rr_i % len(self.cfg.case_ids)])
        self._rr_i += 1
        return cid

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        options = {} if options is None else dict(options)

        # If caller explicitly sets case_id, respect it.
        cid = options.get("case_id", None)
        if cid is None:
            cid = self._pick_case()
            options["case_id"] = cid

        self.last_case_id = str(cid)
        return self.env.reset(seed=seed, options=options)
