# preprocess/state_frames_openkbp/openkbp_criteria.py

from __future__ import annotations

def get_openkbp_plan_criteria():
    """
    Values in Gy. PTV entries are D_99 and are negative in the original repo to denote
    'higher is better'. Here we return absolute positive Gy values for objective-map building.
    """
    return {
        "PTV70": 66.5,
        "PTV63": 59.85,
        "PTV56": 53.2,
        "Brainstem": 50.0,
        "SpinalCord": 45.0,
        "Mandible": 73.5,
        "RightParotid": 26.0,
        "LeftParotid": 26.0,
        "Esophagus": 45.0,
        "Larynx": 45.0,
    }
