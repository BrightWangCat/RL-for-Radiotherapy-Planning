import numpy as np


def default_hlut(radiation_mode: str = None) -> np.ndarray:
    """Return default HLUT by radiation mode.

    Parameters
    ----------
    radiation_mode : str
        The radiation mode.

    Returns
    -------
    np.ndarray
        The default HLUT.

    Notes
    -----
    Currently, we provide a single hlut independent of the radiation mode.
    """

    hus = [
        [-1024, -999, -90, -45, 0, 100, 350, 3000],
        [0.001, 0.001, 0.95, 0.99, 1, 1.095, 1.199, 2.505],
    ]

    return np.array(hus).transpose()
