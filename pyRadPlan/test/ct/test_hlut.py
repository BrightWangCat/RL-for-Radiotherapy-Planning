import numpy as np
from pyRadPlan.ct import default_hlut


def test_default_hlut():
    """Test default_hlut."""
    hlut = default_hlut()
    assert isinstance(hlut, np.ndarray)
    assert hlut.shape[1] == 2
    assert hlut.shape[0] >= 1
