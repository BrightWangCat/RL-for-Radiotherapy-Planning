import pytest
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+
import pymatreader
import numpy as np
import SimpleITK as sitk
from pyRadPlan.ct import create_ct


@pytest.fixture
def sample_image_3d():
    """Create a sample SimpleITK image for testing."""
    dims = [50, 100, 100]
    sample_array = np.ones(dims) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image, dims


@pytest.fixture
def sample_image_4d(sample_image_3d):
    image, dims_3d = sample_image_3d
    image_4d = sitk.JoinSeries([image, image])
    image_4d.SetOrigin((0, 0, 0, 0))
    image_4d.SetSpacing((1, 1, 2, 1))
    image_4d.SetDirection((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    return image_4d, dims_3d


@pytest.fixture
def generic_input_3d(sample_image_3d):
    cube_hu, dims = sample_image_3d
    mask = np.zeros(dims).astype("uint8")
    mask[0, 0, 1] = 1
    return "VOI", create_ct(cube_hu=cube_hu), mask, 0.2, 0.3


@pytest.fixture
def generic_input_4d(sample_image_4d):
    cube_hu, dims_3d = sample_image_4d
    mask_1 = np.zeros(dims_3d, dtype=np.uint8)
    mask_1[0, 0, 1] = 1
    mask_2 = np.zeros(dims_3d, dtype=np.uint8)
    mask_2[1, 1, 0] = 1
    mask_np = [sitk.GetImageFromArray(mask_1, False), sitk.GetImageFromArray(mask_2, False)]
    mask = sitk.JoinSeries(mask_np)
    return "VOI", create_ct(cube_hu=cube_hu), mask, 0.2, 0.3


@pytest.fixture
def matrad_import():
    """Load a matRad phantom for testing."""
    files = resources.files("pyRadPlan.data.phantoms")
    path = files.joinpath("TG119.mat")
    tmp = pymatreader.read_mat(path)
    return tmp
