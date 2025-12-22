import pytest
import numpy as np
import SimpleITK as sitk

# from pyRadPlan.core import Grid
from pyRadPlan.core.resample import resample_image, resample_numpy_array


@pytest.fixture
def sample_image():
    """Create a sample SimpleITK image for testing."""
    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return image


def test_resample_image2grid_with_spacing(sample_image):
    target_spacing = (3, 2, 2)
    resampled_image = resample_image(sample_image, target_grid_spacing=target_spacing)
    assert resampled_image.GetSpacing() == target_spacing
    assert resampled_image.GetSize() == (34, 50, 50)


def test_resample_image2grid_with_target_image(sample_image):
    target_image = sample_image
    resampled_image = resample_image(sample_image, target_image=target_image)
    assert resampled_image.GetSpacing() == target_image.GetSpacing()
    assert resampled_image.GetSize() == target_image.GetSize()


def test_resample_numpy_array(sample_image):
    sample_array = sitk.GetArrayFromImage(sample_image)
    resampled_array = resample_numpy_array(
        sample_array, sample_image, target_grid_spacing=(2, 2, 2)
    )
    assert resampled_array.shape == (50, 50, 50)


def test_resample_numpy_array_with_target_image(sample_image):
    sample_array = sitk.GetArrayFromImage(sample_image)
    resampled_array = resample_numpy_array(sample_array, sample_image, target_image=sample_image)
    assert resampled_array.shape == sample_array.shape
