import pytest
import numpy as np
import SimpleITK as sitk
from pyRadPlan.core.np2sitk import (
    linear_indices_to_sitk_mask,
    linear_indices_to_grid_coordinates,
    linear_indices_to_image_coordinates,
    sitk_mask_to_linear_indices,
)
from pyRadPlan.core._grids import Grid


@pytest.fixture
def ref_image() -> sitk.Image:
    # Create a reference SimpleITK image
    ref_image = sitk.Image(3, 4, 5, sitk.sitkUInt8)
    ref_image.SetSpacing((1.0, 2.0, 3.0))
    ref_image.SetOrigin((0.0, -20.0, 5.0))
    ref_image.SetDirection(np.eye(3).flatten())
    return ref_image


@pytest.fixture
def grid(ref_image: sitk.Image) -> Grid:
    # Create a reference Grid object
    return Grid.from_sitk_image(ref_image)


def test_linear_indices_to_sitk_mask_sitk_order(ref_image: sitk.Image):
    # Create a reference SimpleITK image

    indices = np.array([0, 4, 25])
    indices_unraveled = np.unravel_index(indices, ref_image.GetSize())

    # Convert linear indices to SimpleITK mask
    mask = linear_indices_to_sitk_mask(indices, ref_image, order="sitk")

    # Check the mask properties
    mask_array = sitk.GetArrayViewFromImage(mask)
    assert mask_array.shape == ref_image.GetSize()[::-1]

    for i in range(len(indices)):
        assert (
            mask.GetPixel(
                int(indices_unraveled[0][i]),
                int(indices_unraveled[1][i]),
                int(indices_unraveled[2][i]),
            )
            == 1
        )

    assert mask.GetPixel(1, 0, 0) == 0


def test_linear_indices_to_sitk_mask_numpy_order(ref_image: sitk.Image):
    # Create a reference SimpleITK image

    indices = np.array([0, 4, 25])
    indices_unraveled = np.unravel_index(indices, ref_image.GetSize(), order="F")

    # Convert linear indices to SimpleITK mask
    mask = linear_indices_to_sitk_mask(indices, ref_image, order="numpy")

    # Check the mask properties
    mask_array = sitk.GetArrayViewFromImage(mask)
    assert mask_array.shape == ref_image.GetSize()[::-1]

    for i in range(len(indices)):
        assert (
            mask.GetPixel(
                int(indices_unraveled[0][i]),
                int(indices_unraveled[1][i]),
                int(indices_unraveled[2][i]),
            )
            == 1
        )

    assert np.all(mask_array.flat[indices] == 1)


def test_linear_indices_to_sitk_mask_invalid_order(ref_image: sitk.Image):
    # dummy indices
    indices = np.array([0, 1])

    # Check for invalid order
    with pytest.raises(ValueError):
        mask = linear_indices_to_sitk_mask(indices, ref_image, order="invalid")


def test_sitk_mask_to_linear_indices_sitk_order(ref_image: sitk.Image):
    indices = np.array([0, 4, 25])

    # Convert linear indices to SimpleITK mask
    mask = linear_indices_to_sitk_mask(indices, ref_image, order="sitk")
    new_indices = sitk_mask_to_linear_indices(mask, order="sitk")
    assert np.all(new_indices == indices)


def test_sitk_mask_to_linear_indices_numpy_order(ref_image: sitk.Image):
    indices = np.array([0, 4, 25])

    # Convert linear indices to SimpleITK mask
    mask = linear_indices_to_sitk_mask(indices, ref_image, order="numpy")
    new_indices = sitk_mask_to_linear_indices(mask, order="numpy")
    assert np.all(new_indices == indices)


def test_sitk_mask_to_linear_indices_invalid_order(ref_image: sitk.Image):
    # dummy indices
    indices = np.array([0, 1])

    # Convert linear indices to SimpleITK mask
    mask = linear_indices_to_sitk_mask(indices, ref_image, order="numpy")

    # Check for invalid order
    with pytest.raises(ValueError):
        new_indices = sitk_mask_to_linear_indices(mask, order="invalid")


def test_linear_indices_to_grid_coordinates_numpy_order(grid: Grid, ref_image: sitk.Image):
    indices = np.array([0, 4, 25])
    indices_unraveled = np.unravel_index(indices, grid.dimensions, order="F")
    coordinates = linear_indices_to_grid_coordinates(indices, grid, index_type="numpy")

    for i in range(len(indices)):
        ix_tuple = (
            int(indices_unraveled[0][i]),
            int(indices_unraveled[1][i]),
            int(indices_unraveled[2][i]),
        )
        assert np.all(
            np.isclose(ref_image.TransformIndexToPhysicalPoint(ix_tuple), coordinates[i, :])
        )


def test_linear_indices_to_grid_coordinates_sitk_order(grid: Grid, ref_image: sitk.Image):
    indices = np.array([0, 4, 25])
    indices_unraveled = np.unravel_index(indices, grid.dimensions)
    coordinates = linear_indices_to_grid_coordinates(indices, grid, index_type="sitk")

    for i in range(len(indices)):
        ix_tuple = (
            int(indices_unraveled[0][i]),
            int(indices_unraveled[1][i]),
            int(indices_unraveled[2][i]),
        )
        assert np.all(
            np.isclose(ref_image.TransformIndexToPhysicalPoint(ix_tuple), coordinates[i, :])
        )


def test_linear_indices_to_grid_coordinates_invalid_order(grid: Grid):
    indices = np.array([0, 1])

    with pytest.raises(ValueError):
        coordinates = linear_indices_to_grid_coordinates(indices, grid, index_type="invalid")


def test_linear_indices_to_image_coordinates(grid: Grid, ref_image: sitk.Image):
    indices = np.array([0, 4, 25])
    assert np.all(
        np.isclose(
            linear_indices_to_grid_coordinates(indices, grid),
            linear_indices_to_image_coordinates(indices, ref_image),
        )
    )

    assert np.all(
        np.isclose(
            linear_indices_to_grid_coordinates(indices, grid, index_type="numpy"),
            linear_indices_to_image_coordinates(indices, ref_image, index_type="numpy"),
        )
    )

    with pytest.raises(ValueError):
        coordinates = linear_indices_to_image_coordinates(indices, ref_image, index_type="invalid")
