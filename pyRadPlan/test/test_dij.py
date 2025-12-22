import pytest
import SimpleITK as sitk
import numpy as np
import scipy.sparse as sp
from pyRadPlan.core import Grid
from pyRadPlan.dij import Dij, create_dij, validate_dij, compose_beam_dijs


@pytest.fixture
def sample_dij_dict():
    dose_information = {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (334, 334, 214),
            "num_of_voxels": 334 * 334 * 214,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (167, 167, 107),
            "num_of_voxels": 167 * 167 * 107,
        },
        "physical_dose": [[[sp.csc_matrix((167 * 167 * 107, 6506), dtype=float)]]],
        "num_of_beams": 1,
        "total_num_of_bixels": 6506,
        "bixel_num": np.ones(6506),
        "ray_num": np.ones(6506),
        "beam_num": np.ones(6506),
    }
    return dose_information


@pytest.fixture
def sample_dij_dict_camel():
    dose_information = {
        "ctGrid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (167, 167, 107),
            "numOfVoxels": 167 * 167 * 107,
        },
        "doseGrid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (167, 167, 107),
            "numOfVoxels": 167 * 167 * 107,
        },
        "physicalDose": [[[sp.csc_matrix((167 * 167 * 107, 6506), dtype=float)]]],
        "numOfBeams": 1,
        "totalNumOfBixels": 6506,
        "bixelNum": np.ones(6506),
        "rayNum": np.ones(6506),
        "beamNum": np.ones(6506),
    }
    return dose_information


def test_dij_creation_from_dict(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    assert isinstance(dij, Dij)
    assert isinstance(dij.dose_grid, Grid)
    assert isinstance(dij.ct_grid, Grid)
    assert isinstance(dij.physical_dose, np.ndarray)
    assert isinstance(dij.physical_dose.flat[0], sp.csc_matrix)
    assert dij.total_num_of_bixels == 6506
    assert dij.num_of_voxels == 167 * 167 * 107
    assert dij.bixel_num.shape == (6506,)
    assert dij.ray_num.shape == (6506,)
    assert dij.beam_num.shape == (6506,)


def test_dij_creation_from_camel_dict(sample_dij_dict_camel):
    dij = validate_dij(**sample_dij_dict_camel)
    assert isinstance(dij, Dij)
    assert isinstance(dij.dose_grid, Grid)
    assert isinstance(dij.ct_grid, Grid)
    assert isinstance(dij.physical_dose, np.ndarray)
    assert isinstance(dij.physical_dose.flat[0], sp.csc_matrix)
    assert dij.total_num_of_bixels == 6506
    assert dij.num_of_voxels == 167 * 167 * 107


def test_dij_creation_invalid_physical_dose_shape(sample_dij_dict):
    sample_dij_dict["physical_dose"] = sp.csr_matrix((100, 4), dtype=float)  # Invalid shape
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_creation_invalid_beam_num(sample_dij_dict):
    sample_dij_dict["beam_num"] = np.ones(5)  # Invalid shape
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_creation_invalid_ray_num(sample_dij_dict):
    sample_dij_dict["ray_num"] = np.ones(5)  # Invalid shape
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_creation_invalid_bixel_num(sample_dij_dict):
    sample_dij_dict["bixel_num"] = np.ones(5)  # Invalid shape
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_creation_invalid_unique_indices_in_beam_num(sample_dij_dict):
    # number_of_beams = 1
    sample_dij_dict["beam_num"][3000:] = 2
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_creation_invalid_number_of_beams(sample_dij_dict):
    # beam_num only ones
    sample_dij_dict["num_of_beams"] = 2
    with pytest.raises(ValueError):
        create_dij(**sample_dij_dict)


def test_dij_to_matrad(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    matrad_dict = dij.to_matrad()
    assert isinstance(matrad_dict, dict)
    assert "ctGrid" in matrad_dict
    assert "doseGrid" in matrad_dict
    assert "totalNumOfBixels" in matrad_dict
    assert isinstance(matrad_dict["totalNumOfBixels"], int)
    assert isinstance(matrad_dict["ctGrid"], dict)
    assert isinstance(matrad_dict["doseGrid"], dict)
    assert matrad_dict["totalNumOfBixels"] == 6506


def test_dij_to_matrad_from_csc_matrix(sample_dij_dict):
    sample_dij_dict2 = sample_dij_dict
    sample_dij_dict2["physical_dose"][0][0][0] = sp.csc_matrix(
        (167 * 167 * 107, 6506), dtype=float
    )
    dij = create_dij(**sample_dij_dict2)
    matrad_dict = dij.to_matrad()
    assert isinstance(matrad_dict, dict)
    assert isinstance(matrad_dict["physicalDose"].flat[0], sp.csc_matrix)


def test_dij_to_matrad_from_csc_array(sample_dij_dict):
    sample_dij_dict["physical_dose"][0][0][0] = sp.csc_array((167 * 167 * 107, 6506), dtype=float)
    dij = create_dij(**sample_dij_dict)
    matrad_dict = dij.to_matrad()
    assert isinstance(matrad_dict, dict)
    assert isinstance(matrad_dict["physicalDose"].flat[0], sp.csc_matrix)


def test_dij_to_matrad_from_csr_array(sample_dij_dict):
    sample_dij_dict["physical_dose"][0][0][0] = sp.csr_array((167 * 167 * 107, 6506), dtype=float)
    dij = create_dij(**sample_dij_dict)
    matrad_dict = dij.to_matrad()
    assert isinstance(matrad_dict, dict)
    assert isinstance(matrad_dict["physicalDose"].flat[0], sp.csc_matrix)


def test_create_dij_from_grids():
    ct_dict = {
        "resolution": {"x": 1, "y": 1, "z": 1},
        "dimensions": (167, 167, 107),
    }
    dose_dict = {
        "resolution": {"x": 1, "y": 1, "z": 1},
        "dimensions": (167, 167, 107),
    }
    ct_grid = Grid.model_validate(ct_dict)
    dose_grid = Grid.model_validate(dose_dict)

    sample_dij_dict = {
        "physical_dose": sp.csr_matrix((167 * 167 * 107, 6506), dtype=float),
        "num_of_beams": 1,
        "total_num_of_bixels": 6506,
        "bixelNum": np.ones(6506),
        "rayNum": np.ones(6506),
        "beamNum": np.ones(6506),
    }
    create_dij(ct_grid=ct_grid, dose_grid=dose_grid, **sample_dij_dict)


def test_result_computation_physical_dose_dose_grid(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    f = np.ones((dij.total_num_of_bixels,), dtype=np.float32)
    result = dij.compute_result_dose_grid(f)
    assert isinstance(result, dict)
    assert "physical_dose" in result
    assert isinstance(result["physical_dose"], sitk.Image)
    assert result["physical_dose"].GetSize() == dij.dose_grid.dimensions


def test_result_computation_physical_dose_ct_grid(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    f = np.ones((dij.total_num_of_bixels,), dtype=np.float32)
    result = dij.compute_result_ct_grid(f)
    assert isinstance(result, dict)
    assert "physical_dose" in result
    assert isinstance(result["physical_dose"], sitk.Image)
    assert result["physical_dose"].GetSize() == dij.ct_grid.dimensions


def test_result_computation_let(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    f = np.zeros((dij.total_num_of_bixels,), dtype=np.float32)

    dij.let_dose = dij.physical_dose

    result = dij.compute_result_dose_grid(f)

    assert isinstance(result, dict)
    assert "let" in result
    assert "physical_dose" in result
    assert isinstance(result["let"], sitk.Image)
    assert isinstance(result["physical_dose"], sitk.Image)
    assert result["let"].GetSize() == dij.dose_grid.dimensions
    assert result["physical_dose"].GetSize() == dij.dose_grid.dimensions

    with pytest.raises(ValueError):
        dij.physical_dose = None
        dij.compute_result_dose_grid(f)


def test_result_computation_biodose(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    f = np.zeros((dij.total_num_of_bixels,), dtype=np.float32)

    dij.alpha_dose = dij.physical_dose
    dij.sqrt_beta_dose = dij.physical_dose

    result = dij.compute_result_dose_grid(f)

    assert isinstance(result, dict)
    assert "effect" in result
    assert "physical_dose" in result
    assert isinstance(result["effect"], sitk.Image)
    assert isinstance(result["physical_dose"], sitk.Image)
    assert result["effect"].GetSize() == dij.dose_grid.dimensions
    assert result["physical_dose"].GetSize() == dij.dose_grid.dimensions

    with pytest.raises(ValueError):
        dij.physical_dose = None
        dij.compute_result_dose_grid(f)


def test_compose_beam_dijs(sample_dij_dict):
    dij = create_dij(**sample_dij_dict)
    composed_dij = compose_beam_dijs([dij, dij])
    assert isinstance(composed_dij, Dij)
    assert composed_dij.num_of_beams == 2 * dij.num_of_beams
    assert composed_dij.total_num_of_bixels == 2 * dij.total_num_of_bixels
    assert composed_dij.beam_num.shape == (2 * dij.total_num_of_bixels,)
    assert composed_dij.ray_num.shape == (2 * dij.total_num_of_bixels,)
    assert composed_dij.bixel_num.shape == (2 * dij.total_num_of_bixels,)
    assert composed_dij.dose_grid == dij.dose_grid
    assert composed_dij.ct_grid == dij.ct_grid
