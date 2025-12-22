import numpy as np
from pyRadPlan.geometry.lps import (
    get_beam_rotation_matrix,
    get_gantry_rotation_matrix,
    get_couch_rotation_matrix,
)


def test_get_gantry_rotation_matrix_90():
    gantry_angle = 90
    expected_matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    result = get_gantry_rotation_matrix(gantry_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_gantry_rotation_matrix_0():
    gantry_angle = 0
    expected_matrix = np.eye(3)
    result = get_gantry_rotation_matrix(gantry_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_gantry_rotation_matrix_180():
    gantry_angle = 180
    expected_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    result = get_gantry_rotation_matrix(gantry_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_gantry_rotation_matrix_360():
    gantry_angle = 360
    expected_matrix = np.eye(3)
    result = get_gantry_rotation_matrix(gantry_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_couch_rotation_matrix_90():
    couch_angle = 90
    expected_matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    result = get_couch_rotation_matrix(couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_couch_rotation_matrix_0():
    couch_angle = 0
    expected_matrix = np.eye(3)
    result = get_couch_rotation_matrix(couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_couch_rotation_matrix_180():
    couch_angle = 180
    expected_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    result = get_couch_rotation_matrix(couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_couch_rotation_matrix_360():
    couch_angle = 360
    expected_matrix = np.eye(3)
    result = get_couch_rotation_matrix(couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_beam_rotation_matrix_90_45():
    gantry_angle = 90
    couch_angle = 45
    a45 = np.sin(np.deg2rad(45))
    expected_matrix = np.array([[0.0, -a45, a45], [1.0, 0.0, 0.0], [0.0, a45, a45]])
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_beam_rotation_matrix_0_0():
    gantry_angle = 0
    couch_angle = 0
    expected_matrix = np.eye(3)
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_beam_rotation_matrix_45_90():
    gantry_angle = 45
    couch_angle = 90
    a45 = np.sin(np.deg2rad(45))
    expected_matrix = np.array([[0.0, 0, 1.0], [a45, a45, 0.0], [-a45, a45, 0.0]])
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)


def test_get_beam_rotation_matrix_360_360():
    gantry_angle = 360
    couch_angle = 360
    expected_matrix = np.eye(3)
    result = get_beam_rotation_matrix(gantry_angle, couch_angle)
    np.testing.assert_almost_equal(result, expected_matrix, decimal=6)
