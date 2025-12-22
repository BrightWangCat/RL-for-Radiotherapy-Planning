"""Geometry functions for LPS system."""

import numpy as np


def get_gantry_rotation_matrix(gantry_angle):
    """
    Calculate the rotation matrix for gantry.

    Represents an active, counter-clockwise rotation Matrix for Gantry around z
    with pre-multiplication of the matrix (R*x).

    Parameters
    ----------
    gantry_angle : float
        The angle of gantry rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The rotation matrix.

    Notes
    -----
    Gantry rotation is physically an active rotation of a beam
    vector around the target / isocenterin the patient coordinate
    system

    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.
    """

    gantry_angle = np.deg2rad(gantry_angle)

    r_gantry = np.array(
        [
            [np.cos(gantry_angle), -np.sin(gantry_angle), 0],
            [np.sin(gantry_angle), np.cos(gantry_angle), 0],
            [0, 0, 1],
        ]
    )

    return r_gantry


def get_couch_rotation_matrix(couch_angle):
    """
    Calculate the rotation matrix for the couch.

    Parameters
    ----------
    couch_angle : float
        The angle of couch rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The rotation matrix.

    Notes
    -----
    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.
    """

    couch_angle = np.deg2rad(couch_angle)

    r_gantry = np.array(
        [
            [np.cos(couch_angle), 0.0, np.sin(couch_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(couch_angle), 0.0, np.cos(couch_angle)],
        ]
    )

    return r_gantry


def get_beam_rotation_matrix(gantry_angle, couch_angle):
    """
    Calculate the rotation matrix for gantry and couch angles.

    Represents active, counter-clockwise rotation for couch around y
    with pre-multiplication of the matrix (R*x)

    Parameters
    ----------
    gantry_angle : float
        The angle of gantry rotation in degrees.
    couch_angle : float
        The angle of couch rotation in degrees.

    Returns
    -------
    numpy.ndarray
        The rotation matrix.

    Notes
    -----
    Couch rotation is physically a passive rotation of the
    patient system around the beam target point / isocenter

    The LPS system is right-handed, where the gantry rotates counter-clockwise around the z-axis
    and the couch rotates counter-clockwise around the y-axis.
    The gantry rotation is an active rotation of a beam vector around the target/isocenter
    in the patient coordinate system.

    Examples
    --------
    >>> get_beam_rotation_matrix(90, 45)
    array([[-0.70710678, -0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ],
           [ 0.70710678, -0.70710678,  0.        ]])
    """

    r_gantry = get_gantry_rotation_matrix(gantry_angle)
    r_couch = get_couch_rotation_matrix(couch_angle)

    rot_mat = r_couch @ r_gantry

    return rot_mat
