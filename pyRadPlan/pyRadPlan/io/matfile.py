"""Read and write mat-files."""

from os import PathLike
from scipy import io
import pymatreader


def load(path2file: PathLike) -> dict[str]:
    """Load a .mat file and returns the data as a dictionary."""

    try:
        matrad_patient = pymatreader.read_mat(filename=path2file)
    except NotImplementedError:
        matrad_patient = io.loadmat(
            file_name=path2file, mat_dtype=True, squeeze_me=True, struct_as_record=True
        )
    except Exception as exep:
        raise ValueError(f"Could not load the .mat file: {path2file}") from exep

    return matrad_patient


def save(path2file, the_dict: dict[str]):
    """
    Save a dictionary as a .mat file.

    Parameters
    ----------
    path2file : str
        Path to the file.
    the_dict : dict[str]
        Dictionary to be saved.
    """

    io.savemat(file_name=path2file, mdict=the_dict)
