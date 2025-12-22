"""Handlers for Matlab files."""

import os
from pyRadPlan.io import matfile


class MatlabFileHandler:
    """Handler for Matlab files."""

    def __init__(self, temp_path):
        self.tempPath = temp_path

    def load(self, *args):
        mat_file_contents = {}
        for arg in args:
            mat_file_contents[arg] = matfile.load(os.path.join(self.tempPath, f"{arg}.mat"))
        return mat_file_contents

    def save(self, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                tmp_value = {key: value}
            else:
                tmp_value = value
            matfile.save(os.path.join(self.tempPath, f"{key}.mat"), tmp_value)

    def delete(self, *args):
        for arg in args:
            file_path = os.path.join(self.tempPath, f"{arg}.mat")
            os.remove(file_path)
            setattr(self, arg, {})
