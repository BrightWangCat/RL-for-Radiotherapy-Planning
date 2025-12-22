import pytest
from pyRadPlan.util import dl2ld, ld2dl, swap_orientation_sparse_matrix
import scipy.sparse as sp
import numpy as np


### Test dl2ld function
def test_dl2ld_valid_input():
    dict_of_lists = {"a": [1, 2, 3], "b": [4, 5, 6]}
    expected_output = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    assert dl2ld(dict_of_lists) == expected_output


def test_dl2ld_empty_input():
    dict_of_lists = {}
    assert dl2ld(dict_of_lists) == []


def test_dl2ld_type_check_false():
    dict_of_lists = {"a": [1, 2, 3], "b": [4, 5, 6]}
    expected_output = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    assert dl2ld(dict_of_lists, type_check=False) == expected_output


def test_dl2ld_type_error_not_dict():
    with pytest.raises(TypeError):
        dl2ld([1, 2, 3])


def test_dl2ld_type_error_values_not_lists():
    dict_of_lists = {"a": [1, 2, 3], "b": 4}
    with pytest.raises(TypeError):
        dl2ld(dict_of_lists)


def test_dl2ld_type_error_lists_different_lengths():
    dict_of_lists = {"a": [1, 2, 3], "b": [4, 5]}
    with pytest.raises(TypeError):
        dl2ld(dict_of_lists)


### Test ld2dl function
def test_ld2dl_valid_input():
    list_of_dicts = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    expected_output = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert ld2dl(list_of_dicts) == expected_output


def test_ld2dl_empty_input():
    list_of_dicts = []
    assert ld2dl(list_of_dicts) == {}


def test_ld2dl_type_check_false():
    list_of_dicts = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    expected_output = {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert ld2dl(list_of_dicts, type_check=False) == expected_output


def test_ld2dl_type_error_not_list():
    with pytest.raises(TypeError):
        ld2dl({"a": 1, "b": 4})


def test_ld2dl_type_error_elements_not_dicts():
    list_of_dicts = [{"a": 1, "b": 4}, [2, 5], {"a": 3, "b": 6}]
    with pytest.raises(TypeError):
        ld2dl(list_of_dicts)


def test_swap_orientation_sparse_matrix():
    original_shape = (2, 3, 4)
    data = [1, 2, 3]
    row_indices = [0, 1, 2]
    col_indices = [1, 2, 3]

    sparse_matrix = sp.csc_matrix(
        (data, (row_indices, col_indices)), shape=(np.prod(original_shape), original_shape[2])
    )
    axes = (0, 1)

    result = swap_orientation_sparse_matrix(sparse_matrix, original_shape, axes)

    result_reversed = swap_orientation_sparse_matrix(
        result, (original_shape[1], original_shape[0], original_shape[2]), axes
    )

    result_data = result.data

    assert np.array_equal(sparse_matrix.toarray(), result_reversed.toarray())
    assert sp.issparse(result)
    assert np.array_equal(data, result_data)
    assert result.shape == sparse_matrix.shape
