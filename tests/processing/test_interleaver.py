import pytest

from gia.processing.interleaver import Interleaver, indexing_from_nested, extend_dol


def test_indexing_from_nested_basic():
    nested_dict = {"outer_key1": {"inner_key": [1, 2, 3]}, "outer_key2": {"inner_key": [4, 5, 6]}}
    result = indexing_from_nested(nested_dict, 1)
    expected_result = {"outer_key1": {"inner_key": 2}, "outer_key2": {"inner_key": 5}}
    assert result == expected_result


def test_indexing_from_nested_out_of_range():
    nested_dict = {"outer_key1": {"inner_key": [1, 2, 3]}, "outer_key2": {"inner_key": [4, 5, 6]}}
    with pytest.raises(IndexError):
        indexing_from_nested(nested_dict, 10)


def test_indexing_from_nested_none_support():
    nested_dict = {
        "outer_key1": {"inner_key1": [None, 2, 3], "inner_key2": None},
        "outer_key2": {"inner_key": None},
        "outer_key3": None,
    }
    result = indexing_from_nested(nested_dict, 0)
    expected_result = None
    assert result == expected_result

    result = indexing_from_nested(nested_dict, 1)
    expected_result = {"outer_key1": {"inner_key": 2}}

    result = indexing_from_nested(nested_dict, 2)
    expected_result = {"outer_key1": {"inner_key": 3}}


def test_indexing_from_nested_type_error():
    with pytest.raises(TypeError):
        nested_dict = {"outer_key1": {"inner_key": [1, 2]}, "outer_key2": {"inner_key": 2}}
        indexing_from_nested(nested_dict, 1)


def test_extend_dol_no_overlap():
    dol = {"key1": [1, 2], "key2": [3, 4]}
    other_dol = {"key3": [5, 6], "key4": [7, 8]}
    extend_dol(dol, other_dol)
    assert dol == {
        "key1": [1, 2, None, None],
        "key2": [3, 4, None, None],
        "key3": [None, None, 5, 6],
        "key4": [None, None, 7, 8],
    }


def test_extend_dol_partial_overlap():
    dol = {"key1": [1, 2], "key2": [3, 4]}
    other_dol = {"key1": [5, 6], "key3": [7, 8]}
    extend_dol(dol, other_dol)
    assert dol == {
        "key1": [1, 2, 5, 6],
        "key2": [3, 4, None, None],
        "key3": [None, None, 7, 8],
    }


def test_extend_dol_total_overlap():
    dol = {"key1": [1, 2], "key2": [3, 4]}
    other_dol = {"key1": [5, 6], "key2": [7, 8]}
    extend_dol(dol, other_dol)
    assert dol == {"key1": [1, 2, 5, 6], "key2": [3, 4, 7, 8]}


def test_extend_dol_with_empty_dol():
    dol = {}
    other_dol = {"key1": [1, 2], "key2": [3, 4]}
    extend_dol(dol, other_dol)
    assert dol == {"key1": [1, 2], "key2": [3, 4]}


def test_extend_dol_with_empty_other_dol():
    dol = {"key1": [1, 2], "key2": [3, 4]}
    other_dol = {}
    extend_dol(dol, other_dol)
    assert dol == {"key1": [1, 2], "key2": [3, 4]}


def test_interleave_episode_single_inner_key():
    # Here, inner keys are "a", "b", ... but in reality they are "input_ids", "patches", ...
    separator = {"a": [99]}
    interleaver = Interleaver(separator=separator)
    sample_data = {
        "image_observations": {"a": [[[1], [3], [5]]]},
        "discrete_actions": {"a": [[[2], [4], [6]]]},
    }
    processed_data = interleaver(sample_data)
    expected_output = {"a": [[1, 99, 2, 3, 99, 4, 5, 99, 6]]}
    assert processed_data == expected_output


def test_interleave_episode_multiple_inner_key():
    # Here, inner keys are "a", "b", ... but in reality they are "input_ids", "patches", ...
    separator = {"a": [99], "b": [88]}
    interleaver = Interleaver(separator=separator)
    sample_data = {
        "image_observations": {"a": [[[1], [3], [5]]], "b": [[[7], [9], [11]]]},
        "discrete_actions": {"a": [[[2], [4], [6]]], "b": [[[8], [10], [12]]]},
    }
    processed_data = interleaver(sample_data)
    expected_output = {
        "a": [[1, 99, 2, 3, 99, 4, 5, 99, 6]],
        "b": [[7, 88, 8, 9, 88, 10, 11, 88, 12]],
    }
    assert processed_data == expected_output


def test_interleave_standalone_only_images():
    # Should just return the inner dict
    interleaver = Interleaver()
    data = {"images": {"a": [[1, 2, 3]]}}
    processed_data = interleaver(data)
    assert processed_data == {"a": [[1, 2, 3]]}


def test_interleave_standalone_only_text():
    # Should just return the inner dict
    interleaver = Interleaver()
    data = {"text": {"a": [[1, 2, 3]]}}
    processed_data = interleaver(data)
    assert processed_data == {"a": [[1, 2, 3]]}


def test_interleave_standalone_images_and_text_different_keys():
    interleaver = Interleaver()
    data = {
        "images": {"a": [[1, 2, 3]]},
        "text": {"b": [[4, 5, 6]]},
    }
    processed_data = interleaver(data)
    assert processed_data == {
        "a": [[1, 2, 3, None, None, None]],
        "b": [[None, None, None, 4, 5, 6]],
    }


def test_interleave_standalone_images_and_text_same_keys():
    interleaver = Interleaver()
    data = {
        "images": {"a": [[1, 2, 3]]},
        "text": {"a": [[4, 5, 6]]},
    }
    processed_data = interleaver(data)
    assert processed_data == {"a": [[1, 2, 3, 4, 5, 6]]}


def test_allow_observations_1_timestep_longer_than_actions():
    # useful for evaluation, where we want to infer the next action
    interleaver = Interleaver()
    data = {
        "image_observations": {"a": [[[1], [3], [5]]]},
        "discrete_actions": {"a": [[[2], [4]]]},
    }
    processed_data = interleaver(data)
    assert processed_data == {"a": [[1, 2, 3, 4, 5]]}
