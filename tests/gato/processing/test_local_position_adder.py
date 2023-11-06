import pytest

from gato.processing.local_positions_adder import LocalPositionsAdder


def test_local_positions_adder_with_overlapping_key_groups():
    with pytest.raises(ValueError):
        LocalPositionsAdder([["a", "b"], ["b", "c"]])


def test_local_positions_adder_with_inconsistent_batch_size_inner():
    adder = LocalPositionsAdder([["a"]])
    input_dict = {
        "a": {
            "aa": [  # batch_size = 2
                [[0]],
                [[1]],
            ],
            "ab": [  # batch_size = 1
                [[2]],
            ],
        },
    }
    with pytest.raises(ValueError):
        adder(input_dict)


def test_local_positions_adder_with_inconsistent_batch_size_outer():
    adder = LocalPositionsAdder([["a", "b"]])
    input_dict = {
        "a": {
            "aa": [  # batch_size = 2
                [[0]],
                [[1]],
            ],
        },
        "b": {
            "ba": [  # batch_size = 1
                [[2]],
            ],
        },
    }
    with pytest.raises(ValueError):
        adder(input_dict)


def test_local_positions_adder_with_inconsistent_seq_len_between_sequences():
    adder = LocalPositionsAdder([["a"]])
    input_dict = {
        "a": {
            "aa": [
                [[0], [1]],  # seq_len = 2
                [[2], [3], [4]],  # seq_len = 3
            ],
        },
    }
    adder(input_dict)
    local_positions = input_dict["a"]["local_positions"]
    expected_local_position = [
        [[0], [0]],
        [[0], [0], [0]],
    ]
    assert local_positions == expected_local_position


def test_local_positions_adder_with_inconsistent_seq_len_between_inner_keys():
    adder = LocalPositionsAdder([["a"]])
    input_dict = {
        "a": {
            "aa": [
                [[0], [1]],  # seq_len = 2
            ],
            "ab": [
                [[2], [3], [4]],  # seq_len = 3
            ],
        },
    }
    with pytest.raises(ValueError):
        adder(input_dict)


def test_local_positions_adder_with_inconsistent_seq_len_between_outer_keys():
    adder = LocalPositionsAdder([["a", "b"]])
    input_dict = {
        "a": {
            "aa": [
                [[0], [1]],  # seq_len = 2
            ],
        },
        "b": {
            "ba": [
                [[2], [3], [4]],  # seq_len = 3
            ]
        },
    }
    with pytest.raises(ValueError):
        adder(input_dict)


def test_local_positions_adder_with_inconsistent_data_size():
    adder = LocalPositionsAdder([["a"]])
    input_dict = {
        "a": {
            "aa": [
                [[0], [1, 2]],  # seq_len = 2
            ],
        },
    }
    adder(input_dict)
    local_positons = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0], [0, 1]]]
    assert local_positons == expected_local_positions


def test_local_positions_adder_missing_key_in_groups():
    adder = LocalPositionsAdder([["a"]])
    input_dict = {
        "a": {"aa": [[[0]]]},
        "b": {"aa": [[[1]]]},  # b in not in groups
    }
    adder(input_dict)
    local_positons = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0]]]
    assert local_positons == expected_local_positions
    assert "local_positions" not in input_dict["b"]


def test_local_positions_adder_missing_key_in_input_data():
    adder = LocalPositionsAdder([["a", "b"]])  # b in not in data
    input_dict = {"a": {"aa": [[[0]]]}}
    adder(input_dict)
    local_positons = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0]]]
    assert local_positons == expected_local_positions


def test_local_positions_adder_missing_group_in_input_data():
    adder = LocalPositionsAdder([["a"], ["b"]])  # b in not in data
    input_dict = {"a": {"aa": [[[0]]]}}
    adder(input_dict)
    local_positons = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0]]]
    assert local_positons == expected_local_positions


def test_valid_local_positions_adder_single():
    adder = LocalPositionsAdder([["a"]])
    sequence = [[0], [1, 2], [3, 4, 5], [6, 7]]
    input_dict = {"a": {"aa": [sequence]}}
    adder(input_dict)
    local_positions = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0], [0, 1], [0, 1, 2], [0, 1]]]
    assert local_positions == expected_local_positions


def test_valid_local_positions_adder_single_no_list():
    # occurs with single discrete value
    adder = LocalPositionsAdder([["a"]])
    sequence = [0, 1, 2]
    input_dict = {"a": {"aa": [sequence]}}
    adder(input_dict)
    local_positions = input_dict["a"]["local_positions"]
    expected_local_positions = [[[0], [0], [0]]]
    assert local_positions == expected_local_positions


def test_valid_local_positions_adder_multiple_keys():
    adder = LocalPositionsAdder([["a", "b"]])
    sequence_a = [[0], [1, 2], [3, 4, 5], [6, 7]]
    sequence_b = [[8, 9], [10], [11, 12, 13], [14]]
    input_dict = {"a": {"aa": [sequence_a]}, "b": {"ba": [sequence_b]}}
    adder(input_dict)
    local_positions_a = input_dict["a"]["local_positions"]
    local_positions_b = input_dict["b"]["local_positions"]
    expected_local_positions_a = [[[0], [0, 1], [0, 1, 2], [0, 1]]]
    expected_local_positions_b = [[[1, 2], [2], [3, 4, 5], [2]]]
    assert local_positions_a == expected_local_positions_a
    assert local_positions_b == expected_local_positions_b


def test_valid_local_positions_adder_multiple_groups():
    adder = LocalPositionsAdder([["a"], ["b"]])
    sequence_a = [[0], [1, 2], [3, 4, 5], [6, 7]]
    sequence_b = [[8, 9], [10], [11, 12, 13], [14]]
    input_dict = {"a": {"aa": [sequence_a]}, "b": {"ba": [sequence_b]}}
    adder(input_dict)
    local_positions_a = input_dict["a"]["local_positions"]
    local_positions_b = input_dict["b"]["local_positions"]
    expected_local_positions_a = [[[0], [0, 1], [0, 1, 2], [0, 1]]]
    expected_local_positions_b = [[[0, 1], [0], [0, 1, 2], [0]]]
    assert local_positions_a == expected_local_positions_a
    assert local_positions_b == expected_local_positions_b
