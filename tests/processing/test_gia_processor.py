import pytest

from gia.processing import GiaProcessor


# Fixture for data
@pytest.fixture
def data():
    return {
        "continuous_observations": [
            [[0.1, 0.2], [0.3, 0.4]],
            None,
        ],
        "discrete_observations": [
            None,
            [[1, 2, 3], [4, 5, 6]],
        ],
        "continuous_actions": [
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2]],
        ],
        "rewards": [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
    }


def test_process_continuous_observations():
    data = {
        "continuous_observations": [[[0.1, 0.2], [0.3, 0.4]]],
    }
    processor = GiaProcessor()
    out = processor(**data)
    assert isinstance(out, dict)
    assert out.keys() == {"input_ids", "input_types", "local_positions", "loss_mask", "attention_mask"}
    # Check input_ids
    # (6 = len([val00, val01, separator, val10, val11, separator]))
    assert len(out["input_ids"]) == 1
    assert len(out["input_ids"][0]) == 1024  # check the output is padded
    assert all(val > 30000 for val in out["input_ids"][0][:6])  # check the value is shifted
    assert all(val is None for val in out["loss_mask"][0][6:])  # check the value is padded with None
    # Check input_types
    assert len(out["input_types"]) == 1
    assert len(out["input_types"][0]) == 1024  # check the output is padded
    assert all(val == 0 for val in out["input_types"][0][:6])  # all values are tokens
    assert all(val is None for val in out["input_types"][0][6:])  # check the value is padded with None
    # Check local_positions
    assert len(out["local_positions"]) == 1
    assert len(out["local_positions"][0]) == 1024  # check the output is padded
    assert out["local_positions"][0][0:2] == [0, 1]  # first timestep
    assert out["local_positions"][0][2] is None  # separator
    assert out["local_positions"][0][3:5] == [0, 1]  # second timestep
    assert out["local_positions"][0][5] is None  # separator
    assert all(val is None for val in out["local_positions"][0][6:])  # check the value is padded with None


def test_gia_processor_padding_default(data):
    processor = GiaProcessor()
    out = processor(**data)

    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_padding_true(data):
    processor = GiaProcessor()
    out = processor(**data, padding=True)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == 10 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_padding_longest(data):
    processor = GiaProcessor()
    out = processor(**data, padding="longest")
    for sequences in out.values():
        assert len(sequences[0]) == 12  # 2 for the separator tokens
        assert len(sequences[1]) == 12


def test_gia_processor_padding_max_length_no_value(data):
    processor = GiaProcessor()
    out = processor(**data, padding="max_length")
    for sequences in out.values():
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_padding_max_length_with_value(data):
    processor = GiaProcessor()
    out = processor(**data, padding="max_length", max_length=14)
    for sequences in out.values():
        assert len(sequences[0]) == 14
        assert len(sequences[1]) == 14


def test_gia_processor_padding_false(data):
    processor = GiaProcessor()
    out = processor(**data, padding=False)
    for sequences in out.values():
        assert len(sequences[0]) == 8 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_padding_do_not_pad(data):
    processor = GiaProcessor()
    out = processor(**data, padding="do_not_pad")
    for sequences in out.values():
        assert len(sequences[0]) == 8 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_truncation_default(data):
    processor = GiaProcessor()
    out = processor(**data)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncation_residual_no_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="residual")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncation_residual_with_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="residual", max_length=10)
    for sequences in out.values():
        assert len(sequences) == 3
        assert len(sequences[0]) == 10
        assert len(sequences[1]) == 10
        assert len(sequences[2]) == 10


def test_gia_processor_truncation_true_no_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation=True)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncation_true_with_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation=True, max_length=9)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == 9
        assert len(sequences[1]) == 9


def test_gia_processor_truncation_max_length_no_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="max_length")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncation_max_length_with_value(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="max_length", max_length=11)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == 11
        assert len(sequences[1]) == 11
    assert out["attention_mask"] == [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]


def test_gia_processor_truncation_false(data):
    processor = GiaProcessor()
    out = processor(**data, truncation=False)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncation_do_not_truncate(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="do_not_truncate")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == processor.seq_len
        assert len(sequences[1]) == processor.seq_len


def test_gia_processor_truncate_residual_and_pad(data):
    processor = GiaProcessor()
    out = processor(**data, truncation="residual", padding="max_length", max_length=11)
    for sequences in out.values():
        assert len(sequences) == 3
        assert len(sequences[0]) == 11
        assert len(sequences[1]) == 11
        assert len(sequences[2]) == 11
    assert out["attention_mask"] == [
        [True, True, True, True, True, True, True, True, True, True, False],
        [True, True, True, True, True, True, True, True, True, True, True],
        [True, False, False, False, False, False, False, False, False, False, False],
    ]
