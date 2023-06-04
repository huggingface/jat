import pytest

from gia.config import Arguments
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


def test_gia_processor_padding_default(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data)

    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_padding_true(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding=True)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == 10 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_padding_longest(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding="longest")
    for sequences in out.values():
        assert len(sequences[0]) == 12  # 2 for the separator tokens
        assert len(sequences[1]) == 12


def test_gia_processor_padding_max_length_no_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding="max_length")
    for sequences in out.values():
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_padding_max_length_with_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding="max_length", max_length=14)
    for sequences in out.values():
        assert len(sequences[0]) == 14
        assert len(sequences[1]) == 14


def test_gia_processor_padding_false(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding=False)
    for sequences in out.values():
        assert len(sequences[0]) == 8 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_padding_do_not_pad(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, padding="do_not_pad")
    for sequences in out.values():
        assert len(sequences[0]) == 8 + 2  # 2 for the separator tokens
        assert len(sequences[1]) == 10 + 2


def test_gia_processor_truncation_default(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncation_residual_no_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation="residual")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncation_residual_with_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation="residual", max_length=10)
    for sequences in out.values():
        assert len(sequences) == 3
        assert len(sequences[0]) == 10
        assert len(sequences[1]) == 10
        assert len(sequences[2]) == 10


def test_gia_processor_truncation_true_no_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation=True)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncation_true_with_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation=True, max_length=9)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == 9
        assert len(sequences[1]) == 9


def test_gia_processor_truncation_max_length_no_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation="max_length")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncation_max_length_with_value(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
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
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation=False)
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncation_do_not_truncate(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation="do_not_truncate")
    for sequences in out.values():
        assert len(sequences) == 2
        assert len(sequences[0]) == args.seq_len
        assert len(sequences[1]) == args.seq_len


def test_gia_processor_truncate_residual_and_pad(data):
    args = Arguments(nb_bins=16, output_dir="./")
    processor = GiaProcessor(args=args)
    out = processor(**data, truncation="residual", padding="max_length", max_length=11)
    for sequences in out.values():
        assert len(sequences) == 3
        assert len(sequences[0]) == 11
        assert len(sequences[1]) == 11
        assert len(sequences[2]) == 11
    assert out["attention_mask"] == [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
