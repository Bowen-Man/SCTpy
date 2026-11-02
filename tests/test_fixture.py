import numpy as np

from benchmarks.generate_fixture import generate_counts


def test_fixture_generator_is_deterministic_and_positive_by_cell() -> None:
    first = generate_counts(genes=32, cells=24, seed=7)
    second = generate_counts(genes=32, cells=24, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int64
    assert np.all(first >= 0)
    assert np.all(first.sum(axis=0) > 0)
