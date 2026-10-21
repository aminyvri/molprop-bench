import numpy as np
import pytest

from molprop.splits import (
    make_split,
    random_split,
    scaffold_split_deterministic,
    scaffold_split_randomized,
)


def fake_scaffolds(n=300, seed=0):
    """A few big scaffold groups plus many small ones, like a real chemical dataset."""
    rng = np.random.default_rng(seed)
    sizes = [40, 25, 15] + [int(s) for s in rng.integers(1, 5, size=120)]
    scaffolds = []
    for k, size in enumerate(sizes):
        scaffolds += [f"S{k}"] * size
    scaffolds = scaffolds[:n] if len(scaffolds) >= n else scaffolds + [f"X{i}" for i in range(n - len(scaffolds))]
    return scaffolds


def check_partition(n, parts):
    joined = np.concatenate(parts)
    assert len(joined) == n
    assert len(set(joined.tolist())) == n


def test_random_split_is_partition_and_reproducible():
    a = random_split(200, seed=3)
    b = random_split(200, seed=3)
    c = random_split(200, seed=4)
    check_partition(200, a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])
    assert (len(a[0]), len(a[1]), len(a[2])) == (160, 20, 20)


@pytest.mark.parametrize("name", ["scaffold", "scaffold_det"])
def test_scaffold_splits_do_not_share_scaffolds(name):
    scaffolds = fake_scaffolds()
    train, valid, test = make_split(name, len(scaffolds), scaffolds, seed=1)
    check_partition(len(scaffolds), (train, valid, test))
    sets = [{scaffolds[i] for i in part} for part in (train, valid, test)]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])


def test_scaffold_split_sizes_are_close_to_requested():
    scaffolds = fake_scaffolds()
    n = len(scaffolds)
    for fn in (lambda: scaffold_split_deterministic(scaffolds), lambda: scaffold_split_randomized(scaffolds, 0)):
        train, valid, test = fn()
        assert 0.7 * n <= len(train) <= 0.85 * n
        assert len(valid) > 0 and len(test) > 0


def test_randomized_scaffold_split_depends_on_seed_but_deterministic_does_not():
    scaffolds = fake_scaffolds()
    a = scaffold_split_randomized(scaffolds, seed=0)[2]
    b = scaffold_split_randomized(scaffolds, seed=1)[2]
    a2 = scaffold_split_randomized(scaffolds, seed=0)[2]
    assert not np.array_equal(a, b)
    assert np.array_equal(a, a2)
    d1 = scaffold_split_deterministic(scaffolds)[2]
    d2 = scaffold_split_deterministic(scaffolds)[2]
    assert np.array_equal(d1, d2)


def test_invalid_fractions_and_names_are_rejected():
    with pytest.raises(ValueError):
        random_split(10, 0, frac=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        make_split("nope", 10, ["a"] * 10, 0)
