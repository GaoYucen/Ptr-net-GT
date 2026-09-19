import itertools
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from evaluate_fragment_reconnection import (  # noqa: E402
    exact_fixed_connection, exact_reversible_connection, split_fragments,
)


def brute(coords, fragments, reversible):
    best = float('inf')
    orientation_sets = itertools.product((0, 1), repeat=len(fragments)) if reversible else [(0,) * len(fragments)]
    for orientations in orientation_sets:
        paths = [path if orientation == 0 else list(reversed(path))
                 for path, orientation in zip(fragments, orientations)]
        for order in itertools.permutations(range(1, len(paths))):
            arranged = [paths[0]] + [paths[index] for index in order]
            value = sum(float(torch.dist(coords[a[-1]], coords[b[0]]))
                        for a, b in zip(arranged, arranged[1:] + arranged[:1]))
            best = min(best, value)
    return best


def test_exact_component_dps_match_enumeration():
    coords = torch.rand(10, 2, generator=torch.Generator().manual_seed(91))
    fragments = [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]]
    assert exact_fixed_connection(coords, fragments) == pytest.approx(
        brute(coords, fragments, False), abs=1e-9)
    assert exact_reversible_connection(coords, fragments) == pytest.approx(
        brute(coords, fragments, True), abs=1e-9)


@pytest.mark.parametrize('strategy', ['random', 'longest'])
def test_fragmentation_is_a_nontrivial_partition(strategy):
    coords = torch.rand(20, 2, generator=torch.Generator().manual_seed(92))
    tour = list(range(20))
    fragments = split_fragments(tour, 8, torch.Generator().manual_seed(93),
                                strategy, coords)
    assert len(fragments) == 8
    assert all(len(path) >= 2 for path in fragments)
    assert sorted(node for path in fragments for node in path) == tour
