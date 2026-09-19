"""
tests/test_electronic.py

Cross-checks the tight-binding band structure against the exact, unambiguous
(n - m) % 3 == 0 metallic/semiconducting rule from lib/lattice.py -- these are
two independently-derived results (integer arithmetic vs. a k-space physics
calculation) that must agree.
"""
import pytest

from lib.lattice import classify
from lib.electronic import compute_band_structure


@pytest.mark.parametrize("n,m", [(5, 5), (6, 6), (9, 0), (12, 0), (11, 2)])
def test_metallic_pairs_have_near_zero_minimum_gap(n, m):
    _, conductivity = classify(n, m)
    assert conductivity == "metallic"
    band = compute_band_structure(n, m, n_k=600)
    # True value is exactly 0; a finite k-grid leaves a small residual.
    assert band.band_gap_ev < 0.05


@pytest.mark.parametrize("n,m", [(10, 0), (6, 4), (10, 5), (8, 3)])
def test_semiconducting_pairs_have_finite_gap(n, m):
    _, conductivity = classify(n, m)
    assert conductivity == "semiconducting"
    band = compute_band_structure(n, m, n_k=600)
    assert band.band_gap_ev > 0.05


def test_subband_count_matches_lattice_hexagon_formula():
    from lib.lattice import n_hexagons_per_cell
    band = compute_band_structure(9, 0, n_k=50)
    assert band.n_subbands == n_hexagons_per_cell(9, 0)


def test_larger_gamma0_scales_the_gap_linearly():
    band_1 = compute_band_structure(10, 0, gamma0=2.7, n_k=600)
    band_2 = compute_band_structure(10, 0, gamma0=5.4, n_k=600)
    assert band_2.band_gap_ev == pytest.approx(2 * band_1.band_gap_ev, rel=0.01)
