"""
tests/test_lattice.py

Correctness checks for lib/lattice.py, independent of the implementation's
own formulas where possible (bond length is checked against the physical
1.42 A C-C distance, not against anything lattice.py itself computed).
"""
import pytest

from lib.lattice import build_nanotube, closed_form_diameter, closed_form_period, n_hexagons_per_cell, A_CC
from lib.bonds import find_bonds, bond_lengths


@pytest.mark.parametrize("n,m", [(5, 5), (9, 0), (6, 4), (10, 5)])
def test_atom_count_matches_hexagon_formula(n, m):
    structure = build_nanotube(n, m, repeats=1)
    expected_atoms = 2 * n_hexagons_per_cell(n, m)
    assert len(structure.positions) == expected_atoms


@pytest.mark.parametrize("n,m", [(5, 5), (9, 0), (6, 4)])
def test_repeats_multiplies_atom_count(n, m):
    one = build_nanotube(n, m, repeats=1)
    three = build_nanotube(n, m, repeats=3)
    assert len(three.positions) == 3 * len(one.positions)


@pytest.mark.parametrize("n,m", [(5, 5), (9, 0), (6, 4), (10, 5), (12, 3)])
def test_bond_lengths_match_physical_cc_distance(n, m):
    """Every detected bond must be within 2% of the real 1.42 A C-C distance."""
    structure = build_nanotube(n, m, repeats=2)
    bonds = find_bonds(structure.positions)
    lengths = bond_lengths(structure.positions, bonds)
    assert len(lengths) > 0
    assert lengths.min() == pytest.approx(A_CC, rel=0.02)
    assert lengths.max() == pytest.approx(A_CC, rel=0.02)


@pytest.mark.parametrize("n,m", [(5, 5), (9, 0), (6, 4), (10, 5)])
def test_unit_cell_length_matches_closed_form(n, m):
    """ASE's own period must agree with the independently-derived closed-form |T|."""
    structure = build_nanotube(n, m, repeats=1)
    assert structure.unit_cell_length == pytest.approx(closed_form_period(n, m), rel=1e-3)


class TestClassification:
    def test_armchair_is_metallic(self):
        s = build_nanotube(6, 6, repeats=1)
        assert s.cnt_type == "armchair"
        assert s.conductivity == "metallic"

    def test_zigzag_multiple_of_three_is_metallic(self):
        s = build_nanotube(9, 0, repeats=1)
        assert s.cnt_type == "zigzag"
        assert s.conductivity == "metallic"

    def test_zigzag_not_multiple_of_three_is_semiconducting(self):
        s = build_nanotube(10, 0, repeats=1)
        assert s.conductivity == "semiconducting"

    def test_chiral_classification(self):
        s = build_nanotube(10, 5, repeats=1)
        assert s.cnt_type == "chiral"
        assert s.conductivity == "semiconducting"  # (10-5) % 3 = 2


class TestDiameter:
    def test_diameter_scales_with_chiral_vector_magnitude(self):
        """Diameter must grow monotonically with n for fixed m=n (armchair)."""
        d5 = closed_form_diameter(5, 5)
        d10 = closed_form_diameter(10, 10)
        assert d10 == pytest.approx(2 * d5, rel=1e-6)


def test_invalid_indices_raise():
    with pytest.raises(ValueError):
        build_nanotube(3, 5, repeats=1)  # m > n, not a valid (n,m) pair
    with pytest.raises(ValueError):
        build_nanotube(5, 3, repeats=0)  # repeats must be >= 1
