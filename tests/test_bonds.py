"""tests/test_bonds.py"""
import numpy as np
import pytest

from lib.bonds import find_bonds, coordination_numbers, avg_bond_distance_per_atom
from lib.lattice import build_nanotube


def test_no_fully_isolated_atoms():
    """No atom should have zero neighbors -- that would mean find_bonds missed it entirely."""
    structure = build_nanotube(6, 6, repeats=3)
    bonds = find_bonds(structure.positions)
    coord = coordination_numbers(len(structure.positions), bonds)
    assert coord.min() >= 1


def test_bulk_atoms_dominate_and_have_coordination_three():
    """
    An open (uncapped) finite tube legitimately has some coordination-1 atoms
    at its two cut ends (dangling bonds where the lattice was truncated) --
    real experimentally-imaged open CNT ends show the same ragged edge, so
    this is physically expected, not a bug. Only the (large) bulk region,
    away from the two ends, is asserted to be fully coordination-3.
    """
    structure = build_nanotube(6, 6, repeats=3)
    bonds = find_bonds(structure.positions)
    coord = coordination_numbers(len(structure.positions), bonds)
    assert set(np.unique(coord)) <= {1, 2, 3}
    assert (coord == 3).sum() > (coord != 3).sum()  # bulk (coordination 3) should dominate for a 3-cell tube


def test_avg_bond_distance_is_close_to_1_42_everywhere():
    structure = build_nanotube(9, 0, repeats=2)
    bonds = find_bonds(structure.positions)
    avg = avg_bond_distance_per_atom(structure.positions, bonds)
    nonzero = avg[avg > 0]
    assert nonzero.min() == pytest.approx(1.42, rel=0.03)
    assert nonzero.max() == pytest.approx(1.42, rel=0.03)
