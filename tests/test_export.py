"""tests/test_export.py"""
from lib.export import write_xyz, write_pdb
from lib.lattice import build_nanotube
from lib.bonds import find_bonds


def test_xyz_roundtrip_atom_count():
    structure = build_nanotube(5, 5, repeats=1)
    xyz = write_xyz(structure.positions, comment="test")
    lines = xyz.strip().splitlines()
    assert int(lines[0]) == len(structure.positions)
    assert len(lines) == len(structure.positions) + 2  # count + comment + one line per atom
    for line in lines[2:]:
        element, x, y, z = line.split()
        assert element == "C"
        float(x), float(y), float(z)  # must parse as numbers


def test_pdb_contains_hetatm_and_conect_records():
    structure = build_nanotube(5, 5, repeats=1)
    bonds = find_bonds(structure.positions)
    pdb = write_pdb(structure.positions, bonds)
    assert pdb.count("HETATM") == len(structure.positions)
    assert "CONECT" in pdb
    assert pdb.strip().endswith("END")
