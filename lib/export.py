"""
lib/export.py

Molecular file export (XYZ, PDB) so a generated structure can be opened in
external tools (PyMOL, VMD, Avogadro), in addition to the PNG/SVG/PDF image
exports.
"""
import numpy as np


def write_xyz(positions: np.ndarray, comment: str = "") -> str:
    """Minimal XYZ format: atom count, comment line, then `element x y z` per atom."""
    lines = [str(len(positions)), comment]
    for x, y, z in positions:
        lines.append(f"C {x:.4f} {y:.4f} {z:.4f}")
    return "\n".join(lines) + "\n"


def write_pdb(positions: np.ndarray, bonds: list[tuple[int, int]]) -> str:
    """
    Minimal single-model PDB: HETATM records for carbon atoms, plus CONECT
    records built from the bond list so connectivity survives into external
    viewers. PDB atom serials are 1-indexed.
    """
    lines = []
    for i, (x, y, z) in enumerate(positions, start=1):
        lines.append(
            f"HETATM{i:5d}  C   CNT A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )

    neighbors: dict[int, list[int]] = {}
    for a, b in bonds:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    for atom_idx in sorted(neighbors):
        conn = " ".join(f"{j + 1:5d}" for j in sorted(neighbors[atom_idx]))
        lines.append(f"CONECT{atom_idx + 1:5d} {conn}")

    lines.append("END")
    return "\n".join(lines) + "\n"
