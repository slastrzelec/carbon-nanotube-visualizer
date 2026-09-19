"""
lib/bonds.py

Bond detection via nearest-neighbor distance (KDTree), replacing the old
O(N^2) ring-index heuristic. Real 3D Euclidean distance after the chiral-vector
roll-up already accounts for curvature and axial periodicity, so no manual
angular-wraparound bookkeeping is needed here.
"""
from scipy.spatial import cKDTree
import numpy as np

from .lattice import A_CC

DEFAULT_CUTOFF = 1.1 * A_CC  # 10% tolerance above the ideal 1.42 A bond length


def find_bonds(positions: np.ndarray, cutoff: float = DEFAULT_CUTOFF) -> list[tuple[int, int]]:
    """Return every atom-index pair within `cutoff` angstrom of each other."""
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff)
    return sorted(pairs)


def bond_lengths(positions: np.ndarray, bonds: list[tuple[int, int]]) -> np.ndarray:
    if not bonds:
        return np.array([])
    i = np.array([b[0] for b in bonds])
    j = np.array([b[1] for b in bonds])
    return np.linalg.norm(positions[i] - positions[j], axis=1)


def coordination_numbers(n_atoms: int, bonds: list[tuple[int, int]]) -> np.ndarray:
    """Bond count per atom -> distinguishes edge atoms (2 neighbors) from bulk (3)."""
    coord = np.zeros(n_atoms, dtype=int)
    for i, j in bonds:
        coord[i] += 1
        coord[j] += 1
    return coord


def avg_bond_distance_per_atom(positions: np.ndarray, bonds: list[tuple[int, int]]) -> np.ndarray:
    n_atoms = len(positions)
    totals = np.zeros(n_atoms)
    counts = np.zeros(n_atoms)
    for i, j in bonds:
        d = float(np.linalg.norm(positions[i] - positions[j]))
        totals[i] += d
        totals[j] += d
        counts[i] += 1
        counts[j] += 1
    return np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
