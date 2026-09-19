"""
lib/electronic.py

Nearest-neighbor tight-binding band structure of a single-walled carbon
nanotube, via zone-folding of graphene's pi-band dispersion.

Method (kept deliberately low-level/first-principles rather than quoting a
compact textbook formula from memory, after an earlier attempt at the
compact cutting-line formula failed a numeric cross-check against the known
metallic/semiconducting rule for armchair tubes):

1. Graphene's exact nearest-neighbor tight-binding dispersion,
   E(kx,ky) = +/- gamma0 * |f(kx,ky)|,  f(k) = sum_i exp(i k . delta_i),
   summed over the 3 real nearest-neighbor bond vectors delta_i (each of
   length A_CC, 120 degrees apart). This is the fundamental result and does
   not depend on any tube-specific convention.
2. The tube's allowed k-vectors are quantized along the circumferential
   direction and continuous along the tube axis: k = nu*K1 + kappa*K2, for
   subband index nu = 0..N-1 and kappa in [-0.5, 0.5). K1, K2 are the
   reciprocal vectors of the real-space chiral vector C and translation
   vector T (K1.C = 2*pi, K1.T = 0, K2.C = 0, K2.T = 2*pi), solved
   numerically from C and T rather than quoted from memory.

Reference for the physical model: Saito, Dresselhaus & Dresselhaus,
"Physical Properties of Carbon Nanotubes" (1998).

Validated (see tests/test_electronic.py) against the standard
(n - m) % 3 == 0 <=> metallic rule for a spread of (n, m) pairs, including
the armchair case (whose band crossing sits away from the zone center, which
is exactly what an earlier, memorized compact formula got wrong).
"""
from dataclasses import dataclass
from math import gcd, pi

import numpy as np

from .lattice import A1, A2, A_CC, n_hexagons_per_cell

GAMMA0_DEFAULT = 2.7  # eV, literature nearest-neighbor hopping integral

# The 3 real-space nearest-neighbor bond vectors (A atom -> its 3 B neighbors).
_DELTAS = np.array([
    [A_CC, 0.0],
    [-0.5 * A_CC, (3 ** 0.5) / 2 * A_CC],
    [-0.5 * A_CC, -(3 ** 0.5) / 2 * A_CC],
])


@dataclass(frozen=True)
class BandStructure:
    n: int
    m: int
    gamma0: float
    kappa: np.ndarray          # (nk,) fractional axial momentum, -0.5..0.5
    energies: np.ndarray       # (n_subbands, nk) eV, one row per subband (upper branch)
    energies_lower: np.ndarray  # (n_subbands, nk) eV, lower (valence) branch
    n_subbands: int
    band_gap_ev: float         # 0.0 for metallic tubes


def _graphene_dispersion(kx: np.ndarray, ky: np.ndarray, gamma0: float) -> np.ndarray:
    f = np.zeros(np.shape(kx), dtype=complex)
    for dx, dy in _DELTAS:
        f = f + np.exp(1j * (kx * dx + ky * dy))
    return gamma0 * np.abs(f)


def _reciprocal_vectors(n: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    d = gcd(n, m)
    dr = d if (n - m) % (3 * d) != 0 else 3 * d
    t1 = (2 * m + n) / dr
    t2 = -(2 * n + m) / dr

    C = n * A1 + m * A2
    T = t1 * A1 + t2 * A2

    M = np.array([C, T])  # rows
    K1 = np.linalg.solve(M, np.array([2 * pi, 0.0]))
    K2 = np.linalg.solve(M, np.array([0.0, 2 * pi]))
    return K1, K2


def compute_band_structure(n: int, m: int, gamma0: float = GAMMA0_DEFAULT, n_k: int = 400) -> BandStructure:
    """Compute the tight-binding subbands of an (n, m) nanotube."""
    K1, K2 = _reciprocal_vectors(n, m)
    n_sub = n_hexagons_per_cell(n, m)

    kappa = np.linspace(-0.5, 0.5, n_k)
    upper = np.zeros((n_sub, n_k))
    lower = np.zeros((n_sub, n_k))

    for nu in range(n_sub):
        kx = nu * K1[0] + kappa * K2[0]
        ky = nu * K1[1] + kappa * K2[1]
        E = _graphene_dispersion(kx, ky, gamma0)
        upper[nu] = E
        lower[nu] = -E

    band_gap = float(2 * upper.min())  # smallest |E| over all subbands/k, doubled (valence<->conduction)
    if band_gap < 1e-3:
        band_gap = 0.0

    return BandStructure(
        n=n, m=m, gamma0=gamma0, kappa=kappa,
        energies=upper, energies_lower=lower,
        n_subbands=n_sub, band_gap_ev=band_gap,
    )


def density_of_states(band: BandStructure, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Histogram-based DOS from all subband/k energy samples (both branches)."""
    all_E = np.concatenate([band.energies.ravel(), band.energies_lower.ravel()])
    counts, edges = np.histogram(all_E, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts.astype(float)
