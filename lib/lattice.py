"""
lib/lattice.py

Carbon nanotube lattice geometry.

Atomic coordinates come from ASE's (Atomic Simulation Environment) peer-reviewed
nanotube builder (ase.build.nanotube), which correctly implements the chiral-vector
roll-up construction described in Saito, Dresselhaus & Dresselhaus, "Physical
Properties of Carbon Nanotubes" (1998). This module wraps it and adds the derived
metadata (diameter, chiral angle, conductivity) the rest of the app needs, plus an
independent closed-form diameter/period check so a broken dependency version can't
silently produce wrong geometry without anything noticing (see tests/).
"""
from dataclasses import dataclass
from math import sqrt, pi, atan, degrees, gcd

import numpy as np
from ase.build import nanotube as _ase_nanotube

A_CC = 1.42                 # C-C bond length, angstrom (graphite/CNT literature value)
A = sqrt(3) * A_CC          # graphene lattice constant, angstrom (~2.46 A)

# Graphene primitive lattice vectors (2D), shared with lib/electronic.py so the
# band-structure module builds its chiral/translation vectors from the exact
# same basis as the geometry module.
A1 = A * np.array([sqrt(3) / 2, 1 / 2])
A2 = A * np.array([sqrt(3) / 2, -1 / 2])


@dataclass(frozen=True)
class NanotubeStructure:
    n: int
    m: int
    positions: np.ndarray        # (N, 3) atom coordinates, angstrom, centered on origin
    unit_cell_length: float      # axial period |T|, angstrom
    repeats: int                 # number of unit cells stacked along the axis
    diameter: float              # angstrom, closed-form chiral-vector formula
    chiral_angle_deg: float
    cnt_type: str                 # "armchair" | "zigzag" | "chiral"
    conductivity: str             # "metallic" | "semiconducting"


def dR(n: int, m: int) -> int:
    """d_R from Saito/Dresselhaus/Dresselhaus: gcd(n,m), or 3*gcd(n,m) when
    (n-m) is divisible by 3*gcd(n,m)."""
    d = gcd(n, m)
    return d if (n - m) % (3 * d) != 0 else 3 * d


def n_hexagons_per_cell(n: int, m: int) -> int:
    """Number of hexagons in the 1D unit cell -> 2x this many atoms."""
    return 2 * (n * n + n * m + m * m) // dR(n, m)


def closed_form_diameter(n: int, m: int) -> float:
    """d_t = |C| / pi, with |C| = a * sqrt(n^2 + nm + m^2)."""
    C_len = A * sqrt(n * n + n * m + m * m)
    return C_len / pi


def closed_form_period(n: int, m: int) -> float:
    """|T|, used only as an independent cross-check against ASE's own cell length."""
    d = gcd(n, m)
    dr = dR(n, m)
    t1 = (2 * m + n) / dr
    t2 = -(2 * n + m) / dr
    T = t1 * A1 + t2 * A2
    return float(np.linalg.norm(T))


def chiral_angle_deg(n: int, m: int) -> float:
    if n == m:
        return 30.0
    if m == 0:
        return 0.0
    return degrees(atan(sqrt(3) * m / (2 * n + m)))


def classify(n: int, m: int) -> tuple[str, str]:
    if n == m:
        cnt_type = "armchair"
    elif m == 0:
        cnt_type = "zigzag"
    else:
        cnt_type = "chiral"
    conductivity = "metallic" if (n - m) % 3 == 0 else "semiconducting"
    return cnt_type, conductivity


def build_nanotube(n: int, m: int, repeats: int = 1) -> NanotubeStructure:
    """
    Build a single-walled (n, m) carbon nanotube, `repeats` unit cells long.

    Raises ValueError for a degenerate (n, m) pair, per the standard convention
    n >= 1, 0 <= m <= n.
    """
    if n < 1 or m < 0 or m > n:
        raise ValueError("Require n >= 1 and 0 <= m <= n (standard (n,m) convention).")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    ase_atoms = _ase_nanotube(n, m, length=repeats, bond=A_CC, symbol="C")
    positions = ase_atoms.get_positions()
    cell_length = float(np.linalg.norm(ase_atoms.get_cell()[2])) / repeats

    # Center on the tube's own centroid so rendering/export code never needs to
    # know ASE's internal placement convention.
    positions = positions - positions.mean(axis=0)

    cnt_type, conductivity = classify(n, m)

    return NanotubeStructure(
        n=n,
        m=m,
        positions=positions,
        unit_cell_length=cell_length,
        repeats=repeats,
        diameter=closed_form_diameter(n, m),
        chiral_angle_deg=chiral_angle_deg(n, m),
        cnt_type=cnt_type,
        conductivity=conductivity,
    )
