# Carbon Nanotube Visualizer

Interactive 3D modeling of single-walled carbon nanotubes: real chiral-vector
lattice geometry, a computed tight-binding band structure, and molecular file
export -- built with Python, ASE, Plotly, and Streamlit.

Live demo: https://nanotubes.streamlit.app/

## Overview

Pick a nanotube by its chiral indices (n, m) -- or an Armchair/Zigzag/Chiral
preset -- and explore it interactively:

- **3D structure**: real chiral-vector roll-up geometry (via ASE's nanotube
  builder), rendered in Plotly with full mouse rotate/zoom/pan and per-atom
  hover, in four render styles (ball-and-stick, wireframe, space-filling, stick).
- **Unrolled graphene sheet**: the classic textbook diagram -- the flat
  honeycomb lattice with the chiral vector **C** and translation vector **T**
  drawn on it, showing exactly what gets rolled into the tube.
- **Band structure & DOS**: a nearest-neighbor tight-binding calculation,
  zone-folded onto the tube's allowed k-vectors, that actually *computes* the
  metallic/semiconducting gap rather than only printing the (n-m) mod 3 rule.

## Why this is a rebuild, not the original app

An earlier version of this tool used matplotlib for a static, slider-rotated
image, and only its "custom (n,m)" mode computed real chiral-vector geometry
-- the Armchair/Zigzag/Chiral presets used a simplified ring-stacking
heuristic that didn't actually implement the physics the UI described. This
version fixes that: **one** geometry engine (ASE's peer-reviewed nanotube
builder), used by all four presets, with bonds detected by real nearest-
neighbor distance (KDTree) instead of ring-index guessing, plus a genuine
electronic-structure calculation. See `SPEC.md` for the full technical
rationale and the physics references used.

## Features

- Armchair / Zigzag / Chiral (preset) / Custom (n, m) nanotube selection
- 4 render styles, 5 atom-coloring modes, 4 visual themes
- Adjustable tube length (unit-cell repeats)
- Advanced panel: tunable tight-binding hopping integral (gamma0)
- Export: PNG / SVG / PDF (needs Chrome in the hosting environment) and
  XYZ / PDB (always available -- open the structure in PyMOL, VMD, Avogadro, etc.)

## Installation

```bash
git clone https://github.com/slastrzelec/carbon-nanotube-visualizer.git
cd carbon-nanotube-visualizer
pip install -r requirements.txt
streamlit run app.py
```

## Testing

Core geometry, bonding, and band-structure logic (`lib/`) is covered by unit
tests (`tests/`) that check it against physical ground truth -- e.g. every
detected bond must be within 2% of the real 1.42 A C-C distance, and the
computed band gap must agree with the exact (n-m) mod 3 metallic rule.
Tests run automatically via GitHub Actions on every push and pull request.

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Tech stack

Python, Streamlit, ASE, Plotly, NumPy, SciPy

## Physics references

- N. Saito, G. Dresselhaus, M. S. Dresselhaus, *Physical Properties of
  Carbon Nanotubes* (Imperial College Press, 1998) -- chiral-vector
  construction and the tight-binding zone-folding method used here.
- S. Iijima, "Helical microtubules of graphitic carbon," *Nature* 354,
  56-58 (1991) -- original discovery.

## Known limitation

Tubes are open-ended (uncapped): a small number of atoms at the two cut
edges have only 1 detected neighbor (a dangling bond), which is physically
normal for an open SWCNT end, not a rendering bug. Hydrogen-terminated or
capped ends are a possible future addition.
