# Carbon Nanotube Visualizer — Technical Upgrade Spec

Status: IMPLEMENTED (see README.md for the user-facing summary; this document is kept as the technical rationale + physics references).
Scope: technical/functional upgrade only. Naming, README, LICENSE, repo rename, typos = explicitly deferred to a later phase (agreed).

## 1. Purpose

Turn `03_CNT_vis` from a stylised, matplotlib-based "tube generator" into a scientifically
correct, genuinely interactive carbon nanotube modeling tool that can withstand technical
scrutiny from someone in the field (recruiter with a science background, or a technical
interviewer who opens the code).

Two problems are being fixed at once:
1. **Perceived quality** — static image, slider-only "3D" feels dated next to the rest of
   the portfolio (Streamlit apps with real interactivity: erytro, cuneiform classifier).
2. **Actual correctness** — the current geometry engine does not implement real chiral-vector
   lattice theory for 3 of its 4 modes, while the portfolio page's copy claims it does. This
   is a credibility risk, not just a polish gap.

## 2. Goals, in priority order

| # | Goal | Why it's ranked here |
|---|------|----------------------|
| G1 | Replace static matplotlib rendering with interactive Plotly 3D | Single highest-impact change on perceived quality — free mouse rotate/zoom/pan + hover |
| G2 | Replace the geometry engine with a real chiral-vector roll-up construction, unified across all nanotube types | Fixes the correctness/credibility gap; removes the two-parallel-code-paths inconsistency |
| G3 | Replace index-based bond detection with a distance-based nearest-neighbor method (KDTree) | Correctness + performance, falls out naturally from G2 |
| G4 | Add a tight-binding band structure / DOS panel | The single biggest "domain expert" differentiator — computes something real instead of printing a rule |
| G5 | Add an unrolled graphene-sheet diagram (chiral vector, unit cell) | Classic textbook visual; cheap to build once G2 exists; strong pedagogical/credibility value |
| G6 | Add PDB/XYZ molecular file export | Small effort, real interoperability with PyMOL/VMD/Avogadro — a detail a chemistry-literate reviewer notices |
| G7 | Relabel or fix the "charge" coloring property | Currently a fake function of z-position presented as a physical property |
| G8 | Cache geometry generation (`st.cache_data`) | Performance/UX polish, becomes cheap once G2/G3 land |

## 3. Non-goals (this phase)

- Repo rename, README rewrite, LICENSE, pinned `requirements.txt`, typo fixes — deferred,
  handled in the follow-up "polish" pass, not blocked by this spec.
- Tests/CI — recommended as a fast-follow once the new geometry engine is stable (a unit
  test suite is much more valuable against the *new*, correct engine than the current one),
  not in scope for this spec unless you want it folded in now.
- Multi-walled nanotubes (MWCNTs) — out of scope; single-walled only, as today.
- Any change to hosting/deployment (stays on Streamlit Community Cloud).

## 4. Data & security

No user-submitted data, no file uploads, no authentication, no external network calls, no
model training. All inputs are UI-controlled integers/floats (n, m, ring count, etc.) that
directly parameterize a closed-form geometric/physics calculation. There is no dataset and
therefore no data-leakage surface of the kind that caused the earlier recruitment-process
rejection — flagging this explicitly rather than skipping the section. The only external
I/O this spec adds is local, on-demand file export (PNG/SVG/PDF, now also PDB/XYZ) generated
from in-memory arrays and streamed via `st.download_button`; nothing is written to disk or
transmitted anywhere.

## 5. Architecture change

Current: one file (`app.py`, 671 lines) mixing geometry, rendering, and Streamlit UI, with
two parallel geometry code paths (heuristic ring-stacking for armchair/zigzag/chiral vs. a
partially-correct chiral-vector calculation for "vector" mode).

Target structure (splits concerns, and — not incidentally — becomes testable):

```
03_CNT_vis/
  app.py                 # Streamlit UI only: widgets, layout, calls into lib/
  lib/
    lattice.py           # G2: chiral-vector roll-up geometry (pure functions, no Streamlit)
    bonds.py             # G3: KDTree-based nearest-neighbor bond detection
    electronic.py        # G4: tight-binding band structure / DOS
    export.py            # G6: PDB/XYZ writers
    render_plotly.py     # G1: Plotly figure builders (3D structure, unrolled sheet, band plot)
  requirements.txt       # adds: plotly, scipy (for cKDTree)
```

This split is what makes G2/G3/G4 unit-testable against known reference values (see §7),
independent of the Streamlit process.

## 6. Detailed specs

### 6.1 (G2) Lattice generation — `lib/lattice.py`

Replace `create_carbon_nanotube()` entirely. One function, one correct algorithm, used by
**all four** UI presets (Armchair, Zigzag, Chiral, Custom) — the current split into a
"heuristic" path and a "vector" path is removed. The four presets become convenience
wrappers that just pick `(n, m)`:

- Armchair → `n == m`
- Zigzag → `m == 0`
- Chiral (preset) → a sensible default asymmetric pair, e.g. `(n, m) = (10, 5)`
- Custom (n, m) → direct user control (today's "vector" mode)

**Algorithm** (standard chiral-vector roll-up, reference: Saito/Dresselhaus/Dresselhaus,
*Physical Properties of Carbon Nanotubes*, and cross-checked against ASE's nanotube
generator for validation — see §7):

1. Constants: `a_cc = 1.42` (Å, C–C bond length), `a = sqrt(3) * a_cc` (graphene lattice
   constant).
2. Graphene lattice vectors `a1 = a*(sqrt(3)/2, 1/2)`, `a2 = a*(sqrt(3)/2, -1/2)`, two-atom
   basis (A/B sublattices).
3. Chiral vector `C = n*a1 + m*a2`; circumference `|C| = a*sqrt(n^2 + n*m + m^2)`;
   tube radius `r = |C| / (2*pi)`.
4. Translation vector `T = t1*a1 + t2*a2`, with `d = gcd(n, m)`,
   `dR = d if (n - m) % (3*d) != 0 else 3*d`, `t1 = (2*m + n) / dR`, `t2 = -(2*n + m) / dR`.
   `|T|` is the length of one axial period.
5. Generate the flat honeycomb sheet over a bounding rectangle large enough to contain the
   `[0, |C|) x [0, |T|)` unit cell (plus margin), rotate so `C` aligns with the sheet's local
   x-axis, then keep only atoms falling inside that rectangle — this is the tube's one
   periodic unit cell, `N = 2*(n^2 + n*m + m^2) / dR` hexagons / `2N` atoms.
6. Roll up: for each kept 2D point `(x, y)`, map to
   `(X, Y, Z) = (r*cos(2*pi*x/|C|), r*sin(2*pi*x/|C|), y)`.
7. Tube length control: expose **axial repeats of the unit cell** (an integer) as the UI
   parameter, not a free "number of rings" slider as today — replicate the unit cell along
   `Z` by `k * |T|` for `k = 0 .. repeats-1`. This is the one **user-facing control change**
   this spec introduces (flagged as an open question in §9 — the alternative is a
   continuous "length in Å" slider that's internally rounded to whole unit cells).

Returns: `atoms: np.ndarray[N,3]`, plus metadata (`diameter_angstrom`, `chiral_angle_deg`,
`n`, `m`, `cnt_type: Literal["armchair","zigzag","chiral"]`,
`conductivity: Literal["metallic","semiconducting"]` from `(n - m) % 3 == 0`).

### 6.2 (G3) Bond detection — `lib/bonds.py`

Replace the nested-loop index matching in `create_carbon_nanotube()`. Given the `atoms`
array from §6.1:

- Build `scipy.spatial.cKDTree(atoms)`.
- Query pairs within `cutoff = 1.1 * a_cc` (10% tolerance above the ideal 1.42 Å bond
  length, to absorb the roll-up's small geometric distortion) via `tree.query_pairs(cutoff)`.
- This is correct **because** real 3D Euclidean distance after roll-up already accounts for
  curvature and axial periodicity — no manual angular-wraparound bookkeeping needed, unlike
  the current ring-index approach.
- Complexity: O(N log N) build + query, vs. today's O(N²) — matters once repeats/diameter
  push atom counts up.

Returns: `bonds: list[tuple[int, int]]`.

### 6.3 (G1) Interactive rendering — `lib/render_plotly.py`

Replace `visualize_nanotube()`'s matplotlib figure with a Plotly figure:

- Atoms: `go.Scatter3d(mode="markers", ...)`. `marker.size` driven by the existing
  render-style options (ball-and-stick / space-filling / stick — space-filling uses a
  visibly larger marker size, consistent with today's UX). `marker.color` driven by the
  existing "color by" options (distance / coordination / height — see §6.6 for "charge").
  Hover text per atom: index, coordinates, coordination number.
- Bonds: one `go.Scatter3d(mode="lines", ...)` trace built from the bond list (with `None`
  separators between segments, the standard Plotly technique for disjoint line segments in
  one trace — keeps trace count low for performance).
- Wireframe style = bonds trace only, atoms trace hidden. This maps directly onto today's
  `render_style` options, so the sidebar controls barely change.
- Camera: Plotly's built-in orbit/zoom/pan replaces the elevation/azimuth sliders entirely.
  The sliders are removed from the sidebar (replaced by nothing — Plotly's default mouse
  interaction is the intended replacement, this is a deliberate UX simplification, not a
  missing feature).
- Themes (dark/light/neon/gradient): keep the existing four, ported to Plotly's
  `layout.template` / `paper_bgcolor` / `scene.xaxis.backgroundcolor` etc. — same visual
  intent, different API.
- Export: PNG/SVG/PDF via `fig.write_image()` (requires `kaleido` — new dependency) for
  server-side rendering into `BytesIO`, feeding the existing `st.download_button` pattern
  unchanged.

### 6.4 (G4) Band structure / DOS — `lib/electronic.py`

Nearest-neighbor tight-binding (zone-folding of graphene's π-band dispersion), the standard
simplified model for CNT electronic structure (reference: same Saito/Dresselhaus/Dresselhaus
text as §6.1 — this keeps the whole physics layer traceable to one citable source, which
matters if this is ever asked about in an interview).

- Input: `(n, m)`, hopping integral `gamma0 = 2.7` (eV, literature default — exposed as an
  advanced/optional slider, not a primary control), C–C bond length `a_cc = 1.42`.
- Zone-folding: `N = 2*(n^2 + n*m + m^2) / dR` one-dimensional subbands (cutting lines
  across the graphene Brillouin zone, indexed `ν = 0 .. N-1`), each evaluated over its
  allowed `k` range along the tube axis using graphene's tight-binding dispersion.
- Output: `E(k)` for every subband (for the 3D band plot) and a derived density of states
  `g(E)` via histogram/kernel-density binning of all subband energies (for the DOS panel).
- Classification cross-check: verify computed `E(k)` crosses zero at the zone center for
  `(n - m) % 3 == 0` (metallic) and shows a finite gap otherwise (semiconducting) — this
  becomes a correctness test (§7), and is also the "wow" payoff: the UI's printed
  metallic/semiconducting label is now backed by an actual computed gap, not just the
  arithmetic rule.
- UI: a second tab or an expandable panel next to the 3D view — `st.plotly_chart` line plot
  for `E(k)` per subband, plus a small DOS sub-plot. Not blocking the main 3D view; this is
  an additive panel, default-collapsed if it adds meaningful load time.

### 6.5 (G5) Unrolled graphene sheet diagram

A second, 2D Plotly figure (or matplotlib is fine here — it's static and cheap, no need to
force everything through Plotly): plots the honeycomb sheet computed in step 5 of §6.1,
draws the chiral vector `C` as an arrow from the origin, annotates `(n, m)`, the chiral
angle, and the unit cell rectangle (`C` x `T`) that gets rolled up. Placed as a companion
panel (e.g. a `st.tabs(["3D structure", "Unrolled sheet", "Band structure"])` layout) —
this is also where G4's plot lives, keeping the main view uncluttered.

### 6.6 (G7) Property coloring — fix or relabel

- `distance` (avg bond length) and `coordination` (bond count) are real, computed
  quantities — keep as-is, now computed from the §6.2 bond list.
- `charge`: currently `abs(normalized_z - 0.5) * 2`, a made-up function of axial position
  with no physical basis. Two options, your call (§9):
  (a) Remove it as a coloring option entirely, or
  (b) Relabel it clearly as "Edge position (illustrative)" and keep the same formula,
      honest about what it actually shows (distance from the tube's axial center, a proxy
      for "how close to the open/capped end"), not a simulated charge.
- `height` (z-position) — already honestly labeled, keep as-is.

### 6.7 (G6) Molecular file export — `lib/export.py`

- `write_xyz(atoms, symbols=["C"]*len(atoms)) -> str` — minimal XYZ format (atom count,
  comment line, then `element x y z` per atom), trivial to implement, universally readable.
- `write_pdb(atoms) -> str` — minimal single-model PDB with `HETATM` records for carbon
  atoms and `CONECT` records from the §6.2 bond list, so bond info survives into external
  viewers.
- Both wired into the existing export button row (`st.download_button`), alongside
  PNG/SVG/PDF.

### 6.8 (G8) Caching

- `create_lattice(n, m, repeats)` (§6.1) and `find_bonds(atoms, cutoff)` (§6.2) wrapped in
  `st.cache_data`, keyed on their scalar arguments — pure functions, ideal cache candidates.
  Changing only theme/color-by/render-style should hit cache and skip geometry
  recomputation entirely.

## 7. Testing plan (recommended, confirm scope in §9)

Once `lib/` exists as plain functions independent of Streamlit, this becomes directly
testable — same pattern as the erytro cleanup (`tests/`, synthetic cases with known
ground truth):

- **Geometry** (`lib/lattice.py`): for a small, hand-checkable case (e.g. (5,5) armchair,
  one unit cell) assert atom count, and cross-check `diameter_angstrom` against the closed-
  form formula independently computed in the test (not reusing the implementation's own
  formula — that would only test self-consistency, not correctness).
- **Conductivity rule**: for a grid of `(n, m)` pairs, assert `(n - m) % 3 == 0` matches
  the label the code returns, and — if G4 is implemented — that the computed band gap is
  ≈0 for metallic pairs and >0 for semiconducting ones.
- **Bonds** (`lib/bonds.py`): on the same small case, assert every bond distance is within
  a tight tolerance of 1.42 Å, and that the bond count matches the expected value for a
  known small nanotube (cross-checkable by hand or against a reference tool for one case).
- **Export** (`lib/export.py`): round-trip — write XYZ/PDB, re-parse atom count, assert it
  matches input.
- CI: same GitHub Actions pattern as erytro (`pytest` on push/PR to `main`).

## 8. Rollout

Single-branch, incremental commits in this order (each independently working, so the app
is never broken mid-refactor):
1. `lib/lattice.py` + `lib/bonds.py`, with the old matplotlib UI still calling into them
   (proves correctness before touching rendering).
2. `lib/render_plotly.py`, swapping the rendering layer only.
3. `lib/export.py` (XYZ/PDB).
4. `lib/electronic.py` + band-structure UI tab.
5. Unrolled-sheet UI tab.
6. `st.cache_data` pass.

Deploys to the existing `nanotubes.streamlit.app` (Streamlit Community Cloud, auto-deploys
from `main`) — no infra changes.

## 9. Open questions — need your call before implementation starts

1. **Tube-length control (§6.1 step 7):** replace today's free "number of rings" slider
   with integer "unit cell repeats", or keep a continuous length slider that rounds
   internally? Repeats is physically cleaner; a length-in-Å slider may feel more intuitive
   to a non-specialist visitor. Your call.
2. **Charge coloring (§6.6):** remove it, or relabel as illustrative edge-position?
3. **Testing scope (§7):** include now as part of this same effort, or as a fast-follow
   once the new engine is confirmed working in the deployed app?
4. **γ0 (hopping integral) slider (§6.4):** expose as an advanced control, or hardcode the
   literature default (2.7 eV) and leave it out of the UI entirely?
5. Confirm the rollout order in §8 is fine to execute as one continuous session, or you'd
   rather review/approve after step 1 (geometry+bonds) before I touch rendering.
