"""
app.py

Interactive Carbon Nanotube Visualizer -- Streamlit UI.

All geometry, bonding, band-structure and export logic lives in lib/ as
plain, Streamlit-independent functions (see SPEC.md); this file is UI only.
"""
from io import BytesIO

import numpy as np
import streamlit as st

from lib.lattice import build_nanotube, closed_form_period
from lib.bonds import find_bonds, coordination_numbers, avg_bond_distance_per_atom
from lib.electronic import compute_band_structure, density_of_states, GAMMA0_DEFAULT
from lib.export import write_xyz, write_pdb
from lib.render_plotly import structure_figure, unrolled_sheet_figure, band_structure_figure, dos_figure, THEMES

st.set_page_config(page_title="Carbon Nanotubes 3D", page_icon="⚛️", layout="wide",
                    initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    h1, h2, h3 { color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .stSelectbox label, .stSlider label, .stRadio label { color: #e0e0e0 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 26px; color: #4fd1c5 !important; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px; border: none; padding: 10px 20px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_structure(n: int, m: int, repeats: int):
    s = build_nanotube(n, m, repeats)
    return s


@st.cache_data(show_spinner=False)
def _cached_bonds(positions: np.ndarray):
    return find_bonds(positions)


@st.cache_data(show_spinner=False)
def _cached_bands(n: int, m: int, gamma0: float):
    return compute_band_structure(n, m, gamma0)


# ============= Header =============
st.markdown("""
<h1 style='font-size: 42px; margin-bottom: 0;'>⚛️ Carbon Nanotube Visualizer</h1>
<p style='font-size: 16px; color: #a0aec0; margin-top: 5px;'>
Interactive 3D structure, real chiral-vector geometry, and a computed tight-binding band structure.
</p>
""", unsafe_allow_html=True)

# ============= Sidebar =============
st.sidebar.markdown("# ⚙️ Control Panel")
st.sidebar.markdown("---")

theme = st.sidebar.radio(
    "Theme", options=list(THEMES.keys()), index=0,
    format_func=lambda x: {"dark": "\U0001f319 Dark", "light": "☀️ Light",
                            "neon": "✨ Neon", "gradient": "\U0001f308 Gradient"}[x],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f52c Nanotube Type")
preset = st.sidebar.selectbox(
    "Structure", options=["armchair", "zigzag", "chiral", "custom"],
    format_func=lambda x: {"armchair": "⬡ Armchair", "zigzag": "⚡ Zigzag",
                            "chiral": "\U0001f300 Chiral (preset)", "custom": "\U0001f4d0 Custom (n,m)"}[x],
)

if preset == "armchair":
    n = st.sidebar.slider("n (= m)", 3, 15, 5)
    m = n
elif preset == "zigzag":
    n = st.sidebar.slider("n", 4, 20, 9)
    m = 0
elif preset == "chiral":
    n, m = 10, 5
    st.sidebar.caption(f"Fixed preset: (n, m) = ({n}, {m}). Switch to Custom for full control.")
else:
    c1, c2 = st.sidebar.columns(2)
    n = c1.number_input("n", min_value=1, max_value=25, value=8, step=1)
    m = c2.number_input("m", min_value=0, max_value=25, value=3, step=1)
    if m > n:
        st.sidebar.error("Require m <= n -- clamping m to n.")
        m = n

repeats = st.sidebar.slider("Unit cell repeats", 1, 5, 2,
                             help="Tube length = repeats x the axial unit-cell period |T|.")

st.sidebar.markdown("---")
st.sidebar.markdown("### \U0001f3ad Rendering")
render_style = st.sidebar.selectbox(
    "Style", options=["ball_and_stick", "wireframe", "space_filling", "stick"],
    format_func=lambda x: {"ball_and_stick": "⚫ Ball and Stick", "wireframe": "\U0001f4d0 Wireframe",
                            "space_filling": "\U0001f535 Space Filling", "stick": "\U0001f4cf Stick"}[x],
)
color_by = st.sidebar.selectbox(
    "Color atoms by", options=["none", "distance", "coordination", "edge_position", "height"],
    format_func=lambda x: {"none": "⚪ Default", "distance": "\U0001f4cf Avg. bond distance",
                            "coordination": "\U0001f517 Coordination number",
                            "edge_position": "\U0001f4d0 Edge position (illustrative)",
                            "height": "\U0001f4ca Height (z)"}[x],
)

with st.sidebar.expander("⚙️ Advanced: tight-binding parameters"):
    gamma0 = st.slider("gamma0 -- hopping integral [eV]", 1.0, 4.0, GAMMA0_DEFAULT, 0.1,
                        help="Nearest-neighbor tight-binding hopping integral. 2.7 eV is the standard literature value.")

# ============= Build structure =============
try:
    structure = _cached_structure(n, m, repeats)
except ValueError as e:
    st.error(str(e))
    st.stop()

bonds = _cached_bonds(structure.positions)

color_values, color_label = None, ""
if color_by == "distance":
    color_values = avg_bond_distance_per_atom(structure.positions, bonds)
    color_label = "avg bond dist [A]"
elif color_by == "coordination":
    color_values = coordination_numbers(len(structure.positions), bonds).astype(float)
    color_label = "coordination"
elif color_by == "edge_position":
    z = structure.positions[:, 2]
    zr = z.max() - z.min()
    color_values = np.abs((z - z.min()) / zr - 0.5) * 2 if zr > 0 else np.zeros_like(z)
    color_label = "edge position (illustrative)"
elif color_by == "height":
    color_values = structure.positions[:, 2]
    color_label = "z [A]"

title = f"({structure.n},{structure.m}) {structure.cnt_type} -- d={structure.diameter:.2f} A -- {structure.conductivity}"

# ============= Main content =============
tab_structure, tab_sheet, tab_bands = st.tabs(["\U0001f9ec 3D Structure", "\U0001f4d0 Unrolled Sheet", "⚡ Band Structure & DOS"])

with tab_structure:
    col1, col2 = st.columns([2.6, 1])

    with col1:
        fig = structure_figure(structure.positions, bonds, render_style, color_by, color_values,
                                color_label, theme, title)
        st.plotly_chart(fig, width="stretch")

        st.markdown("### \U0001f4be Export")
        e1, e2, e3, e4, e5 = st.columns(5)
        fname = f"nanotube_{structure.n}_{structure.m}"

        # Image export (PNG/SVG/PDF) needs a Chrome binary at runtime (kaleido
        # v1+). That is not guaranteed in every deployment environment, so this
        # degrades gracefully -- a missing Chrome disables just these three
        # buttons with a short note, instead of crashing the whole page. XYZ/PDB
        # are pure Python and always available regardless.
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            svg_bytes = fig.to_image(format="svg")
            pdf_bytes = fig.to_image(format="pdf")
            image_export_ok = True
        except Exception:
            image_export_ok = False

        with e1:
            if image_export_ok:
                st.download_button("PNG", png_bytes, f"{fname}.png", "image/png", width="stretch")
            else:
                st.button("PNG", disabled=True, width="stretch")
        with e2:
            if image_export_ok:
                st.download_button("SVG", svg_bytes, f"{fname}.svg", "image/svg+xml", width="stretch")
            else:
                st.button("SVG", disabled=True, width="stretch")
        with e3:
            if image_export_ok:
                st.download_button("PDF", pdf_bytes, f"{fname}.pdf", "application/pdf", width="stretch")
            else:
                st.button("PDF", disabled=True, width="stretch")
        with e4:
            xyz_text = write_xyz(structure.positions, comment=title)
            st.download_button("XYZ", xyz_text, f"{fname}.xyz", "chemical/x-xyz", width="stretch")
        with e5:
            pdb_text = write_pdb(structure.positions, bonds)
            st.download_button("PDB", pdb_text, f"{fname}.pdb", "chemical/x-pdb", width="stretch")

        if not image_export_ok:
            st.caption("PNG/SVG/PDF export needs Chrome in the hosting environment and isn't available "
                       "here right now -- use the interactive view or the XYZ/PDB export instead.")

    with col2:
        st.markdown("### \U0001f4ca Structure")
        st.metric("Atoms", f"{len(structure.positions)}")
        st.metric("Bonds", f"{len(bonds)}")
        st.metric("Diameter", f"{structure.diameter:.2f} A")
        st.metric("Chiral angle", f"{structure.chiral_angle_deg:.1f} deg")
        st.metric("Axial period |T|", f"{structure.unit_cell_length:.2f} A")

        st.markdown("---")
        cond_icon = "⚡" if structure.conductivity == "metallic" else "\U0001f50c"
        st.info(f"**{cond_icon} {structure.conductivity.capitalize()}**\n\n"
                f"Type: **{structure.cnt_type}**\n\n"
                f"(n − m) mod 3 = {(structure.n - structure.m) % 3}")

with tab_sheet:
    st.plotly_chart(unrolled_sheet_figure(structure.n, structure.m, theme), width="stretch")
    st.caption(
        "The chiral vector **C** wraps around the tube circumference; the translation vector **T** "
        "(perpendicular to C) is the tube's axial repeat unit. Rolling the shaded parallelogram into "
        "a cylinder along C produces the 3D structure shown in the first tab."
    )

with tab_bands:
    with st.spinner("Computing tight-binding band structure..."):
        band = _cached_bands(structure.n, structure.m, gamma0)
        dos_centers, dos_counts = density_of_states(band)

    b1, b2 = st.columns(2)
    b1.metric("Subbands", f"{band.n_subbands}")
    # The exact (n - m) mod 3 rule (integer arithmetic, unambiguous) decides the
    # metallic/semiconducting label; the tight-binding calculation supplies the
    # quantitative gap. For a metallic tube the true gap is exactly zero -- the
    # numeric minimum from a finite k-grid is a resolution-limited approximation
    # to that zero, not a competing estimate, so the label is not derived from it.
    if structure.conductivity == "metallic":
        gap_txt = "0 eV (metallic)"
    else:
        gap_txt = f"{band.band_gap_ev:.3f} eV"
    b2.metric("Computed band gap", gap_txt)

    st.caption(
        "Nearest-neighbor tight-binding dispersion, zone-folded onto the tube's allowed k-vectors "
        "(Saito, Dresselhaus & Dresselhaus, *Physical Properties of Carbon Nanotubes*). This is what "
        "actually produces the gap value above -- not just the (n−m) mod 3 rule, which only gives "
        f"the metallic/semiconducting label. (Numerical minimum on this k-grid: {band.band_gap_ev:.4f} eV; "
        "the true value for a metallic tube is exactly zero -- any tiny residual above is k-grid resolution, not a real gap.)"
    )

    col_band, col_dos = st.columns([2, 1])
    with col_band:
        st.plotly_chart(
            band_structure_figure(band.kappa, band.energies, band.energies_lower, theme,
                                   title=f"Band structure ({structure.n},{structure.m}), gamma0={gamma0:.1f} eV"),
            width="stretch",
        )
    with col_dos:
        st.plotly_chart(dos_figure(dos_centers, dos_counts, theme), width="stretch")

st.markdown("---")
st.caption(
    "Carbon nanotube discovered 1991 (S. Iijima). Atomic coordinates via ASE's nanotube builder; "
    "bonds via nearest-neighbor KDTree; band structure via nearest-neighbor tight-binding zone-folding."
)
