"""
lib/render_plotly.py

Interactive Plotly figure builders, replacing the old static matplotlib
renderer. Plotly gives free mouse rotate/zoom/pan on the 3D view and hover
tooltips per atom -- the single highest-impact change on perceived quality
(see SPEC.md, G1).
"""
from math import gcd

import numpy as np
import plotly.graph_objects as go

from .lattice import A1, A2, A_CC, dR

THEMES = {
    "dark": dict(
        bg="#0a0e27", plot_bg="#1a1f3a", text="#e0e6ed", grid="#2d3561",
        bond="#4a90e2", atom_scale="Viridis",
    ),
    "light": dict(
        bg="#f8f9fa", plot_bg="#ffffff", text="#2c3e50", grid="#cbd5e0",
        bond="#4a5568", atom_scale="Viridis",
    ),
    "neon": dict(
        bg="#000000", plot_bg="#0d0221", text="#00ff9f", grid="#ff006e",
        bond="#00ff9f", atom_scale="Plasma",
    ),
    "gradient": dict(
        bg="#1a1a2e", plot_bg="#16213e", text="#f0a500", grid="#0f3460",
        bond="#e94560", atom_scale="Twilight",
    ),
}

_MARKER_SIZE = {
    "ball_and_stick": 5,
    "space_filling": 11,
    "stick": 3,
    "wireframe": 0,  # atoms hidden entirely in wireframe mode
}


def structure_figure(
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    render_style: str = "ball_and_stick",
    color_by: str = "none",
    color_values: np.ndarray | None = None,
    color_label: str = "",
    theme: str = "dark",
    title: str = "",
) -> go.Figure:
    t = THEMES[theme]
    fig = go.Figure()

    # Bonds: one Scatter3d "lines" trace with None separators between
    # segments -- the standard Plotly technique for many disjoint segments
    # in a single trace (keeps the figure fast).
    if render_style != "space_filling":
        xs, ys, zs = [], [], []
        for i, j in bonds:
            xs += [positions[i, 0], positions[j, 0], None]
            ys += [positions[i, 1], positions[j, 1], None]
            zs += [positions[i, 2], positions[j, 2], None]
        linewidth = 6 if render_style == "stick" else 3
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=t["bond"], width=linewidth),
            hoverinfo="skip", showlegend=False,
        ))

    if render_style != "wireframe":
        marker_size = _MARKER_SIZE[render_style]
        n_atoms = len(positions)
        hover = [f"atom {i}<br>({x:.2f}, {y:.2f}, {z:.2f}) A"
                  for i, (x, y, z) in enumerate(positions)]

        marker = dict(size=marker_size, line=dict(width=1, color="white"))
        if color_by != "none" and color_values is not None:
            marker.update(color=color_values, colorscale=t["atom_scale"],
                           showscale=True, colorbar=dict(title=color_label))
            if color_label:
                hover = [f"{h}<br>{color_label}: {v:.3f}" for h, v in zip(hover, color_values)]
        else:
            marker.update(color=positions[:, 2], colorscale=t["atom_scale"], showscale=False)

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode="markers", marker=marker, text=hover, hoverinfo="text",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=t["text"])),
        paper_bgcolor=t["bg"],
        scene=dict(
            xaxis=dict(title="X [A]", backgroundcolor=t["plot_bg"], gridcolor=t["grid"], color=t["text"]),
            yaxis=dict(title="Y [A]", backgroundcolor=t["plot_bg"], gridcolor=t["grid"], color=t["text"]),
            zaxis=dict(title="Z [A]", backgroundcolor=t["plot_bg"], gridcolor=t["grid"], color=t["text"]),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
    )
    return fig


def unrolled_sheet_figure(n: int, m: int, theme: str = "dark") -> go.Figure:
    """
    Flat honeycomb patch with the chiral vector C, translation vector T, and
    the resulting unit-cell parallelogram drawn on it -- the classic
    "roll-up" textbook diagram. Uses the same A1/A2 basis as lib/lattice.py
    and lib/electronic.py, but is not required to atom-for-atom match the
    actual rolled 3D tube -- its job is to explain *why* (n, m) determines
    the structure, not to be re-cut into the tube itself.
    """
    t = THEMES[theme]
    C = n * A1 + m * A2
    d = gcd(n, m) or 1
    dr = dR(n, m)
    t1 = (2 * m + n) / dr
    t2 = -(2 * n + m) / dr
    T = t1 * A1 + t2 * A2

    # Bounding range of integer (i,j) lattice indices comfortably covering C and T.
    span = int(max(abs(n), abs(m), abs(t1), abs(t2))) + 3

    delta = A_CC * np.array([1.0, 0.0])  # A -> B sublattice offset

    xs_bond, ys_bond = [], []
    xs_a, ys_a, xs_b, ys_b = [], [], [], []
    for i in range(-2, span):
        for j in range(-2, span):
            base = i * A1 + j * A2
            a_pt = base
            b_pt = base + delta
            xs_a.append(a_pt[0]); ys_a.append(a_pt[1])
            xs_b.append(b_pt[0]); ys_b.append(b_pt[1])
            # 3 bonds per A atom (to neighboring B atoms) -- drawing all of
            # them from every A atom double-covers the interior but that's
            # fine for a static illustrative diagram.
            for other in (base + delta, base + delta - A1, base + delta - A2):
                xs_bond += [a_pt[0], other[0], None]
                ys_bond += [a_pt[1], other[1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_bond, y=ys_bond, mode="lines",
                              line=dict(color=t["grid"], width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs_a + xs_b, y=ys_a + ys_b, mode="markers",
                              marker=dict(size=4, color=t["text"]), hoverinfo="skip", showlegend=False))

    # Chiral vector C and translation vector T as arrows from the origin.
    for vec, name, color in [(C, "C (chiral)", t["bond"]), (T, "T (axial)", "#e94560")]:
        fig.add_annotation(x=vec[0], y=vec[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=3, arrowcolor=color,
                            text="", standoff=0)
        fig.add_trace(go.Scatter(x=[vec[0]], y=[vec[1]], mode="text", text=[name],
                                  textfont=dict(color=color, size=13), showlegend=False, hoverinfo="skip"))

    # Unit-cell parallelogram: 0, C, C+T, T.
    poly = np.array([[0, 0], C, C + T, T, [0, 0]])
    fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines",
                              line=dict(color=t["text"], width=2, dash="dash"),
                              fill="toself", fillcolor="rgba(255,255,255,0.05)",
                              showlegend=False, hoverinfo="skip"))

    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["plot_bg"],
        xaxis=dict(title="x [A]", color=t["text"], gridcolor=t["grid"], scaleanchor="y"),
        yaxis=dict(title="y [A]", color=t["text"], gridcolor=t["grid"]),
        margin=dict(l=10, r=10, t=30, b=10), height=550,
        title=dict(text=f"Unrolled graphene sheet -- chiral vector ({n},{m})", font=dict(color=t["text"])),
    )
    return fig


def band_structure_figure(kappa: np.ndarray, energies_upper: np.ndarray, energies_lower: np.ndarray,
                            theme: str = "dark", title: str = "") -> go.Figure:
    t = THEMES[theme]
    fig = go.Figure()
    for row in energies_upper:
        fig.add_trace(go.Scatter(x=kappa, y=row, mode="lines",
                                  line=dict(color=t["bond"], width=1.4), showlegend=False, hoverinfo="skip"))
    for row in energies_lower:
        fig.add_trace(go.Scatter(x=kappa, y=row, mode="lines",
                                  line=dict(color="#e94560", width=1.4), showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=t["text"], width=1, dash="dot"))
    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["plot_bg"],
        xaxis=dict(title="k (fraction of 1D Brillouin zone)", color=t["text"], gridcolor=t["grid"]),
        yaxis=dict(title="E [eV]", color=t["text"], gridcolor=t["grid"]),
        margin=dict(l=10, r=10, t=30, b=10), height=450,
        title=dict(text=title, font=dict(color=t["text"])),
    )
    return fig


def dos_figure(centers: np.ndarray, counts: np.ndarray, theme: str = "dark") -> go.Figure:
    t = THEMES[theme]
    fig = go.Figure(go.Scatter(x=counts, y=centers, mode="lines", fill="tozerox",
                                line=dict(color=t["bond"])))
    fig.add_hline(y=0, line=dict(color=t["text"], width=1, dash="dot"))
    fig.update_layout(
        paper_bgcolor=t["bg"], plot_bgcolor=t["plot_bg"],
        xaxis=dict(title="DOS (states, arb. units)", color=t["text"], gridcolor=t["grid"]),
        yaxis=dict(title="E [eV]", color=t["text"], gridcolor=t["grid"]),
        margin=dict(l=10, r=10, t=30, b=10), height=450,
        title=dict(text="Density of states", font=dict(color=t["text"])),
    )
    return fig
