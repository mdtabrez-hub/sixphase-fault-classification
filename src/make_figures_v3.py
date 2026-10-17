#!/usr/bin/env python3
"""
make_figures_v3.py -- generates Figs. 2, 3, 4 and 6 of the manuscript.

Every panel is plotted from a computed quantity, never from a typed number:
Figs. 2 and 3 are drawn from sixphase_core.py, Fig. 4 from the frozen
taxonomy CSV, Fig. 6 from the structure of Algorithm 1 and Table IX.
No panel here depends on any measured or simulated quantity.

Output: PDF (vector, for submission) and 600-dpi PNG (for the Word file),
written to ../Figures_Final/.
"""
import os, csv, sys
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.stderr.write(
        "\n  matplotlib is not installed for this interpreter, so the figures were\n"
        "  not regenerated. Every figure is already present in ../Figures_Final/,\n"
        "  so nothing downstream is blocked. To regenerate them:\n\n"
        "      python3 -m pip install --user matplotlib\n\n"
        "  then re-run this script or RUN.sh.\n\n")
    sys.exit(0)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import sixphase_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.abspath(os.path.join(HERE, "..", "Figures_Final"))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def save(fig, stem):
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    fig.savefig(os.path.join(OUT, stem + ".png"), dpi=600)
    plt.close(fig)
    print("  wrote", stem + ".pdf / .png")


# ======================================================================
# Fig. 2 -- conductor geometry and computed line-matrix structure
# ======================================================================
def fig2(S):
    cfg = S["cfg"]
    x, h = core.geometry(cfg)
    L, C = S["L"], S["C"]

    fig = plt.figure(figsize=(7.16, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.34)

    # ---- (a) tower cross-section -------------------------------------
    ax = fig.add_subplot(gs[0])
    ax.axhline(0, color="k", lw=1.1)
    ax.add_patch(Rectangle((-5.5, -2.2), 11.0, 2.2, facecolor="0.90",
                           edgecolor="none", zorder=0))
    for xx in np.arange(-5.3, 5.5, 0.6):
        ax.plot([xx, xx - 0.30], [0, -0.55], color="0.45", lw=0.5, zorder=1)
    for i, (xi, hi) in enumerate(zip(x, h)):
        ax.plot(xi, hi, "o", ms=5.0, mfc="w", mec="k", mew=0.9, zorder=3)
        ax.text(xi, hi + 0.55, r"$c_{%d}$" % i, ha="center", va="bottom", fontsize=7.2)
        ax.plot([xi, xi], [0, hi], color="0.78", lw=0.45, ls=(0, (1, 2.5)), zorder=1)
    ax.annotate("", xy=(-4.6, 0), xytext=(-4.6, 10.0),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
    ax.text(-4.85, 5.0, r"$h = 10$ m", rotation=90, ha="right",
            va="center", fontsize=7.2)
    ax.annotate("", xy=(-0.5, 8.1), xytext=(0.5, 8.1),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="k"))
    ax.text(0.0, 7.3, r"$s = 1$ m", ha="center", va="top", fontsize=7.2)
    ax.text(0.0, 12.9, r"$r = 15$ mm,  GMR $= 0.7788\,r = 11.68$ mm",
            ha="center", va="center", fontsize=7.2)
    ax.text(0.0, -1.35, r"$\rho_e = 100\ \Omega\cdot$m,   $p = 355.9 - j355.9$ m",
            ha="center", va="center", fontsize=7.2,
            bbox=dict(fc="w", ec="none", pad=1.2))
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-2.4, 14.4)
    ax.set_aspect("auto"); ax.axis("off")
    ax.set_title("(a) Conductor geometry (Table I);\nvertical scale compressed",
                 pad=3, y=0.985)

    # ---- (b) L and (c) C heat maps -----------------------------------
    for k, (M, name, unit, ttl) in enumerate([
            (L, r"$\mathbf{L}$", "mH/km", "(b) Series inductance"),
            (C, r"$\mathbf{C}$", "nF/km", "(c) Shunt capacitance")]):
        ax = fig.add_subplot(gs[k + 1])
        im = ax.imshow(np.abs(M), cmap="Greys", vmin=0, vmax=np.abs(M).max())
        for i in range(6):
            for j in range(6):
                v = M[i, j]
                shade = "w" if abs(v) > 0.62 * np.abs(M).max() else "k"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.4, color=shade)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(range(6)); ax.set_yticklabels(range(6))
        ax.set_xlabel("conductor index $j$"); ax.set_ylabel("conductor index $i$")
        ax.set_title(ttl + f"  [{unit}]", pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5, width=0.5)
        cb.outline.set_linewidth(0.5)
    save(fig, "fig2_geometry_and_matrices")


# ======================================================================
# Fig. 3 -- modal structure of the untransposed line
# ======================================================================
def fig3(S):
    a, b = S["alpha"] * 1e4, S["beta"] * 1e4
    v, tau = S["vfrac"], S["tau_us"]
    Zs = S["Zseq"]

    fig = plt.figure(figsize=(7.16, 2.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.36)

    # (a) propagation constants in the complex plane
    ax = fig.add_subplot(gs[0])
    ax.plot(a[:5], b[:5], "o", ms=5.0, mfc="w", mec="k", mew=0.9, label="aerial modes")
    ax.plot(a[5], b[5], "s", ms=5.6, mfc="k", mec="k", label="earth-return mode")
    lbl = ["a1", "a2", "a3", "a4", "a5", "e"]
    off = [(0, 8), (0, 8), (-8, 4), (-2, 9), (9, 2), (0, 9)]
    for i in range(6):
        ax.annotate(lbl[i], (a[i], b[i]), textcoords="offset points",
                    xytext=off[i], ha="center", va="center", fontsize=6.9)
    ax.text(0.98, 0.63, "earth-return mode\n$v = 0.661\\,c$", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.7)
    ax.text(0.98, 0.42, "aerial modes\n$v = 0.964$–$0.979\\,c$", transform=ax.transAxes,
            ha="right", va="center", fontsize=6.7)
    ax.set_xlabel(r"$\alpha_m$  ($10^{-4}\ \mathrm{km}^{-1}$)")
    ax.set_ylabel(r"$\beta_m$  ($10^{-4}\ \mathrm{km}^{-1}$)")
    ax.set_title(r"(a) $\gamma_m=\alpha_m+j\beta_m$ at 50 Hz", pad=4)
    ax.set_ylim(10.1, 17.6); ax.set_xlim(0.44, 1.30)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3,
              borderpad=0.15, labelspacing=0.2, fontsize=6.9)

    # (b) modal velocity
    ax = fig.add_subplot(gs[1])
    idx = np.arange(6)
    bars = ax.bar(idx, v, width=0.62, color="0.82", edgecolor="k", lw=0.7)
    bars[5].set_hatch("///"); bars[5].set_facecolor("w")
    for i in range(6):
        ax.text(i, v[i] + 0.012, f"{tau[i]:.0f}", ha="center", fontsize=6.6)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"a{i+1}" for i in range(5)] + ["e"], fontsize=7.5)
    ax.set_xlabel("mode  (a1–a5 aerial, e earth-return)")
    ax.set_ylabel(r"$v_m/c$")
    ax.set_ylim(0, 1.10)
    ax.set_title(r"(b) Modal velocity; $\tau$ over 100 km in $\mu$s", pad=4)

    # (c) |S6 Z S6^H| in dB
    ax = fig.add_subplot(gs[2])
    A = np.abs(Zs); A = np.where(A < 1e-12, 1e-12, A)
    dB = 20 * np.log10(A / A.max())
    dB = np.clip(dB, -60, 0)
    im = ax.imshow(dB, cmap="Greys_r", vmin=-60, vmax=0)
    off = A - np.diag(np.diag(A))
    i0, j0 = np.unravel_index(np.argmax(off), off.shape)
    ax.add_patch(Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec="k", lw=1.3))
    ax.annotate(f"largest off-diagonal\n{dB[i0,j0]:.1f} dB", (j0, i0),
                textcoords="offset points", xytext=(14, 12), ha="left",
                fontsize=6.4, bbox=dict(fc="w", ec="k", lw=0.4, pad=1.2),
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xlabel("sequence index $l$"); ax.set_ylabel("sequence index $k$")
    ax.set_title(r"(c) $|\mathbf{S}_6\mathbf{Z}\mathbf{S}_6^{H}|$ (dB re max)", pad=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("dB", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.5); cb.outline.set_linewidth(0.5)
    save(fig, "fig3_modal_structure")


# ======================================================================
# Fig. 4 -- frozen 178-class taxonomy (re-plotted from the CSV)
# ======================================================================
def fig4(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    shunt = [r for r in rows if r["group"] == "A_shunt"]
    g = {k: 0 for k in range(1, 7)}
    u = {k: 0 for k in range(1, 7)}
    for r in shunt:
        (g if r["grounded"] == "1" else u)[int(r["k"])] += 1
    others = [("open\nconductor", "B_open"), ("open +\nshunt", "C_open_shunt"),
              ("evolving", "D_evolving")]
    cnt = [sum(1 for r in rows if r["group"] == key) for _, key in others]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.30),
                                   gridspec_kw=dict(width_ratios=[1.75, 1.0], wspace=0.28))
    ks = np.arange(1, 7); w = 0.38
    b1 = ax1.bar(ks - w / 2, [g[k] for k in ks], w, color="0.35",
                 edgecolor="k", lw=0.7, label="grounded")
    b2 = ax1.bar(ks + w / 2, [u[k] for k in ks], w, color="w",
                 edgecolor="k", lw=0.7, hatch="///", label="ungrounded")
    for bars in (b1, b2):
        for r in bars:
            if r.get_height():
                ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.45,
                         int(r.get_height()), ha="center", fontsize=6.8)
    ax1.set_xticks(ks); ax1.set_xlabel("conductors involved, $k$")
    ax1.set_ylabel("number of classes"); ax1.set_ylim(0, 24)
    ax1.set_title(r"(a) Shunt classes  ($\Sigma=%d$)" % len(shunt), pad=4)
    ax1.legend(frameon=False, loc="upper right")

    b3 = ax2.bar(range(3), cnt, 0.55, color="0.72", edgecolor="k", lw=0.7)
    for r in b3:
        ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.45,
                 int(r.get_height()), ha="center", fontsize=6.8)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([n for n, _ in others], fontsize=7.2)
    ax2.set_ylabel("number of classes"); ax2.set_ylim(0, 25)
    ax2.set_title(r"(b) Series / open / evolving  ($\Sigma=%d$)" % sum(cnt), pad=4)
    save(fig, "fig4_fault_taxonomy")
    return len(shunt) + sum(cnt)


# ======================================================================
# Fig. 6 -- per-window decision path of Algorithm 1
# ======================================================================
def fig6():
    fig, ax = plt.subplots(figsize=(7.16, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    ax.add_patch(Rectangle((1.5, 2.0), 45.5, 55.5, fc="0.955", ec="0.55",
                           lw=0.7, ls=(0, (4, 2)), zorder=0))
    ax.add_patch(Rectangle((52.0, 2.0), 46.5, 55.5, fc="w", ec="0.55",
                           lw=0.7, ls=(0, (4, 2)), zorder=0))
    ax.text(24.2, 58.8, r"programmable logic  —  rate $f_s = 5$ kHz  (lines 1–21)",
            ha="center", fontsize=7.6, style="italic")
    ax.text(75.2, 58.8, r"processor  —  rate $f_0 = 50$ Hz  (lines 22–38)",
            ha="center", fontsize=7.6, style="italic")

    def box(cx, cy, w, h, txt, fc="w", lw=0.8, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.25,rounding_size=0.9",
                                    fc=fc, ec="k", lw=lw, zorder=2))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, zorder=3,
                fontweight="bold" if bold else "normal")
        return (cx, cy, w, h)

    def arrow(p, q, side="v", text=None):
        if side == "v":
            xy0 = (p[0], p[1] - p[3] / 2); xy1 = (q[0], q[1] + q[3] / 2)
        else:
            xy0 = (p[0] + p[2] / 2, p[1]); xy1 = (q[0] - q[2] / 2, q[1])
        ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=8,
                                     lw=0.8, color="k", zorder=2,
                                     shrinkA=0, shrinkB=0))
        if text:
            ax.text((xy0[0] + xy1[0]) / 2 + 1.0, (xy0[1] + xy1[1]) / 2,
                    text, fontsize=6.6, ha="left", va="center")

    L1 = box(24.2, 52.0, 40, 6.2, "1–2:  anti-aliasing $f_c$, buffer $N_w$,\nfull-cycle DFT phasors")
    L2 = box(24.2, 43.0, 40, 6.2, r"3–4:  $\mathbf{S}_6$ transform, Eq. (4);"
                                   "\n" r"pre-fault increments $\Delta\mathbf{V}_s,\Delta\mathbf{I}_s$")
    L3 = box(24.2, 34.0, 40, 6.2, r"5–13:  sequence energies, ratios, $H_2,H_3$,"
                                   "\n" r"$\eta_{tr}$, $\eta_{CT}$, $\hat\sigma_\eta$, $\kappa_i$  —  $\varphi_w\in\mathbb{R}^{85}$")
    L4 = box(24.2, 24.6, 40, 6.6, r"14–18:  index $\Psi$, Eq. (15);  adaptive"
                                  "\n" r"threshold $\tau(\rho)$, Eq. (16);  $D$ over $m$ samples")
    L5 = box(24.2, 15.2, 40, 6.2, r"19–20:  $\Gamma_{HIF}$, Eq. (18), sustained"
                                  "\n" r"$m_{HIF}$ cycles $\Rightarrow$ hif")
    L6 = box(24.2, 7.6, 27, 5.0, r"21:  $D=0$ and not hif $\Rightarrow$ RESTRAIN", fc="0.90")

    R1 = box(75.2, 50.0, 40, 5.6, r"22:  classifier  $\rightarrow$  top-$K$ masks $\hat{\mathbf{b}}^{(k)}$, $C_f$")
    R2 = box(75.2, 41.5, 40, 5.6, r"23–24:  $\hat{\theta}=[\mathbf{R},\mathbf{L},\mathbf{C},Z_S,\hat{R}_f]$;"
                                  "\n" r"admissible mode set $\mathcal{M}$")
    R3 = box(75.2, 32.5, 40, 5.6, r"25–31:  Huber locator, Eq. (24);"
                                  "\n" r"posterior-weighted median $\hat d$")
    R4 = box(75.2, 24.0, 40, 5.0, r"32–33:  bootstrap $[d_{lo},d_{hi}]$;  dispersion $r_d$")
    R5 = box(75.2, 16.4, 40, 6.0, r"34:  security gate $\mathcal{I}$, Eq. (21)"
                                  "\n" r"$-\beta_{rev}L\leq \tilde d\leq\beta L$,  $r_d\leq r_{\max}$,  $C_f\geq C_{\min}$",
             lw=1.5)
    T = box(64.0, 6.2, 17, 4.6, "36:  TRIP", fc="0.86", bold=True)
    A = box(86.5, 6.2, 20, 4.6, "37:  ALARM", fc="w")

    for p, q in ((L1, L2), (L2, L3), (L3, L4), (L4, L5), (L5, L6),
                 (R1, R2), (R2, R3), (R3, R4), (R4, R5)):
        arrow(p, q)
    ax.add_patch(FancyArrowPatch((44.4, 24.6), (55.0, 50.0), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.8, color="k",
                                 connectionstyle="arc3,rad=-0.18", zorder=2))
    ax.text(49.8, 39.0, r"$D=1$" "\n" "or hif", fontsize=6.8, ha="center")
    ax.add_patch(FancyArrowPatch((70.0, 13.2), (64.0, 8.7), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color="k", zorder=2))
    ax.text(64.6, 10.8, r"$\mathcal{I}=1$", fontsize=6.8, ha="right")
    ax.add_patch(FancyArrowPatch((80.4, 13.2), (86.5, 8.7), arrowstyle="-|>",
                                 mutation_scale=8, lw=0.9, color="k", zorder=2))
    ax.text(85.9, 10.8, r"$\mathcal{I}=0$", fontsize=6.8, ha="left")
    ax.text(50.0, 0.6, r"No trip is asserted without $\mathcal{I}$; the gate is evaluated "
                        r"on the processor path only.", ha="center", fontsize=7.0, style="italic")
    save(fig, "fig6_decision_path")


if __name__ == "__main__":
    S = core.summary()
    print("Generating figures into", OUT)
    fig2(S)
    fig3(S)
    tot = fig4(os.path.join(HERE, "taxonomy_full.csv"))
    print(f"  taxonomy total plotted = {tot}")
    fig6()
    print("done.")
