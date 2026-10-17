"""Shared publication style: IEEE column widths, serif text, vector output."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL1, COL2 = 3.5, 7.16                      # IEEE single / double column, inches
OUT = os.environ.get("SP_FIGDIR", os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "Figures")))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Categorical palette for the six conductors, assigned in fixed order and never
# cycled.  Validated with the six-check palette validator (light surface):
#   lightness band PASS, chroma floor PASS, adjacent-pair CVD separation PASS
#   (worst dE 9.5 protan / 15.1 tritan), normal-vision floor PASS (worst 20.2).
# Full six-way CVD separation is not achievable with six hues, so identity is
# additionally carried by a distinct dash pattern per conductor and by a direct
# label at the right-hand edge of every trace -- colour is never the only cue.
C6 = ["#3565b5", "#d4901d", "#2f9d76", "#8a5fbd", "#cc4a44", "#4aa8c9"]
LS6 = ["-", (0, (4, 1.2)), (0, (1, 1)), (0, (5, 1.2, 1, 1.2)),
       (0, (3, 1, 1, 1, 1, 1)), (0, (7, 1.5))]
LBL6 = ["c0", "c1", "c2", "c3", "c4", "c5"]
ACC = "#cc4a44"
NEU = "#4a4a4a"
GRID = "#d8d8d4"
# Sequential ramp: one hue, light -> dark (magnitude).
SEQ = ["#eef3fa", "#cfdff1", "#a7c4e4", "#7aa5d4", "#4f83bf", "#3565b5", "#20406f"]
# Diverging pair: two hues about a neutral grey midpoint (signed quantities).
DIV = ["#8a3a33", "#c1766c", "#e2b7b0", "#eeeeec", "#aec6de", "#5d8bbd", "#274d80"]


def seq_cmap(name="sp_seq"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ)


def div_cmap(name="sp_div"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, DIV)


def endlabel(ax, x, y, txt, color, dx=0.012, fs=6.2):
    ax.annotate(txt, xy=(x, y), xytext=(dx, 0), textcoords="offset fontsize",
                color=color, fontsize=fs, va="center", ha="left",
                annotation_clip=False)


def save(fig, stem, png=True):
    fig.savefig(os.path.join(OUT, stem + ".pdf"))
    if png:
        fig.savefig(os.path.join(OUT, stem + ".png"), dpi=600)
    plt.close(fig)
    print("  wrote", stem)
