#!/usr/bin/env python3
"""
makefigs_results.py -- Figs. 7-13, the results figures.

Every panel is drawn from a computed quantity: waveform records from
out/Data/waveforms.npz, statistics from out/Tables/*.csv.  Nothing is typed.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import spstyle as S

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TBL = os.path.join(ROOT, "out", "Tables")
DAT = os.path.join(ROOT, "out", "Data")
FS = 5000.0
T_FAULT = 0.10


def W():
    z = np.load(os.path.join(DAT, "waveforms.npz"))
    names = sorted({k.split("__")[0] for k in z.files})
    return {n: {k.split("__")[1]: z[k] for k in z.files if k.startswith(n + "__")}
            for n in names}


# typeset names for the feature blocks of (11)-(14), which the tables carry as
# plain ASCII so that the CSVs stay portable
FEAT_TEX = {
    "|I_k|": r"$|I_k|$", "angle I_k": r"$\angle I_k$",
    "|V_k|": r"$|V_k|$", "angle V_k": r"$\angle V_k$",
    "E_k^I": r"$E_k^{I}$", "E_k^V": r"$E_k^{V}$",
    "R_kl": r"$R_{kl}$", "dtheta_kl": r"$\Delta\theta_{kl}$",
    "H2": r"$H_2$", "H3": r"$H_3$",
    "eta_tr": r"$\eta_{\rm tr}$", "eta_CT": r"$\eta_{\rm CT}$",
}


def tidy(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=S.GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


# =====================================================================
# Fig. 7 -- representative records
# =====================================================================
def fig7(w):
    cols = [("internal_bolted_ag", "(i) internal bolted, c0-g at 50 km"),
            ("internal_hif_ag", "(ii) internal high-impedance arc, c0-g at 50 km"),
            ("external_forward", "(iii) external fault, 10 km beyond bus R")]
    GAM_TH = 0.25
    fig, ax = plt.subplots(4, 3, figsize=(S.COL2, 6.35), sharex=True)
    for c, (key, ttl) in enumerate(cols):
        r = w[key]
        t = np.arange(len(r["i"])) / FS * 1e3
        for k in range(6):
            ax[0, c].plot(t, r["i"][:, k] / 1e3, color=S.C6[k], ls=S.LS6[k],
                          lw=0.7, label=S.LBL6[k] if c == 0 else None)
            ax[1, c].plot(t, r["v"][:, k] / 1e3, color=S.C6[k], ls=S.LS6[k], lw=0.7)
        ax[2, c].plot(t, r["Psi"], color=S.NEU, lw=0.9, label=r"$\Psi(t)$")
        ax[2, c].plot(t, r["tau"], color=S.ACC, lw=0.9, ls="--", label=r"$\tau(\rho)$")
        ax[2, c].set_yscale("log")
        ax[2, c].set_ylim(3e-6, max(10, float(np.nanmax(r["Psi"])) * 3))
        g = np.asarray(r["Gam"], float)
        ax[3, c].plot(t, g, color=S.NEU, lw=0.9, label=r"$\Gamma_{\mathrm{HIF}}(t)$")
        ax[3, c].axhline(GAM_TH, color=S.ACC, lw=0.9, ls="--",
                         label=r"$\Gamma_{\mathrm{th}}$")
        gmax = float(np.nanmax(g)) if np.isfinite(g).any() else 1.0
        ax[3, c].set_ylim(0.0, max(0.40, 1.25 * gmax))
        for row in range(4):
            a = ax[row, c]
            a.axvline(T_FAULT * 1e3, color="#888884", lw=0.6, ls=":")
            tidy(a, grid="y")
            a.set_xlim(60, t[-1])
        ax[0, c].set_title(ttl, fontsize=7.5, pad=3)
    ax[0, 0].set_ylabel("conductor current (kA)")
    ax[1, 0].set_ylabel("conductor voltage (kV)")
    ax[2, 0].set_ylabel(r"$\Psi$, $\tau$ (-)")
    ax[3, 0].set_ylabel(r"$\Gamma_{\mathrm{HIF}}$ (-)")
    for c in range(3):
        ax[3, c].set_xlabel("time (ms)")
    # direct labels for the six conductors on the first panel, so that colour is
    # never the only identity cue
    r0 = w[cols[0][0]]
    j = int(0.085 * FS)
    for k in range(6):
        S.endlabel(ax[0, 0], 0.085 * 1e3, r0["i"][j, k] / 1e3, S.LBL6[k], S.C6[k],
                   dx=-3.2, fs=5.6)
    ax[2, 0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)
    ax[3, 0].legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)
    # the inception marker is annotated once, in the pre-fault (empty) region of
    # the lower-left panel, so that it collides with no trace
    ax[3, 1].annotate("fault inception", xy=(T_FAULT * 1e3, 0.97),
                      xycoords=("data", "axes fraction"),
                      xytext=(-4, 0), textcoords="offset points", fontsize=6,
                      color="#666663", ha="right", va="top")
    h, lb = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=6, frameon=False, fontsize=6.6,
               handlelength=2.4, columnspacing=1.5, bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(pad=0.4, w_pad=0.9, h_pad=0.5, rect=(0, 0.028, 1, 1))
    S.save(fig, "fig7_representative_records")


# =====================================================================
# Fig. 8 -- robustness summary
# =====================================================================
def fig8():
    st = pd.read_csv(os.path.join(TBL, "stress_matrix.csv"))
    sw = pd.read_csv(os.path.join(TBL, "sweep_rmax.csv"))
    ibr = pd.read_csv(os.path.join(TBL, "ibr_sweep.csv"))
    fig = plt.figure(figsize=(S.COL2, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.34)

    # (a) D against fault resistance and SNR, as a single-hue heat map
    piv = []
    for f, lab in [("fault resistance (ohm)", r"$R_f$ (Ω)"), ("SNR (dB)", "SNR (dB)"),
                   ("line-parameter error", r"$\theta$ error"),
                   ("load level (pu)", "load (pu)")]:
        s = st[st.factor == f]
        piv.append((lab, s.level.values, s.D.values, s.D_lo.values, s.D_hi.values))
    a = fig.add_subplot(gs[0, 0])
    s = st[st.factor == "fault resistance (ohm)"]
    a.errorbar(np.arange(len(s)), s.D, yerr=[s.D - s.D_lo, s.D_hi - s.D],
               marker="o", ms=3.4, lw=1.0, color=S.C6[0], capsize=2)
    a.set_xticks(np.arange(len(s))); a.set_xticklabels([f"{v:g}" for v in s.level])
    a.set_xlabel(r"fault resistance $R_f$ (Ω)"); a.set_ylabel(r"dependability $D$")
    a.set_ylim(0, 1.02); tidy(a); a.set_title("(a)", loc="left", fontsize=8)

    b = fig.add_subplot(gs[0, 1])
    s = st[st.factor == "SNR (dB)"]
    b.errorbar(s.level, s.D, yerr=[s.D - s.D_lo, s.D_hi - s.D], marker="s", ms=3.4,
               lw=1.0, color=S.C6[2], capsize=2)
    b.set_xlabel("measurement SNR (dB)"); b.set_ylabel(r"$D$")
    b.set_ylim(0, 1.02); tidy(b); b.set_title("(b)", loc="left", fontsize=8)

    c = fig.add_subplot(gs[0, 2])
    s = st[st.factor == "line-parameter error"].sort_values("level")
    c.errorbar(s.level * 100, s.e_d_med, marker="o", ms=3.4, lw=1.0, color=S.C6[1],
               label="median")
    c.plot(s.level * 100, s.e_d_p95, marker="^", ms=3.2, lw=1.0, ls="--",
           color=S.C6[4], label="95th pct")
    c.set_xlabel(r"line-parameter error (%)"); c.set_ylabel(r"$e_d$ (% of $L$)")
    tidy(c); c.legend(frameon=False, fontsize=6.2); c.set_title("(c)", loc="left", fontsize=8)

    # (d) dependability-security trade-off
    d = fig.add_subplot(gs[1, 0])
    m = sw.r_max < 1e8
    d.plot(sw.S[m], sw.D[m], marker="o", ms=3.2, lw=1.0, color=S.NEU, zorder=2)
    # the declared operating point is marked, since it is the one every result
    # in the paper is reported at
    dec = sw[np.isclose(sw.r_max, 0.15)]
    if len(dec):
        d.plot(dec.S, dec.D, marker="o", ms=6.5, mfc="none", mec=S.ACC, mew=1.1,
               ls="none", zorder=3, label="declared $r_{\max}=0.15$")
        d.legend(frameon=False, fontsize=6.0, loc="upper right")
    for _, r in sw[m].iterrows():
        if r.r_max in (0.05, 0.10, 0.15, 0.3, 0.5, 1.5):
            off = (-4, 5) if r.r_max >= 0.5 else (5, 4)
            ha = "right" if r.r_max >= 0.5 else "left"
            d.annotate(f"{r.r_max:g}", (r.S, r.D), textcoords="offset points",
                       xytext=off, fontsize=5.8, color="#666663", ha=ha)
    d.set_xlabel(r"security $S$"); d.set_ylabel(r"dependability $D$")
    d.margins(x=0.14, y=0.12)
    tidy(d); d.set_title(r"(d)  $r_{\max}$ sweep", loc="left", fontsize=8)

    # (e) IBR penetration
    e = fig.add_subplot(gs[1, 1])
    for mode, mk, col in [("grid-following", "o", S.C6[0]), ("grid-forming", "s", S.C6[4])]:
        s = ibr[ibr.control == mode].sort_values("penetration")
        if len(s):
            e.errorbar(s.penetration * 100, s.D, yerr=[s.D - s.D_lo, s.D_hi - s.D],
                       marker=mk, ms=3.4, lw=1.0, color=col, capsize=2, label=mode)
    e.set_xlabel("IBR penetration (% of remote SCC)"); e.set_ylabel(r"$D$")
    e.set_ylim(0, 1.02); tidy(e); e.legend(frameon=False, fontsize=6.2)
    e.set_title("(e)", loc="left", fontsize=8)

    # (f) location error against IBR penetration
    f_ = fig.add_subplot(gs[1, 2])
    for mode, mk, col in [("grid-following", "o", S.C6[0]), ("grid-forming", "s", S.C6[4])]:
        s = ibr[ibr.control == mode].sort_values("penetration")
        if len(s):
            f_.plot(s.penetration * 100, s.e_d_med, marker=mk, ms=3.4, lw=1.0,
                    color=col, label=mode)
    f_.set_xlabel("IBR penetration (%)"); f_.set_ylabel(r"median $e_d$ (% of $L$)")
    tidy(f_); f_.legend(frameon=False, fontsize=6.2)
    f_.set_title("(f)", loc="left", fontsize=8)
    S.save(fig, "fig8_robustness_summary")


# =====================================================================
# Fig. 9 -- detection and trip time
# =====================================================================
def fig9(cases):
    z1 = cases[(cases.kind == "internal") & (cases.d_km <= 85)]
    fam = pd.read_csv(os.path.join(TBL, "dependability_by_family.csv"))
    grp = pd.read_csv(os.path.join(TBL, "dependability_by_group.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2, 2.15))
    tt = z1.m_trip_delay_ms.dropna().values
    ax[0].hist(tt, bins=np.arange(0, 41, 1.0), color=S.C6[0], edgecolor="white", lw=0.3)
    ax[0].axvline(np.median(tt), color=S.ACC, lw=1.0, ls="--")
    ax[0].annotate(f"median {np.median(tt):.1f} ms\n95th pct {np.percentile(tt,95):.1f} ms",
                   xy=(0.97, 0.93), xycoords="axes fraction", ha="right", va="top",
                   fontsize=6.2, color="#444442")
    ax[0].set_xlabel("trip delay after inception (ms)"); ax[0].set_ylabel("cases")
    tidy(ax[0]); ax[0].set_title("(a)", loc="left", fontsize=8)

    lab = [f"{int(r.conductors)}C{'G' if r.grounded else 'U'}" for _, r in fam.iterrows()]
    y = np.arange(len(fam))
    ax[1].barh(y, fam.D, xerr=[fam.D - fam.lo, fam.hi - fam.D], color=S.C6[2],
               height=0.68, error_kw=dict(lw=0.6, capsize=1.6))
    ax[1].set_yticks(y); ax[1].set_yticklabels(lab, fontsize=6)
    ax[1].set_xlabel(r"dependability $D$"); ax[1].set_xlim(0, 1.02)
    tidy(ax[1], grid="x"); ax[1].set_title("(b)  by fault family", loc="left", fontsize=8)

    g = grp.sort_values("group")
    y2 = np.arange(len(g))
    ax[2].barh(y2 - 0.19, g.detection, height=0.36, color=S.C6[0], label="detected")
    ax[2].barh(y2 + 0.19, g.D, height=0.36, color=S.C6[4], label="tripped")
    ax[2].set_yticks(y2)
    ax[2].set_yticklabels([s.replace("_", " ") for s in g.group], fontsize=6)
    ax[2].set_xlim(0, 1.02); ax[2].set_xlabel("fraction of Zone-1 cases")
    tidy(ax[2], grid="x"); ax[2].legend(frameon=False, fontsize=6.2, loc="lower right")
    ax[2].set_title("(c)  by taxonomy group", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    S.save(fig, "fig9_detection_and_trip_time")


# =====================================================================
# Fig. 10 -- classification
# =====================================================================
def fig10():
    cm = pd.read_csv(os.path.join(TBL, "confusion_family.csv"), index_col=0)
    pc = pd.read_csv(os.path.join(TBL, "per_conductor_scores.csv"))
    bm = pd.read_csv(os.path.join(TBL, "classifier_benchmark.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2, 2.5),
                           gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    M = cm.values.astype(float)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    im = ax[0].imshow(Mn, cmap=S.seq_cmap(), vmin=0, vmax=1)
    ax[0].set_xticks(range(len(cm.columns)))
    ax[0].set_xticklabels(cm.columns, rotation=90, fontsize=5.6)
    ax[0].set_yticks(range(len(cm.index)))
    ax[0].set_yticklabels(cm.index, fontsize=5.6)
    ax[0].set_xlabel("predicted family"); ax[0].set_ylabel("true family")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
    cb.set_label("recall", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax[0].set_title("(a)  fault-family confusion", loc="left", fontsize=8)

    y = np.arange(len(pc))
    ax[1].barh(y - 0.19, pc.precision, height=0.36, color=S.C6[0], label="precision")
    ax[1].barh(y + 0.19, pc.recall, height=0.36, color=S.C6[2], label="recall")
    ax[1].set_yticks(y); ax[1].set_yticklabels(pc.conductor, fontsize=6.5)
    ax[1].set_xlim(0, 1.02); ax[1].set_xlabel("score")
    tidy(ax[1], grid="x")
    ax[1].legend(frameon=False, fontsize=6.2, loc="upper center", ncol=2,
                 bbox_to_anchor=(0.5, -0.16))
    # every score is at or within a fraction of a per cent of unity, so the
    # axis is expanded to make the one imperfect conductor visible at all
    lo = float(min(pc.precision.min(), pc.recall.min()))
    ax[1].set_xlim(max(0.0, min(0.99, lo - 0.004)), 1.0005)
    ax[1].set_title("(b)  faulty-conductor identification", loc="left", fontsize=8)

    b = bm[bm.task == "family"]
    y2 = np.arange(len(b))
    ax[2].barh(y2, b.macro_F1, xerr=[b.macro_F1 - b.F1_lo, b.F1_hi - b.macro_F1],
               color=S.C6[1], height=0.6, error_kw=dict(lw=0.6, capsize=1.6))
    ax[2].set_yticks(y2)
    ax[2].set_yticklabels([m.split(" (")[0] for m in b.model], fontsize=6)
    ax[2].set_xlim(0, 1.02); ax[2].set_xlabel(r"macro-$F_1$ (test partition)")
    tidy(ax[2], grid="x")
    ax[2].set_title("(c)  model families", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    S.save(fig, "fig10_classification")


# =====================================================================
# Fig. 11 -- fault location
# =====================================================================
def fig11(cases):
    z1 = cases[(cases.kind == "internal") & (cases.group == "A_shunt") & (cases.d_km <= 85)]
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2, 2.15))
    e = z1.e_d.values; e = e[np.isfinite(e)]
    ax[0].hist(e, bins=np.linspace(0, 40, 41), color=S.C6[0], edgecolor="white", lw=0.3)
    ax[0].axvline(np.median(e), color=S.ACC, lw=1.0, ls="--")
    ax[0].annotate(f"median {np.median(e):.2f}%\n95th pct {np.percentile(e,95):.2f}%\n"
                   f"max {e.max():.1f}%", xy=(0.97, 0.93), xycoords="axes fraction",
                   ha="right", va="top", fontsize=6.2, color="#444442")
    ax[0].set_xlabel(r"$e_d$ (% of line length)"); ax[0].set_ylabel("cases")
    tidy(ax[0]); ax[0].set_title("(a)", loc="left", fontsize=8)

    ds = sorted(z1.d_km.unique())
    data = [z1[z1.d_km == v].e_d.dropna().values for v in ds]
    bp = ax[1].boxplot(data, positions=np.arange(len(ds)), widths=0.62,
                       showfliers=False, patch_artist=True)
    for p in bp["boxes"]:
        p.set(facecolor=S.SEQ[2], edgecolor=S.NEU, lw=0.6)
    for k in ("whiskers", "caps", "medians"):
        for p in bp[k]:
            p.set(color=S.NEU, lw=0.7)
    ax[1].set_xticks(np.arange(len(ds)))
    ax[1].set_xticklabels([f"{v:g}" for v in ds], fontsize=6)
    ax[1].set_xlabel("true fault distance (km)"); ax[1].set_ylabel(r"$e_d$ (% of $L$)")
    tidy(ax[1]); ax[1].set_title("(b)", loc="left", fontsize=8)

    rf = sorted(z1.Rf_ohm.unique())
    data = [z1[z1.Rf_ohm == v].e_d.dropna().values for v in rf]
    bp = ax[2].boxplot(data, positions=np.arange(len(rf)), widths=0.62,
                       showfliers=False, patch_artist=True)
    for p in bp["boxes"]:
        p.set(facecolor=S.SEQ[3], edgecolor=S.NEU, lw=0.6)
    for k in ("whiskers", "caps", "medians"):
        for p in bp[k]:
            p.set(color=S.NEU, lw=0.7)
    ax[2].set_xticks(np.arange(len(rf)))
    ax[2].set_xticklabels([f"{v:g}" for v in rf], fontsize=6)
    ax[2].set_xlabel(r"fault resistance $R_f$ (Ω)"); ax[2].set_ylabel(r"$e_d$ (% of $L$)")
    tidy(ax[2]); ax[2].set_title("(c)", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    S.save(fig, "fig11_location_error")


# =====================================================================
# Fig. 12 -- HIF detection and external-fault security
# =====================================================================
def fig12(cases, H):
    roc = pd.read_csv(os.path.join(TBL, "hif_roc_gamma_only.csv"))
    sec = pd.read_csv(os.path.join(TBL, "security_breakdown.csv"))
    krp = os.path.join(TBL, "kappa_by_remanence.csv")
    kr = pd.read_csv(krp) if os.path.exists(krp) else None
    ncol = 4 if kr is not None else 3
    fig, ax = plt.subplots(1, ncol, figsize=(S.COL2, 2.25))
    r = roc.sort_values("Pfa")
    ax[0].plot(r.Pfa, r.Pd, color=S.C6[0], lw=1.1)
    ax[0].plot([0, 1], [0, 1], color="#b4b4b0", lw=0.6, ls=":")
    ax[0].set_xlabel(r"false-alarm rate on the no-fault set")
    ax[0].set_ylabel(r"HIF detection probability")
    ax[0].annotate(f"AUC = {H['hif_auc_gamma_only']:.3f}", xy=(0.95, 0.08),
                   xycoords="axes fraction", ha="right", fontsize=6.5, color="#444442")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1.02); tidy(ax[0])
    ax[0].set_title(r"(a)  $\Gamma_{\rm HIF}$ alone", loc="left", fontsize=8)

    hif = cases[cases.kind == "hif"]
    nof = cases[cases.kind == "nofault"]
    bolt = cases[(cases.kind == "internal") & (cases.Rf_ohm == 0)]
    bins = np.linspace(0, 1, 41)
    for j, (v, lab_, col, ls) in enumerate([
            (hif.m_kappa, "high-impedance arc", S.C6[0], "-"),
            (bolt.m_kappa, "bolted shunt fault", S.C6[4], (0, (4, 1.2))),
            (nof.m_kappa, "no-fault", S.C6[2], (0, (1, 1)))]):
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        if len(v):
            ax[1].hist(v, bins=bins, histtype="step", lw=1.0, color=col, ls=ls,
                       label=lab_, density=True)
    ax[1].set_xlabel(r"half-cycle asymmetry $\kappa_i$"); ax[1].set_ylabel("density")
    tidy(ax[1]); ax[1].legend(frameon=False, fontsize=5.6, loc="upper center")
    ax[1].set_title("(b)", loc="left", fontsize=8)

    if kr is not None:
        cols = [c for c in kr.columns
                if c != "ct_remanence" and not c.startswith("n_")]
        mk = ["o", "s", "^", "D", "v"]
        nice = {"no-fault": "no fault", "external": "external fault",
                "internal, bolted": "internal, bolted",
                "internal, Rf >= 300 ohm": r"internal, $R_f \geq 300\,\Omega$",
                "high-impedance arc": "high-impedance arc"}
        for j, c in enumerate(cols):
            ax[2].plot(kr.ct_remanence, kr[c], marker=mk[j % len(mk)], ms=3.0,
                       lw=1.0, color=S.C6[j % 6], ls=S.LS6[j % 6],
                       label=nice.get(c, c))
        ax[2].set_xlabel("CT remanence (pu)")
        ax[2].set_ylabel(r"median $\kappa_i$")
        ax[2].set_ylim(-0.03, 1.32)
        tidy(ax[2])
        ax[2].legend(frameon=False, fontsize=5.2, loc="lower right",
                     handlelength=2.2, borderaxespad=0.2)
        ax[2].set_title("(c)", loc="left", fontsize=8)
        k3 = 3
    else:
        k3 = 2

    pretty = {"forward_external": "forward external", "reverse": "reverse",
              "nofault": "no fault", "power_swing": "power swing",
              "load_switch": "load switch", "quiet": "quiet"}
    lab = [(pretty.get(k, k) + (", " + pretty.get(sk, sk)
                                if isinstance(sk, str) and sk else ""))
           for k, sk in zip(sec.kind, sec.subkind)]
    y = np.arange(len(sec))
    ax[k3].barh(y, sec.S, xerr=[sec.S - sec.lo, sec.hi - sec.S], color=S.C6[2],
                height=0.6, error_kw=dict(lw=0.6, capsize=1.6))
    ax[k3].set_yticks(y); ax[k3].set_yticklabels(lab, fontsize=5.6)
    ax[k3].set_xlim(0, 1.02); ax[k3].set_xlabel(r"security $S$")
    tidy(ax[k3], grid="x")
    ax[k3].set_title("(d)" if kr is not None else "(c)", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.1)
    S.save(fig, "fig12_hif_and_security")


# =====================================================================
# Fig. 13 -- measurement-chain attribution, ablation, computational cost
# =====================================================================
def fig13():
    at = pd.read_csv(os.path.join(TBL, "measurement_chain.csv"))
    ab = pd.read_csv(os.path.join(TBL, "ablation.csv"))
    cc = pd.read_csv(os.path.join(TBL, "computational_cost.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(S.COL2, 2.2))
    y = np.arange(len(at))
    ax[0].barh(y, at.D, xerr=[at.D - at.D_lo, at.D_hi - at.D], color=S.C6[0],
               height=0.55, error_kw=dict(lw=0.6, capsize=1.6))
    ax[0].set_yticks(y); ax[0].set_yticklabels(at.chain, fontsize=6)
    ax[0].set_xlim(0, 1.02); ax[0].set_xlabel(r"dependability $D$")
    ax[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    tidy(ax[0], grid="x")
    ax[0].set_title("(a)  measurement-model attribution", loc="left", fontsize=8)

    a = ab[ab.removed != "none (all features)"].sort_values("delta")
    y2 = np.arange(len(a))
    ax[1].barh(y2, a.delta, color=[S.C6[4] if v < 0 else S.C6[2] for v in a.delta],
               height=0.6)
    ax[1].axvline(0, color=S.NEU, lw=0.6)
    ax[1].set_yticks(y2)
    ax[1].set_yticklabels([FEAT_TEX.get(t, t) for t in a.removed], fontsize=5.8)
    ax[1].set_xlabel(r"$\Delta$ macro-$F_1$ when the block is removed")
    tidy(ax[1], grid="x"); ax[1].set_title("(b)  feature ablation", loc="left", fontsize=8)

    y3 = np.arange(len(cc))
    ax[2].barh(y3, cc.us_per_window, color=S.SEQ[4], height=0.6)
    ax[2].set_yticks(y3); ax[2].set_yticklabels(cc.stage, fontsize=5.8)
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"execution time per window (µs)")
    tidy(ax[2], grid="x")
    ax[2].set_title("(c)  measured relay cost", loc="left", fontsize=8)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    S.save(fig, "fig13_attribution_ablation_cost")


def main():
    H = json.load(open(os.path.join(DAT, "headline.json")))
    cases = pd.read_csv(os.path.join(DAT, "main", "cases.csv"))
    cases["e_d"] = np.abs(cases.m_d_hat - cases.d_km)
    for c in ("m_detected", "m_gate"):
        cases[c] = cases[c].astype(float) > 0.5
    w = W()
    fig7(w)
    fig8()
    fig9(cases)
    fig11(cases)
    fig12(cases, H)
    for f in (fig10, fig13):
        try:
            f()
        except FileNotFoundError as e:
            print("  skipped", f.__name__, "-", e)


if __name__ == "__main__":
    main()
