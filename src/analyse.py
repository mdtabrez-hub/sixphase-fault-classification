#!/usr/bin/env python3
"""
analyse.py -- every statistic reported in Sections 4-6, computed from the
frozen case set produced by runsim.py.

Outputs (out/Tables/*.csv and out/Data/headline.json) are the sole source of
every number that enters the manuscript; nothing is typed by hand.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "out", "Data", "main")
TBL = os.path.join(ROOT, "out", "Tables")
os.makedirs(TBL, exist_ok=True)

L_LINE = 100.0
BETA = 0.85
SEED = 20260829


# ----------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def cluster_ci(flag, cluster, B=1000, seed=SEED):
    """95 % interval on a proportion by CLUSTER bootstrap.

    The Wilson interval below assumes independent Bernoulli trials.  The case
    set is not that: it is 178 taxonomy classes x N operating points, with the
    fault position fixed inside the open-conductor, open-plus-shunt and
    evolving groups, so cases inside a class are not independent draws.
    Section 3.5 of the manuscript makes exactly this argument about partitioning
    and it applies with equal force to interval estimation, so the cluster
    interval is reported ALONGSIDE the Wilson interval everywhere a headline
    proportion is quoted.  Clusters are resampled with replacement; the
    within-cluster sample is kept whole.

    Returns (lo, hi, n_clusters).
    """
    flag = np.asarray(flag, bool)
    cl = np.asarray(cluster)
    uc, inv = np.unique(cl, return_inverse=True)
    idx = [np.nonzero(inv == j)[0] for j in range(len(uc))]
    if len(uc) < 3:
        return (np.nan, np.nan, len(uc))
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uc), len(uc))
        sel = np.concatenate([idx[j] for j in pick])
        out[b] = flag[sel].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), int(len(uc))


def wrow(name, k, n, extra=None):
    p, lo, hi = wilson(k, n)
    r = dict(metric=name, n=int(n), successes=int(k), estimate=p,
             ci95_lo=lo, ci95_hi=hi)
    if extra:
        r.update(extra)
    return r


def boot_ci(x, stat=np.median, B=2000, seed=SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load():
    d = pd.read_csv(os.path.join(DATA, "cases.csv"))
    for c in d.columns:
        if c.endswith(("detected", "gate")):
            d[c] = d[c].astype(float) > 0.5
    d["subkind"] = d.get("subkind", pd.Series([""] * len(d))).fillna("")
    d["e_d"] = np.abs(d.m_d_hat - d.d_km) / L_LINE * 100.0
    d["e_d_or"] = np.abs(d.o_d_hat - d.d_km) / L_LINE * 100.0
    d["e_d_sat"] = np.abs(d.s_d_hat - d.d_km) / L_LINE * 100.0
    d["zone1"] = (d.kind == "internal") & (d.d_km <= BETA * L_LINE)
    d["is_ext"] = d.kind.isin(["forward_external", "reverse", "nofault"])
    return d


def trip(d, pre="m_", gam_th=None, r_max=None, beta=None):
    """TRIP = (disturbance detected OR HIF declared) AND security gate."""
    det = d[pre + "detected"].values.astype(bool)
    if gam_th is not None:
        det = det | (d[pre + "gamma_sus"].values > gam_th)
    g = d[pre + "gate"].values.astype(bool)
    if r_max is not None or beta is not None:
        rm = 0.15 if r_max is None else r_max
        bb = BETA if beta is None else beta
        dm = d[pre + "d_med"].values
        g = (dm >= -0.05 * L_LINE) & (dm <= bb * L_LINE) & (d[pre + "r_d"].values <= rm)
    return det & g


# ------------------------------------------------------------------- main
def main():
    d = load()
    H = {}
    rows = []
    ext = d[d.is_ext]
    z1 = d[d.zone1]

    # --- 1. threshold calibration on the external / no-fault set only -----
    # Gamma_th is calibrated on the NO-FAULT set only: an external fault also
    # raises Gamma_HIF, and it is the security gate (21) -- not the threshold --
    # that restrains for it (Sec. 4.1).  The sustained value is used, so a
    # single-window inception spike cannot declare a HIF.
    # Only an external or no-fault case that ALSO passes the security gate can
    # produce a false HIF trip (Sec. 4.1 makes both conditions necessary), so
    # Gamma_th is calibrated on that subset.
    nof = ext[ext.kind == "nofault"]
    cal = ext[ext.m_gate.values]
    gam_cal = cal.m_gamma_sus.values if len(cal) >= 20 else ext.m_gamma_sus.values
    gam_nof = nof.m_gamma_sus.values
    gam_ext = ext.m_gamma_sus.values
    gam_th_cal = float(np.quantile(gam_cal, 0.99))
    H["Gamma_th_calibrated"] = gam_th_cal
    H["Gamma_th_declared"] = 0.25
    # The OPERATING point is the declared setting of Table VI.  The calibrated
    # value is reported alongside as a sensitivity, never substituted for it:
    # a threshold fitted to the evaluation set is not a declared setting, and
    # reporting performance at a fitted threshold as though it were the
    # designed one is precisely the failure this benchmark exists to avoid.
    gam_th = 0.25
    H["Gamma_th_used"] = gam_th
    H["n_nofault_calibration"] = int(len(nof))
    H["n_gate_passing_external"] = int(len(cal))
    H["n_external_calibration"] = int(len(ext))

    # --- 2. security / dependability trade-off over r_max and beta --------
    sweep = []
    for rm in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 1e9]:
        tz = trip(z1, gam_th=gam_th, r_max=rm)
        te = trip(ext, gam_th=gam_th, r_max=rm)
        D, Dl, Dh = wilson(tz.sum(), len(tz))
        S, Sl, Sh = wilson((~te).sum(), len(te))
        sweep.append(dict(r_max=rm, D=D, D_lo=Dl, D_hi=Dh, S=S, S_lo=Sl, S_hi=Sh,
                          n_int=len(tz), n_ext=len(te)))
    sw = pd.DataFrame(sweep)
    sw.to_csv(os.path.join(TBL, "sweep_rmax.csv"), index=False)
    # Operating-point rule.  If some r_max attains the security target, take the
    # one with the largest dependability among those.  If none does -- which is
    # the case here, because security is limited by the reverse-fault failure
    # mode and not by the dispersion criterion -- the DECLARED value of Table VI
    # is retained rather than widened, since widening it would buy dependability
    # by giving up security with no principled stopping point.
    ok = None
    for tgt in (0.99, 0.98, 0.95):
        cand = sw[sw.S >= tgt]
        if len(cand):
            ok = cand
            break
    if ok is not None:
        r_sel = float(ok.r_max.iloc[int(np.argmax(ok.D.values))])
        H["S_target_met"] = float(tgt)
        H["r_max_rule"] = ("largest dependability subject to the security target "
                           f"S >= {tgt}")
    else:
        r_sel = 0.15
        H["S_target_met"] = None
        H["r_max_rule"] = ("no value of r_max attains S >= 0.95, so the declared "
                           "setting of Table VI is retained; security is limited "
                           "by the close-in reverse-fault case and not by the "
                           "dispersion criterion")
    H["S_best_attainable"] = float(sw.S.max())
    H["D_at_best_S"] = float(sw.D.iloc[int(np.argmax(sw.S.values))])
    H["r_max_declared"] = 0.15
    H["r_max_selected"] = r_sel

    # --- 3. headline dependability and security ---------------------------
    for pre, tag in [("m_", "measured"), ("o_", "ideal_oracle"), ("s_", "saturating_CT")]:
        tz = trip(z1, pre=pre, gam_th=gam_th, r_max=r_sel)
        te = trip(ext, pre=pre, gam_th=gam_th, r_max=r_sel)
        rows.append(wrow(f"Dependability D ({tag})", tz.sum(), len(tz)))
        rows.append(wrow(f"Security S ({tag})", (~te).sum(), len(te)))
    tz = trip(z1, gam_th=gam_th, r_max=r_sel)
    te = trip(ext, gam_th=gam_th, r_max=r_sel)
    H["D"], H["D_lo"], H["D_hi"] = wilson(tz.sum(), len(tz))
    H["S"], H["S_lo"], H["S_hi"] = wilson((~te).sum(), len(te))
    H["n_zone1"], H["n_ext"] = int(len(z1)), int(len(ext))
    # cluster-robust intervals, reported alongside the Wilson intervals
    H["D_cl_lo"], H["D_cl_hi"], H["D_n_clusters"] = cluster_ci(tz, z1.case_id.values)
    H["D_op_lo"], H["D_op_hi"], H["D_n_oppoints"] = cluster_ci(
        tz, [f"{a}|{b}" for a, b in zip(z1.d_km.values, z1.Rf_ohm.values)])
    H["S_cl_lo"], H["S_cl_hi"], H["S_n_clusters"] = cluster_ci(
        ~te, [f"{a}|{b}" for a, b in zip(ext.kind.values, ext.subkind.values)])
    rows.append(dict(metric="Dependability D, cluster interval over taxonomy classes",
                     n=int(len(tz)), successes=int(tz.sum()), estimate=float(tz.mean()),
                     ci95_lo=H["D_cl_lo"], ci95_hi=H["D_cl_hi"]))
    rows.append(dict(metric="Dependability D, cluster interval over (position, R_f) points",
                     n=int(len(tz)), successes=int(tz.sum()), estimate=float(tz.mean()),
                     ci95_lo=H["D_op_lo"], ci95_hi=H["D_op_hi"]))
    rows.append(dict(metric="Security S, cluster interval over disturbance classes",
                     n=int(len(te)), successes=int((~te).sum()),
                     estimate=float((~te).mean()),
                     ci95_lo=H["S_cl_lo"], ci95_hi=H["S_cl_hi"]))
    rows.append(wrow("Dependability, declared settings (r_max = 0.15)",
                     trip(z1, gam_th=gam_th, r_max=0.15).sum(), len(z1)))
    rows.append(wrow("Security, declared settings (r_max = 0.15)",
                     (~trip(ext, gam_th=gam_th, r_max=0.15)).sum(), len(ext)))
    # sensitivity: the same quantities at a Gamma_th calibrated on the
    # gate-passing external set rather than at the declared setting
    tzc = trip(z1, gam_th=gam_th_cal, r_max=r_sel)
    tec = trip(ext, gam_th=gam_th_cal, r_max=r_sel)
    rows.append(wrow(f"Dependability at a calibrated Gamma_th = {gam_th_cal:.3f}",
                     tzc.sum(), len(tzc)))
    rows.append(wrow(f"Security at a calibrated Gamma_th = {gam_th_cal:.3f}",
                     (~tec).sum(), len(tec)))
    H["D_at_calibrated_gamma_th"] = float(tzc.mean())
    H["S_at_calibrated_gamma_th"] = float((~tec).mean())

    # per family
    fam = []
    for (nc, g), s in z1.groupby(["nc", "grounded"]):
        t = trip(s, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson(t.sum(), len(t))
        fam.append(dict(conductors=nc, grounded=int(g), n=len(s), D=p, lo=lo, hi=hi,
                        trip_ms_med=float(np.nanmedian(s.m_trip_delay_ms)),
                        trip_ms_p95=float(np.nanpercentile(s.m_trip_delay_ms.dropna(), 95))
                        if s.m_trip_delay_ms.notna().any() else np.nan))
    pd.DataFrame(fam).to_csv(os.path.join(TBL, "dependability_by_family.csv"), index=False)

    # per group (shunt / open / open+shunt / evolving)
    grp = []
    for g, s in d[d.kind == "internal"].groupby("group"):
        sz = s[s.d_km <= BETA * L_LINE]
        t = trip(sz, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson(t.sum(), len(t))
        det = sz.m_detected
        grp.append(dict(group=g, n=len(sz), detection=det.mean(), D=p, lo=lo, hi=hi,
                        trip_ms_med=float(np.nanmedian(sz.m_trip_delay_ms))))
    pd.DataFrame(grp).to_csv(os.path.join(TBL, "dependability_by_group.csv"), index=False)

    # --- 4. trip time ------------------------------------------------------
    tt = z1.m_trip_delay_ms.dropna().values
    H["trip_ms_median"] = float(np.median(tt))
    H["trip_ms_p95"] = float(np.percentile(tt, 95))
    H["trip_ms_max"] = float(tt.max())
    H["trip_ms_ci"] = boot_ci(tt)

    # --- 5. fault location -------------------------------------------------
    loc = []
    sel = z1[(z1.group == "A_shunt")]
    for lbl, s in [("all", sel)] + [(f"Rf={r} ohm", sel[sel.Rf_ohm == r])
                                    for r in sorted(sel.Rf_ohm.unique())]:
        e = s.e_d.values
        e = e[np.isfinite(e)]
        if len(e) < 3:
            continue
        lo, hi = boot_ci(e)
        loc.append(dict(subset=lbl, n=len(e), mean=e.mean(), median=np.median(e),
                        sd=e.std(ddof=1), p95=np.percentile(e, 95), max=e.max(),
                        med_ci_lo=lo, med_ci_hi=hi,
                        fail_rate=float((e > 10.0).mean())))
    for lbl, s in [(f"d={dk} km", sel[sel.d_km == dk]) for dk in sorted(sel.d_km.unique())]:
        e = s.e_d.values; e = e[np.isfinite(e)]
        loc.append(dict(subset=lbl, n=len(e), mean=e.mean(), median=np.median(e),
                        sd=e.std(ddof=1), p95=np.percentile(e, 95), max=e.max(),
                        med_ci_lo=np.nan, med_ci_hi=np.nan,
                        fail_rate=float((e > 10.0).mean())))
    pd.DataFrame(loc).to_csv(os.path.join(TBL, "location_error.csv"), index=False)
    e_all = sel.e_d.values[np.isfinite(sel.e_d.values)]
    H["e_d_median"] = float(np.median(e_all)); H["e_d_p95"] = float(np.percentile(e_all, 95))
    H["e_d_mean"] = float(e_all.mean()); H["e_d_max"] = float(e_all.max())
    H["RMSE_d_km"] = float(np.sqrt(np.mean((sel.m_d_hat - sel.d_km) ** 2)))
    lowR = sel[sel.Rf_ohm <= 25]
    H["e_d_median_lowRf"] = float(np.median(lowR.e_d))
    H["e_d_p95_lowRf"] = float(np.percentile(lowR.e_d, 95))

    # --- 6. HIF -------------------------------------------------------------
    hif = d[d.kind == "hif"]
    det_h = (hif.m_gamma_sus.values > gam_th) & hif.m_gate.values
    rows.append(wrow("HIF detection probability (calibrated Gamma_th)",
                     det_h.sum(), len(det_h)))
    rows.append(wrow("HIF detection probability (declared Gamma_th = 0.25)",
                     ((hif.m_gamma_sus.values > 0.25) & hif.m_gate.values).sum(), len(hif)))
    H["hif_Pd"], H["hif_Pd_lo"], H["hif_Pd_hi"] = wilson(det_h.sum(), len(det_h))
    H["hif_n"] = int(len(hif))
    H["hif_false_alarm"] = float(((gam_ext > gam_th) & ext.m_gate.values).mean())
    H["hif_false_alarm_nofault"] = float(((gam_nof > gam_th) & nof.m_gate.values).mean())
    # ROC over the HIF-vs-external decision
    gh = np.where(hif.m_gate.values, hif.m_gamma_sus.values, -1.0)
    ge = np.where(ext.m_gate.values, gam_ext, -1.0)
    ths = np.unique(np.quantile(np.concatenate([gh, ge]), np.linspace(0, 1, 300)))
    roc = [dict(threshold=t, Pd=float((gh > t).mean()),
                Pfa=float((ge > t).mean())) for t in ths]
    pd.DataFrame(roc).to_csv(os.path.join(TBL, "hif_roc.csv"), index=False)
    fpr = np.array([r["Pfa"] for r in roc]); tpr = np.array([r["Pd"] for r in roc])
    o = np.argsort(fpr)
    H["hif_auc"] = float(np.trapezoid(tpr[o], fpr[o]))
    H["hif_gamma_only_Pd"] = float((hif.m_gamma_sus.values > gam_th).mean())
    H["hif_gate_pass"] = float(hif.m_gate.values.mean())
    H["hif_gamma_only_th_nofault"] = float(np.quantile(gam_nof, 0.99))
    gth2 = float(np.quantile(gam_nof, 0.99))
    H["hif_Pd_gamma_only_nofault_th"] = float((hif.m_gamma_sus.values > gth2).mean())
    ghs = hif.m_gamma_sus.values
    ths2 = np.unique(np.quantile(np.concatenate([ghs, gam_nof]), np.linspace(0, 1, 300)))
    roc2 = [dict(threshold=t, Pd=float((ghs > t).mean()),
                 Pfa=float((gam_nof > t).mean())) for t in ths2]
    pd.DataFrame(roc2).to_csv(os.path.join(TBL, "hif_roc_gamma_only.csv"), index=False)
    f2 = np.array([r["Pfa"] for r in roc2]); t2 = np.array([r["Pd"] for r in roc2])
    o2 = np.argsort(f2)
    H["hif_auc_gamma_only"] = float(np.trapezoid(t2[o2], f2[o2]))
    for sub in ("quiet", "load_switch", "power_swing"):
        g = nof[nof.subkind == sub].m_gamma_sus.values
        if len(g) < 5:
            continue
        th = np.unique(np.quantile(np.concatenate([ghs, g]), np.linspace(0, 1, 200)))
        rr = [dict(threshold=t, Pd=float((ghs > t).mean()), Pfa=float((g > t).mean()))
              for t in th]
        pd.DataFrame(rr).to_csv(os.path.join(TBL, f"hif_roc_vs_{sub}.csv"), index=False)
        fp = np.array([x["Pfa"] for x in rr]); tp = np.array([x["Pd"] for x in rr])
        oo = np.argsort(fp)
        H[f"hif_auc_vs_{sub}"] = float(np.trapezoid(tp[oo], fp[oo]))
        H[f"Gamma_th_vs_{sub}"] = float(np.quantile(g, 0.99))
        H[f"hif_Pd_vs_{sub}"] = float((ghs > np.quantile(g, 0.99)).mean())
    hs = []
    for lbl, s in ([("all", hif)]
                   + [(f"d={k} km", hif[hif.d_km == k]) for k in sorted(hif.d_km.unique())]
                   + [(f"earthing Zn={z}", hif[hif.Zn_ohm == z])
                      for z in sorted(hif.Zn_ohm.unique())]):
        p, lo, hi = wilson(((s.m_gamma_sus.values > gam_th) & s.m_gate.values).sum(), len(s))
        hs.append(dict(subset=lbl, n=len(s), Pd=p, lo=lo, hi=hi,
                       i_fault_pk_med=float(np.median(s.i_fault_pk)),
                       kappa_med=float(np.median(s.m_kappa))))
    pd.DataFrame(hs).to_csv(os.path.join(TBL, "hif_detection.csv"), index=False)
    # --- 5b. where the declared Gamma_th actually sits --------------------
    # Gamma_HIF has a non-zero quiescent floor on an untransposed line, because
    # the natural unbalance produces standing zero- and negative-sequence
    # current with no fault present.  The floor, the arc value and the value
    # against fault resistance are therefore tabulated together, so that the
    # declared threshold can be read against all three.
    H["gamma_nofault_median"] = float(np.median(gam_nof)) if len(gam_nof) else np.nan
    H["gamma_nofault_p99"] = float(np.quantile(gam_nof, 0.99)) if len(gam_nof) else np.nan
    H["hif_gamma_sus_median"] = float(np.median(hif.m_gamma_sus))
    H["hif_gamma_sus_p95"] = float(np.percentile(hif.m_gamma_sus, 95))
    gr = []
    for rf, sgr in z1.groupby("Rf_ohm"):
        gr.append(dict(Rf_ohm=float(rf), n=len(sgr),
                       gamma_sus_med=float(np.median(sgr.m_gamma_sus)),
                       gamma_sus_lo=float(np.percentile(sgr.m_gamma_sus, 5)),
                       gamma_sus_hi=float(np.percentile(sgr.m_gamma_sus, 95)),
                       frac_above_th=float((sgr.m_gamma_sus > gam_th).mean())))
    gr.append(dict(Rf_ohm=np.nan, n=len(hif),
                   gamma_sus_med=float(np.median(hif.m_gamma_sus)),
                   gamma_sus_lo=float(np.percentile(hif.m_gamma_sus, 5)),
                   gamma_sus_hi=float(np.percentile(hif.m_gamma_sus, 95)),
                   frac_above_th=float((hif.m_gamma_sus > gam_th).mean())))
    grd = pd.DataFrame(gr)
    grd.to_csv(os.path.join(TBL, "gamma_by_Rf.csv"), index=False)
    fin = grd.dropna(subset=["Rf_ohm"]).sort_values("Rf_ohm")
    below = fin[fin.gamma_sus_med < gam_th]
    H["gamma_Rf_first_below_th"] = (float(below.Rf_ohm.iloc[0]) if len(below)
                                    else float("nan"))
    above = fin[fin.gamma_sus_med >= gam_th]
    H["gamma_Rf_last_above_th"] = (float(above.Rf_ohm.iloc[-1]) if len(above)
                                   else float("nan"))
    H["gamma_bolted_median"] = float(fin.gamma_sus_med.iloc[0]) if len(fin) else np.nan

    H["kappa_hif_median"] = float(np.median(hif.m_kappa))
    H["kappa_bolted_median"] = float(np.median(z1[z1.Rf_ohm == 0].m_kappa))
    # i_fault_pk is the current into the earth at the fault node.  For a
    # single-conductor arc that is the arc current; for a two-conductor
    # grounded fault it is the RESIDUAL of the two branches, which is a
    # different quantity, so the arc-current envelope is reported over the
    # single-conductor classes alone.
    h1 = hif[hif.nc == 1]
    H["hif_ifault_pk_min"] = float(h1.i_fault_pk.min())
    H["hif_ifault_pk_max"] = float(h1.i_fault_pk.max())
    H["hif_ifault_pk_med"] = float(h1.i_fault_pk.median())
    H["n_hif_single"] = int(len(h1))

    # --- 5c. residual cancellation between antipodal conductors ------------
    # A six-phase-specific effect with a direct protection consequence: the
    # zero-sequence contributions of two conductors 180 deg apart very nearly
    # cancel, so a two-conductor-to-ground fault on an antipodal pair produces
    # almost no earth current even when it is bolted.
    def _sep(mb):
        b = [i for i in range(6) if (int(mb) >> i) & 1]
        if len(b) != 2:
            return np.nan
        dd = abs(b[0] - b[1])
        return float(min(dd, 6 - dd))
    rc = []
    for lbl, sub in [("bolted (R_f = 0)",
                      d[(d.kind == "internal") & (d.nc == 2) & (d.grounded == 1)
                        & (d.Rf_ohm == 0)]),
                     ("high-impedance arc",
                      d[(d.kind == "hif") & (d.nc == 2)])]:
        sub = sub.copy(); sub["sep"] = sub.mask_bits.map(_sep)
        for sp, g in sub.groupby("sep"):
            rc.append(dict(fault=lbl, separation_deg=60.0 * sp, n=len(g),
                           i_earth_med_A=float(np.median(g.i_fault_pk)),
                           detected=float(g.m_detected.mean()),
                           gate=float(g.m_gate.mean())))
    rcd = pd.DataFrame(rc)
    rcd.to_csv(os.path.join(TBL, "residual_cancellation.csv"), index=False)
    bo = rcd[rcd.fault.str.startswith("bolted")].sort_values("separation_deg")
    if len(bo) >= 2:
        H["resid_bolted_60_A"] = float(bo.i_earth_med_A.iloc[0])
        H["resid_bolted_180_A"] = float(bo.i_earth_med_A.iloc[-1])
        H["resid_bolted_ratio"] = float(bo.i_earth_med_A.iloc[0]
                                        / max(bo.i_earth_med_A.iloc[-1], 1e-9))

    # --- 6b. dependability in the conventional evaluation regime -----------
    # The regime on which [1]-[12] report accuracy: low fault resistance, high
    # signal-to-noise ratio, exact line constants.
    conv = z1[(z1.Rf_ohm <= 25) & (z1.snr >= 40) & (z1.par_err == 0)]
    tc = trip(conv, gam_th=gam_th, r_max=r_sel)
    rows.append(wrow("Dependability, conventional regime (Rf<=25, SNR>=40 dB, exact theta)",
                     tc.sum(), len(tc)))
    H["D_conventional"], H["D_conv_lo"], H["D_conv_hi"] = wilson(tc.sum(), len(tc))
    H["n_conventional"] = int(len(conv))
    ec = conv.e_d.values[np.isfinite(conv.e_d.values)]
    H["e_d_median_conventional"] = float(np.median(ec)) if len(ec) else np.nan
    H["e_d_p95_conventional"] = float(np.percentile(ec, 95)) if len(ec) else np.nan
    dr = []
    for rf, s in z1.groupby("Rf_ohm"):
        t = trip(s, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson(t.sum(), len(t))
        dr.append(dict(Rf_ohm=rf, n=len(s), D=p, lo=lo, hi=hi,
                       detection=float(s.m_detected.mean()),
                       gate=float(s.m_gate.mean())))
    pd.DataFrame(dr).to_csv(os.path.join(TBL, "dependability_by_Rf.csv"), index=False)

    # --- 7. stress dimensions ----------------------------------------------
    st = []
    for factor, col in [("SNR (dB)", "snr"), ("CT remanence (pu)", "ct_rem"),
                        ("CT burden (ohm)", "ct_burd"),
                        ("line-parameter error", "par_err"),
                        ("load level (pu)", "load_pu"), ("load type", "ldtype"),
                        ("source S_sc (MVA)", "S_sc"), ("source X/R", "XR"),
                        ("fault resistance (ohm)", "Rf_ohm")]:
        for lev, s in z1.groupby(col):
            t = trip(s, gam_th=gam_th, r_max=r_sel)
            te2 = ext[ext[col] == lev] if col in ext.columns else ext
            tsec = trip(te2, gam_th=gam_th, r_max=r_sel) if len(te2) else np.array([])
            p, lo, hi = wilson(t.sum(), len(t))
            e = s.e_d.values[np.isfinite(s.e_d.values)]
            st.append(dict(factor=factor, level=lev, n=len(s), D=p, D_lo=lo, D_hi=hi,
                           S=float((~tsec).mean()) if len(tsec) else np.nan,
                           n_ext=len(tsec),
                           trip_ms_med=float(np.nanmedian(s.m_trip_delay_ms)),
                           e_d_med=float(np.median(e)) if len(e) else np.nan,
                           e_d_p95=float(np.percentile(e, 95)) if len(e) else np.nan))
    pd.DataFrame(st).to_csv(os.path.join(TBL, "stress_matrix.csv"), index=False)

    # --- 7b. what the half-cycle asymmetry statistic actually tracks --------
    # kappa_i is claimed in Section 2.5 to carry the arc signature.  Its median
    # is therefore tabulated against CT remanence, separately for case types
    # that differ enormously in fault current, so that the reader can see
    # whether it sorts by the fault or by the instrument transformer.
    kr = []
    groups = [("no-fault", d[d.kind == "nofault"]),
              ("external", d[d.kind.isin(["forward_external", "reverse"])]),
              ("internal, bolted", z1[z1.Rf_ohm == 0]),
              ("internal, Rf >= 300 ohm", z1[z1.Rf_ohm >= 300]),
              ("high-impedance arc", d[d.kind == "hif"])]
    for lev in sorted(d.ct_rem.unique()):
        row = dict(ct_remanence=float(lev))
        for lbl, g in groups:
            sub = g[g.ct_rem == lev]
            row[lbl] = float(np.median(sub.m_kappa)) if len(sub) else np.nan
            row["n_" + lbl] = int(len(sub))
        kr.append(row)
    krd = pd.DataFrame(kr)
    krd.to_csv(os.path.join(TBL, "kappa_by_remanence.csv"), index=False)
    lo_r, hi_r = float(krd.ct_remanence.min()), float(krd.ct_remanence.max())
    cols_k = [lbl for lbl, _ in groups]
    H["kappa_rem_lo"] = lo_r
    H["kappa_rem_hi"] = hi_r
    H["kappa_at_rem_lo_min"] = float(krd.iloc[0][cols_k].min())
    H["kappa_at_rem_lo_max"] = float(krd.iloc[0][cols_k].max())
    H["kappa_at_rem_hi_min"] = float(krd.iloc[-1][cols_k].min())
    H["kappa_at_rem_hi_max"] = float(krd.iloc[-1][cols_k].max())
    # spread across case type at fixed remanence, against spread across
    # remanence at fixed case type: if the second dominates, kappa_i is a
    # property of the CT and not of the fault
    H["kappa_spread_across_casetype"] = float(
        (krd[cols_k].max(axis=1) - krd[cols_k].min(axis=1)).max())
    H["kappa_spread_across_remanence"] = float(
        (krd[cols_k].max(axis=0) - krd[cols_k].min(axis=0)).max())

    # --- 8. measurement-chain attribution (oracle vs measured vs sat CT) ----
    att = []
    for pre, tag in [("o_", "ideal current (oracle)"), ("m_", "declared CT 5P20"),
                     ("s_", "saturation-forcing CT")]:
        t = trip(z1, pre=pre, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson(t.sum(), len(t))
        ecol = dict(o_="e_d_or", m_="e_d", s_="e_d_sat")[pre]
        e = z1[z1.group == "A_shunt"][ecol].values
        e = e[np.isfinite(e)]
        att.append(dict(chain=tag, n=len(z1), D=p, D_lo=lo, D_hi=hi,
                        e_d_med=float(np.median(e)), e_d_p95=float(np.percentile(e, 95)),
                        q_CT_med=float(np.median(z1[pre + "q_CT"])),
                        trip_ms_med=float(np.nanmedian(z1[pre + "trip_delay_ms"]))))
    pd.DataFrame(att).to_csv(os.path.join(TBL, "measurement_chain.csv"), index=False)
    H["q_CT_declared_med"] = float(d.m_q_CT.median())
    H["q_CT_forcing_med"] = float(d.s_q_CT.median())
    H["q_CT_declared_max"] = float(d.m_q_CT_max.max())
    H["q_CT_forcing_max"] = float(d.s_q_CT_max.max())

    # --- 9. IBR -------------------------------------------------------------
    ibr = d[d.kind == "ibr"]
    ib = []
    for (pen, mode), s in ibr.groupby(["ibr_pen", "ibr_mode"]):
        sz = s[s.d_km <= BETA * L_LINE]
        t = trip(sz, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson(t.sum(), len(t))
        e = sz.e_d.values[np.isfinite(sz.e_d.values)]
        ib.append(dict(penetration=pen, control=["grid-following", "grid-forming"][int(mode)],
                       n=len(sz), D=p, D_lo=lo, D_hi=hi,
                       trip_ms_med=float(np.nanmedian(sz.m_trip_delay_ms)),
                       i_fault_pk_med=float(np.median(sz.i_fault_pk)),
                       detected=float(sz.m_detected.mean()),
                       gate=float(sz.m_gate.mean()),
                       e_d_med=float(np.median(e)) if len(e) else np.nan,
                       e_d_p95=float(np.percentile(e, 95)) if len(e) else np.nan))
    ibd = pd.DataFrame(ib)
    ibd.to_csv(os.path.join(TBL, "ibr_sweep.csv"), index=False)
    for lbl, mode in (("gfl", "grid-following"), ("gfm", "grid-forming")):
        sub = ibd[ibd.control == mode].sort_values("penetration")
        if len(sub) >= 2:
            H[f"ibr_{lbl}_if_lo"] = float(sub.i_fault_pk_med.iloc[0])
            H[f"ibr_{lbl}_if_hi"] = float(sub.i_fault_pk_med.iloc[-1])
            H[f"ibr_{lbl}_pen_hi"] = float(sub.penetration.iloc[-1])

    # --- 10. external-fault breakdown ---------------------------------------
    exr = []
    for (k, sk), s in ext.groupby(["kind", "subkind"]):
        t = trip(s, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson((~t).sum(), len(t))
        exr.append(dict(kind=k, subkind=sk, n=len(s), S=p, lo=lo, hi=hi,
                        detected=float(s.m_detected.mean()),
                        gamma_sus_max=float(s.m_gamma_sus.max())))
    pd.DataFrame(exr).to_csv(os.path.join(TBL, "security_breakdown.csv"), index=False)
    rv = ext[ext.kind == "reverse"]
    rr2 = []
    for dk, s in rv.groupby("d_km"):
        t = trip(s, gam_th=gam_th, r_max=r_sel)
        p, lo, hi = wilson((~t).sum(), len(t))
        rr2.append(dict(source_side_km=dk, n=len(s), S=p, lo=lo, hi=hi,
                        d_med_median=float(np.median(s.m_d_med)),
                        r_d_median=float(np.median(s.m_r_d))))
    pd.DataFrame(rr2).to_csv(os.path.join(TBL, "security_reverse_by_distance.csv"),
                             index=False)
    H["S_reverse"] = float((~trip(rv, gam_th=gam_th, r_max=r_sel)).mean())
    H["S_forward_external"] = float((~trip(ext[ext.kind == "forward_external"],
                                           gam_th=gam_th, r_max=r_sel)).mean())
    H["S_nofault"] = float((~trip(nof, gam_th=gam_th, r_max=r_sel)).mean())

    pd.DataFrame(rows).to_csv(os.path.join(TBL, "headline_proportions.csv"), index=False)
    # merge rather than overwrite: validate.py writes the model-verification
    # keys into the same file and must not be clobbered by this pass
    _hp = os.path.join(ROOT, "out", "Data", "headline.json")
    if os.path.exists(_hp):
        _old = json.load(open(_hp)); _old.update(H); H = _old
    json.dump(H, open(_hp, "w"),
              indent=1, default=float)
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in H.items() if not isinstance(v, (list, tuple))},
                     indent=1))


if __name__ == "__main__":
    main()
