#!/usr/bin/env python3
"""
hif_ablation.py -- why the high-impedance-fault index does not separate an arc
from quiescent operation, term by term.

Two controlled experiments, both run through the same solver, measurement chain
and relay as the main benchmark:

  (1) TERM ABLATION.  Gamma_HIF of (18) is a weighted sum of four terms.  Each
      term is evaluated separately on an arc set and on a no-fault set, and the
      area under the detection/false-alarm curve is reported for each term
      alone and for the index with each term removed.  This identifies which
      term carries the discrimination and which does not.

  (2) SYMMETRY CONTROL.  The manuscript attributes the high-impedance-fault
      signature to the V_s+ != V_s-, R_p+ != R_p-, I_0+ != I_0- asymmetry of
      the arc model (6).  The same arc set is therefore re-run with all three
      asymmetries removed.  If kappa_i is unchanged, the statistic is not
      responding to the arc asymmetry at all, and the design rationale for the
      weight a_4 does not hold.
"""
import os, json
import numpy as np
import pandas as pd
import spconst as sc, spemt, spmeas, sprelay

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TBL = os.path.join(ROOT, "out", "Tables")
DAT = os.path.join(ROOT, "out", "Data")
FS, FC, T_FAULT, T_END = 5000.0, 1800.0, 0.10, 0.24
SEED = 20260829
A = dict(a1=0.35, a2=0.25, a3=0.20, a4=0.20)


def run(n, arc, symmetric=False, quiet=False, seed=SEED, rem_mode="zero"):
    rng = np.random.default_rng(seed)
    d = rng.choice([15., 25., 35., 45., 55., 65., 75., 85.], n).astype(float)
    spec = dict(d_km=d, Rf_ohm=np.zeros(n), grounded=np.ones(n, bool),
                mask=np.zeros((n, 6), bool), open_mask=np.zeros((n, 6), bool),
                S_sc=rng.choice([500., 1500., 3500., 8000.], n),
                XR=rng.choice([5., 15., 30.], n), Zn_ohm=np.zeros(n),
                load_pu=rng.choice([0.3, 0.7, 1.0], n),
                inception_deg=rng.uniform(0, 360, n))
    spec["mask"][np.arange(n), rng.integers(0, 6, n)] = True
    if quiet:
        spec["mask"][:] = False           # nothing happens at t_fault
    if symmetric:
        spec.update(Vs_plus=np.full(n, 6.15e3), Vs_minus=np.full(n, 6.15e3),
                    I0_plus=np.full(n, 10.0), I0_minus=np.full(n, 10.0),
                    Rp_min=np.full(n, 2.5e3), Rp_max=np.full(n, 6.0e3))
    b = spemt.Batch(spec, dt=20e-6, t_end=T_END, t_fault=T_FAULT, arc=arc, seed=seed)
    r = b.run(decim=1)
    dtr = r["dt"]
    pref = slice(int(0.5 * T_FAULT / dtr), int(0.95 * T_FAULT / dtr))
    snr = rng.choice([20., 30., 40., 60.], n)
    rem = (np.zeros(n) if rem_mode == "zero"
           else rng.choice([0.0, 0.20, 0.50, 0.80], n))
    burd = np.full(n, 0.5) if rem_mode == "zero" else rng.choice([0.5, 1.0], n)
    im = spmeas.add_noise(spmeas.ct_secondary(r["i"], dtr, burd, rem),
                          snr, rng, ref_slice=pref)
    vm = spmeas.add_noise(r["v"].astype(float), snr, rng, ref_slice=pref)
    vf, _ = spmeas.antialias_decimate(vm, dtr, FC, FS)
    iff, _ = spmeas.antialias_decimate(im, dtr, FC, FS)
    R = sprelay.Relay()
    o = R.process(np.asarray(vf, float), np.asarray(iff, float), 1.0 / FS, T_FAULT)
    k = int(T_FAULT * FS) + 200
    Ib = R.I_base
    # kappa_i as specified, and kappa_i on a d.c.-removed residual: the mimic
    # filter of Section 5.3 applied to the residual before the half-cycle
    # integrals are formed.
    ir = np.asarray(iff, float).sum(-1)
    imf, g = sprelay.mimic_filter(ir, 1.0 / FS, R.tau_mimic)
    Nw = int(R.s["Nw"])
    res_rms = np.sqrt(sprelay.moving_sum(imf ** 2, Nw) / Nw)
    kap_dc = np.where(res_rms > 0.02 * Ib,
                      sprelay.half_cycle_asymmetry(imf, Nw), 0.0)
    terms = dict(
        # a1 exactly as (18) prints it: the TOTAL zero- and negative-sequence
        # currents.  t1s is the superimposed variant, retained as the obvious
        # design alternative and reported alongside.
        t1=(np.abs(o["Is"][k][:, 0]) + np.abs(o["Is"][k][:, 5])) / Ib,
        t1s=(np.abs(o["dIs"][k][:, 0]) + np.abs(o["dIs"][k][:, 5])) / Ib,
        t2=o["eta_tr"][k], t3=o["THD"][k], t4=o["kappa"][k],
        t4dc=kap_dc[k])
    arcpk = np.abs(r["i_fault"][int(T_FAULT / dtr):]).max(0)
    return terms, arcpk


def auc(pos, neg):
    """Area under the detection/false-alarm curve, by rank."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x)
    r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    n1, n0 = len(pos), len(neg)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    N = 256
    print("running arc, symmetric-arc and no-fault sets ...", flush=True)
    ta, pk = run(N, arc=True)
    ts, pks = run(N, arc=True, symmetric=True, seed=SEED + 1)
    tq, _ = run(N, arc=False, quiet=True, seed=SEED + 2)
    print("repeating through the declared CT burden and remanence sweep ...", flush=True)
    ta2, _ = run(N, arc=True, seed=SEED + 3, rem_mode="sampled")
    tq2, _ = run(N, arc=False, quiet=True, seed=SEED + 4, rem_mode="sampled")

    rows = []
    for lab, key in [("a1 (|I_0|+|I_5|)/I_base", "t1"), ("a2 eta_tr", "t2"),
                     ("a3 THD_I", "t3"), ("a4 kappa_i", "t4")]:
        rows.append(dict(component=lab,
                         median_arc=float(np.median(ta[key])),
                         median_nofault=float(np.median(tq[key])),
                         AUC_alone=auc(ta[key], tq[key])))
    g_full = sum(A[f"a{i}"] * ta[f"t{i}"] for i in range(1, 5))
    q_full = sum(A[f"a{i}"] * tq[f"t{i}"] for i in range(1, 5))
    rows.append(dict(component="Gamma_HIF, all four terms",
                     median_arc=float(np.median(g_full)),
                     median_nofault=float(np.median(q_full)),
                     AUC_alone=auc(g_full, q_full)))
    for drop in range(1, 5):
        ga = sum(A[f"a{i}"] * ta[f"t{i}"] for i in range(1, 5) if i != drop)
        gq = sum(A[f"a{i}"] * tq[f"t{i}"] for i in range(1, 5) if i != drop)
        rows.append(dict(component=f"Gamma_HIF with a{drop} term removed",
                         median_arc=float(np.median(ga)),
                         median_nofault=float(np.median(gq)),
                         AUC_alone=auc(ga, gq)))
    rows.append(dict(component="a4 kappa_i on a d.c.-removed residual",
                     median_arc=float(np.median(ta["t4dc"])),
                     median_nofault=float(np.median(tq["t4dc"])),
                     AUC_alone=auc(ta["t4dc"], tq["t4dc"])))
    rows.append(dict(component="a1 on SUPERIMPOSED sequence currents (variant)",
                     median_arc=float(np.median(ta["t1s"])),
                     median_nofault=float(np.median(tq["t1s"])),
                     AUC_alone=auc(ta["t1s"], tq["t1s"])))
    gv = A["a1"] * ta["t1s"] + sum(A[f"a{i}"] * ta[f"t{i}"] for i in (2, 3, 4))
    qv = A["a1"] * tq["t1s"] + sum(A[f"a{i}"] * tq[f"t{i}"] for i in (2, 3, 4))
    rows.append(dict(component="Gamma_HIF with a1 on superimposed currents (variant)",
                     median_arc=float(np.median(gv)), median_nofault=float(np.median(qv)),
                     AUC_alone=auc(gv, qv)))
    # the same quantities through the declared CT burden and remanence sweep
    for lab, key in [("a1 (|I_0|+|I_5|)/I_base", "t1"), ("a3 THD_I", "t3"),
                     ("a4 kappa_i", "t4"),
                     ("a4 kappa_i on a d.c.-removed residual", "t4dc")]:
        rows.append(dict(component=lab + "  [declared CT remanence sweep]",
                         median_arc=float(np.median(ta2[key])),
                         median_nofault=float(np.median(tq2[key])),
                         AUC_alone=auc(ta2[key], tq2[key])))
    gf2 = sum(A[f"a{i}"] * ta2[f"t{i}"] for i in range(1, 5))
    qf2 = sum(A[f"a{i}"] * tq2[f"t{i}"] for i in range(1, 5))
    rows.append(dict(component="Gamma_HIF, all four terms  [declared CT remanence sweep]",
                     median_arc=float(np.median(gf2)), median_nofault=float(np.median(qf2)),
                     AUC_alone=auc(gf2, qf2)))
    gd2 = sum(A[f"a{i}"] * ta2[f"t{i}"] for i in range(1, 4))
    qd2 = sum(A[f"a{i}"] * tq2[f"t{i}"] for i in range(1, 4))
    rows.append(dict(component="Gamma_HIF with a4 removed  [declared CT remanence sweep]",
                     median_arc=float(np.median(gd2)), median_nofault=float(np.median(qd2)),
                     AUC_alone=auc(gd2, qd2)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TBL, "hif_index_ablation.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- symmetry control -------------------------------------------------
    sym = pd.DataFrame([
        dict(quantity="kappa_i, declared asymmetric arc",
             median=float(np.median(ta["t4"])), iqr_lo=float(np.percentile(ta["t4"], 25)),
             iqr_hi=float(np.percentile(ta["t4"], 75))),
        dict(quantity="kappa_i, arc with all three asymmetries removed",
             median=float(np.median(ts["t4"])), iqr_lo=float(np.percentile(ts["t4"], 25)),
             iqr_hi=float(np.percentile(ts["t4"], 75))),
        dict(quantity="kappa_i, no-fault (noise only)",
             median=float(np.median(tq["t4"])), iqr_lo=float(np.percentile(tq["t4"], 25)),
             iqr_hi=float(np.percentile(tq["t4"], 75))),
        dict(quantity="kappa_i, arc, through the declared CT remanence sweep",
             median=float(np.median(ta2["t4"])), iqr_lo=float(np.percentile(ta2["t4"], 25)),
             iqr_hi=float(np.percentile(ta2["t4"], 75))),
        dict(quantity="kappa_i, no-fault, through the declared CT remanence sweep",
             median=float(np.median(tq2["t4"])), iqr_lo=float(np.percentile(tq2["t4"], 25)),
             iqr_hi=float(np.percentile(tq2["t4"], 75))),
        dict(quantity="arc peak current, declared arc (A)",
             median=float(np.median(pk)), iqr_lo=float(np.percentile(pk, 25)),
             iqr_hi=float(np.percentile(pk, 75))),
        dict(quantity="arc peak current, symmetric arc (A)",
             median=float(np.median(pks)), iqr_lo=float(np.percentile(pks, 25)),
             iqr_hi=float(np.percentile(pks, 75))),
    ])
    sym.to_csv(os.path.join(TBL, "hif_symmetry_control.csv"), index=False)
    print()
    print(sym.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    H = json.load(open(os.path.join(DAT, "headline.json")))
    H["hif_auc_full_index"] = float(df[df.component.str.startswith("Gamma_HIF, all")]
                                    .AUC_alone.iloc[0])
    for i, key in enumerate(["t1", "t2", "t3", "t4"], start=1):
        H[f"hif_auc_term_a{i}"] = float(df.AUC_alone.iloc[i - 1])
        H[f"hif_auc_drop_a{i}"] = float(df[df.component ==
                                           f"Gamma_HIF with a{i} term removed"]
                                        .AUC_alone.iloc[0])
    H["kappa_arc_declared"] = float(np.median(ta["t4"]))
    H["kappa_arc_symmetric"] = float(np.median(ts["t4"]))
    H["kappa_nofault"] = float(np.median(tq["t4"]))
    H["kappa_arc_CTsweep"] = float(np.median(ta2["t4"]))
    H["kappa_nofault_CTsweep"] = float(np.median(tq2["t4"]))
    H["hif_auc_kappa_dcremoved"] = auc(ta["t4dc"], tq["t4dc"])
    H["hif_auc_full_index_CTsweep"] = auc(gf2, qf2)
    H["hif_auc_drop_a4_CTsweep"] = auc(gd2, qd2)
    H["hif_auc_a1_CTsweep"] = auc(ta2["t1"], tq2["t1"])
    H["hif_auc_a1_superimposed"] = auc(ta["t1s"], tq["t1s"])
    H["hif_auc_variant_superimposed"] = auc(gv, qv)
    json.dump(H, open(os.path.join(DAT, "headline.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
