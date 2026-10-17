#!/usr/bin/env python3
"""
validate.py -- model-verification measurements quoted in Section 2.2.

  (a) pre-fault agreement between the time-domain solution and the phasor
      solution of the identical network (the admission test of Section 2.2);
  (b) Bergeron representation against an exact distributed two-port;
  (c) real modal transformation against the exact 6x6 series impedance;
  (d) rest-initialisation deviation after one, two and three cycles;
  (e) time-step convergence at 20, 10 and 5 microseconds;
  (f) bolted single-conductor-to-ground current against fault position, and
      the simulated high-impedance arc-current envelope.

Writes out/Tables/model_verification.csv and merges the headline numbers into
out/Data/headline.json.
"""
import os, json
import numpy as np
import pandas as pd
import spconst as sc, spemt, spmeas, sprelay

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TBL = os.path.join(ROOT, "out", "Tables"); os.makedirs(TBL, exist_ok=True)
DAT = os.path.join(ROOT, "out", "Data"); os.makedirs(DAT, exist_ok=True)
FS, FC = 5000.0, 1800.0


def base_spec(d=50.0, Rf=0.0, mask=(0,), grounded=True, n=1, incep=0.0):
    m = np.zeros((n, 6), bool)
    for c in mask:
        m[:, c] = True
    return dict(d_km=np.full(n, d), Rf_ohm=np.full(n, Rf),
                grounded=np.full(n, grounded), mask=m,
                open_mask=np.zeros((n, 6), bool), S_sc=np.full(n, 3500.0),
                XR=np.full(n, 15.0), Zn_ohm=np.zeros(n), load_pu=np.ones(n),
                inception_deg=np.full(n, incep))


def main():
    rows = []
    H = {}

    # (a) pre-fault time-domain vs phasor, and (b),(c) model comparisons
    b = spemt.Batch(base_spec(), dt=20e-6, t_end=0.10, t_fault=1e9)
    r = b.run(decim=1)
    V = r["v"][:, 0, :].astype(float)
    w = int(round(1.0 / (50.0 * r["dt"])))
    rms_td = np.sqrt((V[-4 * w:] ** 2).mean(0))
    Vp = b.phasor_solve(b.stlist[0])
    rms_ph = np.abs(Vp[0, 0:6]) / np.sqrt(2)
    err = np.abs(rms_td - rms_ph) / rms_ph
    H["valid_prefault_pct"] = float(err.max() * 100)
    rows.append(dict(quantity="pre-fault time-domain vs phasor, max relative deviation",
                     value=f"{err.max():.3e}", unit="-",
                     note="admission limit 1e-5 (1e-3 per cent)"))

    Ve = b.phasor_solve(b.stlist[0], kind="exact")
    rel = np.abs(Vp[0, :24] - Ve[0, :24]).max() / np.abs(Ve[0, :24]).max()
    H["valid_bergeron_rel"] = float(rel)
    rows.append(dict(quantity="Bergeron vs exact distributed two-port, node voltage",
                     value=f"{rel:.3e}", unit="-", note="line-model representation error"))

    # (c2) sequence-transform verification: unitarity, power invariance and the
    # index convention declared in Section 2.3.  The convention is what fixes
    # which index Eq. (18) reads, so it is verified explicitly rather than
    # assumed: a balanced set with the conventional 60 deg progressive lag must
    # map to k = 1 (positive sequence) and to nothing else.
    Sm = sc.S6(); Ss = sc.S6_conjugate()
    uni = float(np.abs(Sm @ np.conj(Sm).T - np.eye(6)).max())
    xb = np.array([np.exp(-1j * np.pi / 3.0) ** k for k in range(6)])
    yb = Sm @ xb; ys = Ss @ xb
    pw = abs(float((np.abs(xb) ** 2).sum() - (np.abs(yb) ** 2).sum()))
    H["valid_S6_unitarity"] = uni
    H["valid_S6_power"] = pw
    H["valid_S6_pos_index"] = int(np.argmax(np.abs(yb)))
    H["valid_S6_pos_index_shipped"] = int(np.argmax(np.abs(ys)))
    rows.append(dict(quantity="S6 unitarity, max |S6 S6^H - I|",
                     value=f"{uni:.3e}", unit="-", note="power-invariant form, Eq. (4)"))
    rows.append(dict(quantity="S6 power invariance, |sum|x|^2 - sum|x_s|^2|",
                     value=f"{pw:.3e}", unit="-", note="balanced test vector"))
    rows.append(dict(quantity="index of a balanced 60 deg-lag set under S6",
                     value=f"{H['valid_S6_pos_index']}", unit="-",
                     note="Section 2.3 requires k = 1 (positive sequence)"))
    rows.append(dict(quantity="index of the same set under the conjugate transform",
                     value=f"{H['valid_S6_pos_index_shipped']}", unit="-",
                     note="reverses the declared convention, "
                          "so Eq. (18) would read the load current as unbalance"))

    R = sprelay.Relay()
    H["valid_modalZ_rel"] = float(R.modal_Z_err)
    rows.append(dict(quantity="real modal transformation vs exact 6x6 series impedance",
                     value=f"{R.modal_Z_err:.3e}", unit="-",
                     note="max element-wise relative deviation"))

    # (d) rest vs phasor initialisation
    b2 = spemt.Batch(base_spec(), dt=20e-6, t_end=0.12, t_fault=1e9)
    rp = b2.run(decim=1, init="phasor"); rr = b2.run(decim=1, init="rest")
    ip, ir = rp["i"][:, 0, :].astype(float), rr["i"][:, 0, :].astype(float)
    fs = 1.0 / rp["dt"]; wc = int(round(fs / 50.0))
    for nc in (1, 2, 3):
        k = int(round((nc + 1) / 50.0 * fs))
        a = np.sqrt((ip[k - wc:k] ** 2).mean(0)); c = np.sqrt((ir[k - wc:k] ** 2).mean(0))
        dr = 100 * np.abs(c - a).max() / np.abs(a).max()
        dp = 100 * np.abs(ir[k] - ip[k]).max() / np.abs(ip).max()
        rows.append(dict(quantity=f"rest-initialisation deviation after {nc} cycle(s)",
                         value=f"{dr:.3f} (r.m.s.) / {dp:.3f} (instantaneous)",
                         unit="per cent", note="phasor initialisation is the reference"))
        if nc == 2:
            H["init_dev_2cyc_rms"] = float(dr); H["init_dev_2cyc_pk"] = float(dp)

    # (e) time-step convergence, on the declared base case
    conv = []
    for dt in (20e-6, 10e-6, 5e-6):
        bb = spemt.Batch(base_spec(d=50.0), dt=dt, t_end=0.24, t_fault=0.10)
        rr2 = bb.run(decim=1)
        ii = rr2["i"].astype(float)
        vv, _ = spmeas.antialias_decimate(rr2["v"].astype(float), rr2["dt"], FC, FS)
        iif, _ = spmeas.antialias_decimate(ii, rr2["dt"], FC, FS)
        o = R.process(vv, iif, 1.0 / FS, 0.10)
        kf = int(0.10 * FS)
        Dd = o["D"][kf:, 0]
        tt = float(np.argmax(Dd) / FS * 1e3) if Dd.any() else np.nan
        ipk = float(np.abs(rr2["i_fault"][int(0.10 / rr2["dt"]):, 0]).max())
        loc = R.locate(o, kf + 200, np.array([[True] + [False] * 5]), np.array([True]))
        conv.append(dict(dt_us=dt * 1e6, peak_fault_current_A=ipk,
                         trip_delay_ms=tt, d_hat_km=float(loc["d_hat"][0])))
    cv = pd.DataFrame(conv)
    cv.to_csv(os.path.join(TBL, "timestep_convergence.csv"), index=False)
    H["conv_ipk_spread_pct"] = float(100 * (cv.peak_fault_current_A.max()
                                            - cv.peak_fault_current_A.min())
                                     / cv.peak_fault_current_A.mean())
    H["conv_trip_spread_ms"] = float(cv.trip_delay_ms.max() - cv.trip_delay_ms.min())
    rows.append(dict(quantity="time-step convergence, peak fault-current spread",
                     value=f"{H['conv_ipk_spread_pct']:.3f}", unit="per cent",
                     note="20, 10 and 5 microseconds"))

    # (f) bolted current against position, and the arc-current envelope
    pos = [5., 15., 25., 35., 45., 55., 65., 75., 85., 95.]
    bl = []
    for dd in pos:
        bb = spemt.Batch(base_spec(d=dd), dt=10e-6, t_end=0.02, t_fault=1e9)
        Vf = bb.phasor_solve(bb.stlist[1])
        bl.append(dict(d_km=dd, I_fault_A_rms=float(abs(Vf[0, 24]) * bb.ggs[1][0] / np.sqrt(2))))
    bdf = pd.DataFrame(bl); bdf.to_csv(os.path.join(TBL, "bolted_fault_current.csv"), index=False)
    H["I_bolted_5km_A"] = float(bdf.I_fault_A_rms.iloc[0])
    H["I_bolted_95km_A"] = float(bdf.I_fault_A_rms.iloc[-1])

    arc = []
    for dd in (5., 50., 95.):
        for mode, incep in (("t", 0.0), ("t", 120.0)):
            dtx = 20e-6 if min(dd, 100 - dd) >= 15 else 10e-6
            bb = spemt.Batch(base_spec(d=dd, incep=incep), dt=dtx, t_end=0.24,
                             t_fault=0.10, arc=True, seed=7)
            rr3 = bb.run(decim=1)
            k0 = int(0.12 / rr3["dt"])
            arc.append(dict(d_km=dd, inception_deg=incep,
                            arc_peak_A=float(np.abs(rr3["i_fault"][k0:, 0]).max())))
    adf = pd.DataFrame(arc); adf.to_csv(os.path.join(TBL, "arc_current_envelope.csv"), index=False)
    H["arc_pk_min_A"] = float(adf.arc_peak_A.min())
    H["arc_pk_max_A"] = float(adf.arc_peak_A.max())
    rows.append(dict(quantity="simulated high-impedance arc peak current",
                     value=f"{adf.arc_peak_A.min():.1f} to {adf.arc_peak_A.max():.1f}",
                     unit="A", note="declared envelope 21 to 50 A (Sec. 2.5)"))
    rows.append(dict(quantity="bolted single-conductor-to-ground current, 5 to 95 km",
                     value=f"{bdf.I_fault_A_rms.iloc[0]:.0f} to {bdf.I_fault_A_rms.iloc[-1]:.0f}",
                     unit="A r.m.s.", note="full network, base-case source"))

    pd.DataFrame(rows).to_csv(os.path.join(TBL, "model_verification.csv"), index=False)
    p = os.path.join(DAT, "headline.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(H)
    json.dump(old, open(p, "w"), indent=1, default=float)
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(cv.to_string(index=False))


if __name__ == "__main__":
    main()
