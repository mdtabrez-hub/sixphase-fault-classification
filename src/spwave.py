#!/usr/bin/env python3
"""
spwave.py -- representative waveform records for Fig. 7 and Fig. 13.

Runs a small, explicitly declared set of cases through the same EMT solver,
measurement chain and relay used for the whole benchmark, and stores the
records so that the waveform figures are drawn from the same code path as the
statistics.  Nothing is hand-drawn.
"""
import numpy as np, os, json
import spemt, spmeas, sprelay, spconst as sc

T_FAULT, T_END, FS, FC = 0.10, 0.24, 5000.0, 1800.0
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "out", "Data"))


def one(name, d_km, mask, grounded, Rf, arc=False, external=False,
        relay_at="S", L=100.0, dt=20e-6, incep=0.0, snr=40.0, load=1.0,
        ct_burd=0.5, ct_rem=0.0, seed=11):
    tab = dict(sc.TABLE_I); tab["length_km"] = L
    lc = sc.line_constants(tab); md = sc.modal_transform(lc, tab)
    m = np.zeros((1, 6), bool); m[0, list(mask)] = True
    spec = dict(d_km=np.array([d_km]), Rf_ohm=np.array([Rf]),
                grounded=np.array([grounded]), mask=m,
                open_mask=np.zeros((1, 6), bool), S_sc=np.array([3500.0]),
                XR=np.array([15.0]), Zn_ohm=np.array([0.0]),
                load_pu=np.array([load]), inception_deg=np.array([incep]),
                ibr_pen=np.array([0.0]), ibr_mode=np.array([0]),
                remote_source=np.array([False]))
    b = spemt.Batch(spec, dt=dt, t_end=T_END, t_fault=T_FAULT, table=tab,
                    lc=lc, md=md, arc=arc, seed=seed)
    r = b.run(decim=1, relay_at=relay_at)
    v = r["v"].astype(float); i = r["i"].astype(float); dtr = r["dt"]
    pref = slice(int(0.5 * T_FAULT / dtr), int(0.95 * T_FAULT / dtr))
    rng = np.random.default_rng(seed)
    isec = spmeas.ct_secondary(i, dtr, np.array([ct_burd]), np.array([ct_rem]))
    im = spmeas.add_noise(isec, np.array([snr]), rng, ref_slice=pref)
    vm = spmeas.add_noise(spmeas.cvt_transfer(v, dtr), np.array([snr]), rng,
                          ref_slice=pref)
    vf, _ = spmeas.antialias_decimate(vm, dtr, FC, FS)
    iff, _ = spmeas.antialias_decimate(im, dtr, FC, FS)
    R = sprelay.Relay()
    o = R.process(vf, iff, 1.0 / FS, T_FAULT)
    loc = R.locate(o, int(T_FAULT * FS) + 200, m, np.array([grounded]))
    return dict(name=name, v=vf[:, 0, :], i=iff[:, 0, :], fs=FS,
                Psi=o["Psi"][:, 0], tau=o["tau"][:, 0], Gam=o["Gam"][:, 0],
                Is=o["Is"][:, 0, :], Vs=o["Vs"][:, 0, :], D=o["D"][:, 0],
                i_fault=r["i_fault"][:, 0], dt_emt=dtr,
                d_true=d_km, d_hat=float(loc["d_hat"][0]),
                r_d=float(loc["r_d"][0]), Rf=Rf, external=external)


def main():
    recs = [
        one("internal_bolted_ag", 50.0, [0], True, 0.0),
        one("internal_hif_ag", 50.0, [0], True, 0.0, arc=True),
        one("external_forward", 110.0, [0], True, 0.0, L=130.0, external=True),
        one("internal_bolted_abcg", 30.0, [0, 1, 2], True, 0.0),
        one("internal_ad_ungrounded", 70.0, [0, 3], False, 5.0),
        one("internal_hr_ag_300", 50.0, [0], True, 300.0),
        one("reverse_close_in", 40.0, [0], True, 0.0, relay_at="F2", external=True),
    ]
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "waveforms.npz"),
                        **{f"{r['name']}__{k}": np.asarray(v)
                           for r in recs for k, v in r.items() if k != "name"})
    meta = [{k: (v if np.ndim(v) == 0 else None) for k, v in r.items()} for r in recs]
    json.dump([{k: (None if isinstance(v, np.ndarray) else v) for k, v in m.items()}
               for m in meta], open(os.path.join(OUT, "waveforms_meta.json"), "w"),
              indent=1, default=str)
    for r in recs:
        print(f"  {r['name']:24s} d={r['d_true']:5.1f} d_hat={r['d_hat']:6.1f} "
              f"r_d={r['r_d']:.3f} trip={'yes' if r['D'][int(T_FAULT*FS):].any() else 'no'} "
              f"i_f_pk={np.abs(r['i_fault'][int(T_FAULT/r['dt_emt']):]).max():8.1f} A")


if __name__ == "__main__":
    main()
