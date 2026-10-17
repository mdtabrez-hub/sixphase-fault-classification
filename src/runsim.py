#!/usr/bin/env python3
"""
runsim.py -- driver: simulate the frozen case set, apply the measurement chain,
run the relay, and record one row per case.

Usage:  python3 runsim.py <N_main> <N_ext> <N_hif> <N_ibr> <outdir> [batch]
"""
import sys, os, time, json, hashlib
import numpy as np
import spconst as sc, spemt, spmeas, sprelay, spcases

T = sc.TABLE_I
T_FAULT = 0.10
T_END = 0.24
FS = 5000.0
FC = 1800.0
SEED = T["master_seed"]
WIN_OFF = [50, 100, 150]          # samples after inception at which phi_w is stored


def dt_for(d, L):
    """Integration step: the Bergeron history buffer requires the shortest modal
    travel time of every section to exceed one step."""
    s = min(d, L - d)
    return 5e-6 if s < 7.5 else (10e-6 if s < 15.0 else 20e-6)


def group_key(r):
    L = 130.0 if r["kind"] == "forward_external" else 100.0
    return (dt_for(r["d_km"], L), r["kind"] in ("hif",), r["kind"] in ("ibr",),
            "F2" if r["kind"] == "reverse" else "S", L,
            r["mask2"] is not None, r.get("subkind", ""))


def make_spec(rows, L):
    n = len(rows)
    g = lambda k: np.array([r[k] for r in rows])
    spec = dict(d_km=g("d_km"), Rf_ohm=g("Rf_ohm"), grounded=g("grounded").astype(bool),
                mask=np.stack([r["mask"] for r in rows]),
                open_mask=np.stack([r["open_mask"] for r in rows]),
                S_sc=g("S_sc"), XR=g("XR"), Zn_ohm=g("Zn_ohm").astype(float),
                load_pu=g("load_pu"), inception_deg=g("inception_deg"),
                ibr_pen=g("ibr_pen"), ibr_mode=g("ibr_mode").astype(int),
                remote_source=(g("ibr_pen") > 0),
                harm_A=np.where(g("ldtype") == 1, 0.20 * g("load_pu") * 179.21 * np.sqrt(2), 0.0))
    if "load_pu2" in rows[0]:
        lp2 = np.array([r["load_pu"] if r.get("load_pu2") is None else r["load_pu2"]
                        for r in rows], float)
        spec["load_pu2"] = lp2
        spec["slip_hz"] = np.array([r.get("slip_hz", 0.0) for r in rows], float)
        spec["remote_source"] = np.array([bool(r.get("remote_source", False)) or
                                          r["ibr_pen"] > 0 for r in rows], bool)
    if rows[0]["mask2"] is not None:
        spec["mask2"] = np.stack([r["mask2"] for r in rows])
        spec["grounded2"] = g("grounded2").astype(bool)
        spec["open_mask2"] = np.stack([r["open_mask"] for r in rows])
        spec["t_evolve"] = T_FAULT + g("evo_del") / T["f0"]
    return spec


def chunk_seed(tag):
    """Deterministic per-chunk seed.

    The measurement-noise and arc streams must not depend on how many chunks
    happened to run in one invocation: the driver is resumable, so a run that
    is interrupted and continued would otherwise draw different realisations
    from one that is not, and RUN.sh would not reproduce the published set.
    The seed is therefore derived from the master seed and the chunk tag alone.
    """
    h = hashlib.sha256(f"{SEED}:{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


def run_group(rows, key, relay, rng, log, tag="000"):
    dt, arc, ibr, relay_at, L, evolving, subkind = key
    tab = dict(T); tab["length_km"] = L
    lc = sc.line_constants(tab); md = sc.modal_transform(lc, tab)
    spec = make_spec(rows, L)
    b = spemt.Batch(spec, dt=dt, t_end=T_END, t_fault=T_FAULT, table=tab,
                    lc=lc, md=md, arc=arc, ibr=ibr, seed=chunk_seed(tag))
    r = b.run(decim=1, relay_at=relay_at)
    v = r["v"]; i = r["i"]                       # float32 from the solver
    ifault = r["i_fault"]
    dtr = r["dt"]
    del r
    burd = np.array([x["ct_burd"] for x in rows])
    rem = np.array([x["ct_rem"] for x in rows])
    snr = np.array([x["snr"] for x in rows])
    pref = slice(int(0.5 * T_FAULT / dtr), int(0.95 * T_FAULT / dtr))
    im = spmeas.add_noise(spmeas.ct_secondary(i, dtr, burd, rem), snr, rng,
                          ref_slice=pref).astype(np.float32)
    vm = spmeas.add_noise(spmeas.cvt_transfer(v, dtr), snr, rng,
                          ref_slice=pref).astype(np.float32)
    # third chain: a saturation-forcing current transformer.  The declared
    # 200:1 5P20 unit does not saturate anywhere in the Table X burden and
    # remanence sweep at the fault levels this system produces, so the same
    # unit is additionally exercised at ten times rated burden, which is the
    # condition under which it does saturate.  Declared as such.
    imsat = spmeas.add_noise(spmeas.ct_secondary(i, dtr, 10.0 * burd, rem),
                             snr, rng, ref_slice=pref).astype(np.float32)
    out = {}
    for tag, vx, ix in (("meas", vm, im), ("oracle", v, i), ("satct", vm, imsat)):
        vf, _ = spmeas.antialias_decimate(vx, dtr, FC, FS)
        iff, _ = spmeas.antialias_decimate(ix, dtr, FC, FS)
        out[tag] = relay.process(np.asarray(vf, np.float64),
                                 np.asarray(iff, np.float64), 1.0 / FS, T_FAULT)
        del vf, iff
    del vm, im, imsat, v, i
    import gc
    gc.collect()
    return out, rows, ifault, dtr


def summarise(o, rows, relay, tag, ifault=None, dtr=None):
    kf = int(round(T_FAULT * FS))
    # the phasors and the two-cycle memory are both valid only after
    # mem + N_w samples; pre-fault security statistics start there so that the
    # filter start-up transient is not reported as a relay quantity
    kv = int(o["mem"] + o["Nw"])
    B = len(rows)
    mask = np.stack([r["mask"] for r in rows])
    gnd = np.array([r["grounded"] for r in rows], bool)
    par = np.array([r["par_err"] for r in rows])
    D = o["D"][kf:]; G = o["Gam"][kf:]
    trip_i = np.array([np.argmax(D[:, j]) if D[:, j].any() else -1 for j in range(B)])
    gth_i = np.full(B, -1)
    # the location is reported two cycles after fault inception, so that the
    # estimation window lies wholly in the post-fault record (declared)
    idx_loc = np.full(B, min(kf + 200, o["Psi"].shape[0] - 1))
    # locate each case at its own reporting window
    d_hat = np.zeros(B); r_d = np.zeros(B); Rf_h = np.zeros(B); d_med = np.zeros(B)
    for k in np.unique(idx_loc):
        sel = idx_loc == k
        l = relay.locate(o, int(k), mask[sel], gnd[sel],
                         theta_scale=1.0 + par[sel], sel=sel)
        d_hat[sel] = l["d_hat"]; r_d[sel] = l["r_d"]
        Rf_h[sel] = l["Rf_hat"]; d_med[sel] = l["d_med"]
    gate = relay.gate(dict(d_med=d_med, r_d=r_d))
    kw = min(kf + 200, o["Psi"].shape[0] - 1)          # settled reporting window
    hold = slice(kf + 150, min(kf + 600, o["Psi"].shape[0]))
    # Gamma_HIF must exceed the threshold for m_HIF = 3 consecutive cycles.
    # gamma_sus is the largest value that was held for a whole 3-cycle window,
    # i.e. max_t min_{t-3T < s <= t} Gamma(s), so a single-window inception
    # spike cannot declare a HIF.
    nc3 = int(3 * FS / T["f0"])
    from numpy.lib.stride_tricks import sliding_window_view
    Gp = o["Gam"][kf:]
    gsus = (sliding_window_view(Gp, nc3, axis=0).min(-1).max(0)
            if Gp.shape[0] > nc3 else Gp.min(0))
    return dict(trip_delay_ms=np.where(trip_i >= 0, trip_i / FS * 1e3, np.nan),
                detected=trip_i >= 0,
                # Gamma_HIF must be sustained for m_HIF cycles (Sec. 4.1): the
                # reported value is the minimum over a three-cycle hold window,
                # so a single-window inception spike cannot declare a HIF.
                gamma_max=G.max(0), gamma_sus=gsus,
                psi_max=o["Psi"][kf:].max(0),
                gamma_pre=o["Gam"][kv:kf - 10].max(0), psi_pre=o["Psi"][kv:kf - 10].max(0),
                tau=o["tau"][kf - 20], d_hat=d_hat, r_d=r_d, Rf_hat=Rf_h,
                d_med=d_med, gate=gate,
                q_CT=o["q_CT"][kw], q_CT_max=o["q_CT"][hold].max(0),
                sig_eta=o["sig_eta"][kf - 20],
                kappa=o["kappa"][kw], thd=o["THD"][kw],
                eta_tr=o["eta_tr"][kw], psi_w=o["Psi"][kw])


def main():
    Nm, Ne, Nh, Ni = [int(x) for x in sys.argv[1:5]]
    outdir = sys.argv[5]; BS = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    budget = float(sys.argv[7]) if len(sys.argv) > 7 else 1e9   # seconds per call
    parts = os.path.join(outdir, "parts")
    os.makedirs(parts, exist_ok=True)
    tax = spcases.load_taxonomy("taxonomy_full.csv")
    rows = (spcases.build_main(tax, Nm, SEED) + spcases.build_external(Ne, SEED + 3) +
            spcases.build_hif(tax, Nh, SEED) + spcases.build_ibr(tax, Ni, SEED))
    print(f"total cases {len(rows)}", flush=True)
    relay = sprelay.Relay()
    rng = np.random.default_rng(SEED)
    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    recs = []; feats = []
    t0 = time.time(); done = 0
    import pandas as pd
    # keep (batch x steps) bounded so the measurement chain fits in memory
    BUDGET = BS * int(round(T_END / 20e-6))
    chunks = []
    for ci, (key, grp) in enumerate(sorted(groups.items(), key=lambda kv: (str(kv[0]),))):
        nst = int(round(T_END / key[0]))
        bsz = max(25, min(BS, BUDGET // nst))
        for s in range(0, len(grp), bsz):
            chunks.append((f"{ci:03d}_{s:06d}", key, grp[s:s + bsz]))
    todo = [c for c in chunks
            if not os.path.exists(os.path.join(parts, f"p{c[0]}.csv"))]
    print(f"chunks total {len(chunks)}, remaining {len(todo)}", flush=True)
    if not todo:
        return finalise(outdir, parts, chunks, Nm, Ne, Nh, Ni)
    for tag, key, chunk in todo:
        if time.time() - t0 > budget:
            print("BUDGET REACHED -- rerun to continue", flush=True)
            return
        rng = np.random.default_rng(chunk_seed(tag))
        o, ch, ifa, dtr = run_group(chunk, key, relay, rng, None, tag=tag)
        sm = summarise(o["meas"], ch, relay, "meas")
        so = summarise(o["oracle"], ch, relay, "oracle")
        ss = summarise(o["satct"], ch, relay, "satct")
        kf = int(round(T_FAULT * FS))
        fv = np.stack([relay.features(o["meas"], kf + w) for w in WIN_OFF], 1)
        crecs = []
        if True:
            for j, rw in enumerate(ch):
                d = {k: rw.get(k) for k in ("case_id", "group", "idx", "kind", "subkind", "d_km",
                                        "Rf_ohm", "inception_deg", "ct_rem", "ct_burd",
                                        "snr", "par_err", "load_pu", "ldtype",
                                        "S_sc", "XR", "Zn_ohm", "ibr_pen", "ibr_mode",
                                        "external", "evo_del")}
                d["Zn_ohm"] = float(d["Zn_ohm"])
                d["nc"] = int(rw["mask"].sum()); d["grounded"] = int(rw["grounded"])
                d["nopen"] = int(rw["open_mask"].sum())
                d["mask_bits"] = int("".join("1" if x else "0" for x in rw["mask"]), 2)
                d["i_fault_pk"] = float(np.abs(ifa[int(T_FAULT / dtr):, j]).max())
                for k2, v2 in sm.items():
                    d["m_" + k2] = float(v2[j]) if np.ndim(v2) else float(v2)
                for k2, v2 in so.items():
                    d["o_" + k2] = float(v2[j]) if np.ndim(v2) else float(v2)
                for k2, v2 in ss.items():
                    d["s_" + k2] = float(v2[j]) if np.ndim(v2) else float(v2)
                crecs.append(d)
        pd.DataFrame(crecs).to_csv(os.path.join(parts, f"p{tag}.csv"), index=False)
        np.save(os.path.join(parts, f"f{tag}.npy"), fv.astype(np.float32))
        done += len(chunk)
        el = time.time() - t0
        print(f"  chunk {tag}  {done} cases this call  {el/60:.1f} min", flush=True)
    return finalise(outdir, parts, chunks, Nm, Ne, Nh, Ni)


def finalise(outdir, parts, chunks, Nm, Ne, Nh, Ni):
    import pandas as pd
    tags = [c[0] for c in chunks]
    dfs = [pd.read_csv(os.path.join(parts, f"p{t}.csv")) for t in tags]
    fs_ = [np.load(os.path.join(parts, f"f{t}.npy")) for t in tags]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(os.path.join(outdir, "cases.csv"), index=False)
    np.save(os.path.join(outdir, "features.npy"), np.concatenate(fs_, 0))
    import scipy
    meta = dict(N_main=Nm, N_ext=Ne, N_hif=Nh, N_ibr=Ni, seed=SEED,
                t_fault=T_FAULT, t_end=T_END, fs=FS, fc=FC, win_off=WIN_OFF,
                n_cases=int(len(df)), n_chunks=len(tags),
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__, pandas=pd.__version__)
    json.dump(meta, open(os.path.join(outdir, "run_meta.json"), "w"), indent=1)
    print("FINALISED", len(df), "rows", flush=True)

if __name__ == "__main__":
    main()
