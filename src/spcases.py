#!/usr/bin/env python3
"""
spcases.py -- the frozen case set.

The 178-class taxonomy is read from taxonomy_full.csv (frozen, unchanged).
Scenario parameters are drawn from the Table X levels by a scrambled Sobol'
sequence under the master seed of Table I, so the whole case set is
reproducible from this file alone.
"""
import numpy as np, csv, re
from scipy.stats import qmc
import spconst as sc

LEVELS = dict(
    Rf      = [0.0, 5.0, 25.0, 100.0, 300.0, 1000.0],
    loc     = [5., 15., 25., 35., 45., 55., 65., 75., 85., 95.],
    incep   = list(np.arange(0, 360, 30, dtype=float)),
    ct_rem  = [0.0, 0.20, 0.50, 0.80],
    ct_burd = [0.5, 1.0],
    snr     = [20.0, 30.0, 40.0, 60.0],
    par_err = [0.0, .05, -.05, .10, -.10, .20, -.20],
    load    = [0.3, 0.7, 1.0],
    ldtype  = [0, 1],
    ibr_pen = [0.0, .20, .40, .60, .80],
    ibr_mod = [0, 1],
    evo_del = [0.5, 1.0, 2.0],
    ssc     = [500., 1500., 3500., 8000.],
    xr      = [5., 15., 30.],
    zn      = [0.0, 20.0, np.inf],
)
DIMS = ["Rf", "loc", "incep", "ct_rem", "ct_burd", "snr", "par_err",
        "load", "ldtype", "evo_del", "ssc", "xr", "zn"]


def load_taxonomy(path="taxonomy_full.csv"):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        cl = [int(c) for c in r["conductors"].strip() if c.isdigit()]
        out.append(dict(case_id=r["case_id"], group=r["group"], k=r["k"],
                        conductors=cl, grounded=r["grounded"].strip() == "1",
                        topology=r["topology"], position=r["position"],
                        detail=r["detail"]))
    return out


POS_KM = {"near": 10.0, "mid": 50.0, "remote": 90.0}


def parse_class(cls):
    """Return (mask, grounded, open_mask, mask2, grounded2, fixed_position_km)."""
    g = cls["group"]
    conds = cls["conductors"]
    m = np.zeros(6, bool); om = np.zeros(6, bool)
    m2 = None; g2 = None
    gr = cls["grounded"]
    pos = None
    if g == "A_shunt":
        m[conds] = True
    elif g == "B_open":
        om[conds] = True
        gr = False
        pos = POS_KM[cls["position"].split("(")[0]]
    elif g == "C_open_shunt":
        om[conds] = True
        t = re.search(r"LG on c(\d)", cls["detail"])
        if t:
            m[int(t.group(1))] = True
        gr = True
        pos = POS_KM[cls["position"].split("(")[0]]
    elif g == "D_evolving":
        mm = re.findall(r"\((\d)C,(ung|G)\)", cls["detail"])
        (k1, s1), (k2, s2) = mm[0], mm[1]
        m[: int(k1)] = True
        gr = (s1 == "G")
        m2 = np.zeros(6, bool); m2[: int(k2)] = True
        g2 = (s2 == "G")
        pos = POS_KM[cls["position"].split("(")[0]]
    return m, gr, om, m2, g2, pos


def sobol_levels(n, seed, dims=DIMS):
    s = qmc.Sobol(d=len(dims), scramble=True, seed=seed)
    u = s.random(n)
    out = {}
    for j, k in enumerate(dims):
        lv = LEVELS[k]
        out[k] = np.array(lv, dtype=object)[np.minimum((u[:, j] * len(lv)).astype(int),
                                                       len(lv) - 1)]
    return out


def mask_from(conds):
    m = np.zeros(6, bool)
    for c in conds:
        m[c % 6] = True
    return m


def build_main(tax, N, seed, table=None):
    """N scenarios for each of the 178 frozen classes."""
    recs = []
    for ci, cls in enumerate(tax):
        m, gr, om, m2, g2, pos = parse_class(cls)
        lv = sobol_levels(N, seed + ci)
        for j in range(N):
            d = pos if pos is not None else float(lv["loc"][j])
            recs.append(dict(case_id=cls["case_id"], group=cls["group"], idx=j,
                             mask=m.copy(), grounded=bool(gr), open_mask=om.copy(),
                             mask2=None if m2 is None else m2.copy(),
                             grounded2=g2, d_km=float(d),
                             Rf_ohm=float(lv["Rf"][j]),
                             inception_deg=float(lv["incep"][j]),
                             ct_rem=float(lv["ct_rem"][j]), ct_burd=float(lv["ct_burd"][j]),
                             snr=float(lv["snr"][j]), par_err=float(lv["par_err"][j]),
                             load_pu=float(lv["load"][j]), ldtype=int(lv["ldtype"][j]),
                             evo_del=float(lv["evo_del"][j]), S_sc=float(lv["ssc"][j]),
                             XR=float(lv["xr"][j]), Zn_ohm=float(lv["zn"][j]),
                             ibr_pen=0.0, ibr_mode=0, external=0, kind="internal"))
    return recs


def build_external(N, seed, table=None):
    """External, reverse and no-fault cases for the security evaluation."""
    t = table or sc.TABLE_I
    recs = []
    lv = sobol_levels(3 * N, seed)
    for j in range(N):                                  # forward external
        recs.append(_ext(lv, j, kind="forward_external",
                         d=float([102., 106., 110., 114.][j % 4])))
    for j in range(N, 2 * N):                           # reverse
        recs.append(_ext(lv, j, kind="reverse", d=float(lv["loc"][j])))
    for j in range(2 * N, 3 * N):                       # no-fault disturbances
        sub = ["quiet", "load_switch", "power_swing"][j % 3]
        r = _ext(lv, j, kind="nofault", d=50.0)
        r["mask"] = np.zeros(6, bool); r["grounded"] = False
        r["subkind"] = sub
        if sub == "load_switch":
            r["load_pu2"] = float([0.3, 0.7, 1.0][(j // 3) % 3])
        elif sub == "power_swing":
            r["slip_hz"] = float([0.5, 1.0, 2.0][(j // 3) % 3])
            r["ibr_pen"] = 0.0
            r["remote_source"] = True
        recs.append(r)
    return recs


def _ext(lv, j, kind, d):
    m = np.zeros(6, bool)
    if kind != "nofault":
        rng = np.random.default_rng(1000 + j)
        k = rng.integers(1, 4)
        m[rng.choice(6, size=k, replace=False)] = True
    return dict(case_id=kind.upper(), group="E_" + kind, idx=j, mask=m,
                grounded=bool(j % 2), open_mask=np.zeros(6, bool), mask2=None,
                grounded2=None, d_km=d, Rf_ohm=float(lv["Rf"][j]),
                inception_deg=float(lv["incep"][j]), ct_rem=float(lv["ct_rem"][j]),
                ct_burd=float(lv["ct_burd"][j]), snr=float(lv["snr"][j]),
                par_err=float(lv["par_err"][j]), load_pu=float(lv["load"][j]),
                ldtype=int(lv["ldtype"][j]), evo_del=float(lv["evo_del"][j]),
                S_sc=float(lv["ssc"][j]), XR=float(lv["xr"][j]),
                Zn_ohm=float(lv["zn"][j]), ibr_pen=0.0, ibr_mode=0,
                external=1, kind=kind, subkind="", load_pu2=None, slip_hz=0.0,
                remote_source=False)


def build_hif(tax, N, seed):
    """High-impedance-fault set: single- and two-conductor ground classes with
    the arc model of (6) in place of a fixed fault resistance."""
    sel = [c for c in tax if c["group"].startswith("A") and c["grounded"]
           and len(c["conductors"]) <= 2]
    recs = []
    for ci, cls in enumerate(sel):
        lv = sobol_levels(N, seed + 500 + ci)
        for j in range(N):
            mm = np.zeros(6, bool); mm[cls["conductors"]] = True
            recs.append(dict(case_id=cls["case_id"], group="HIF", idx=j,
                             mask=mm, grounded=True,
                             open_mask=np.zeros(6, bool), mask2=None, grounded2=None,
                             d_km=float(lv["loc"][j]), Rf_ohm=0.0,
                             inception_deg=float(lv["incep"][j]),
                             ct_rem=float(lv["ct_rem"][j]), ct_burd=float(lv["ct_burd"][j]),
                             snr=float(lv["snr"][j]), par_err=float(lv["par_err"][j]),
                             load_pu=float(lv["load"][j]), ldtype=int(lv["ldtype"][j]),
                             evo_del=1.0, S_sc=float(lv["ssc"][j]), XR=float(lv["xr"][j]),
                             Zn_ohm=float(lv["zn"][j]), ibr_pen=0.0, ibr_mode=0,
                             external=0, kind="hif"))
    return recs


def build_ibr(tax, N, seed):
    """IBR penetration and control-mode sweep over a representative subset."""
    sel = [c for c in tax if c["group"].startswith("A") and len(c["conductors"]) <= 3][:24]
    recs = []
    for ci, cls in enumerate(sel):
        lv = sobol_levels(N, seed + 900 + ci)
        for j in range(N):
            mm = np.zeros(6, bool); mm[cls["conductors"]] = True
            recs.append(dict(case_id=cls["case_id"], group="IBR", idx=j,
                             mask=mm, grounded=bool(cls["grounded"]),
                             open_mask=np.zeros(6, bool), mask2=None, grounded2=None,
                             d_km=float(lv["loc"][j]), Rf_ohm=float(lv["Rf"][j]),
                             inception_deg=float(lv["incep"][j]),
                             ct_rem=float(lv["ct_rem"][j]), ct_burd=float(lv["ct_burd"][j]),
                             snr=float(lv["snr"][j]), par_err=float(lv["par_err"][j]),
                             load_pu=float(lv["load"][j]), ldtype=int(lv["ldtype"][j]),
                             evo_del=1.0, S_sc=float(lv["ssc"][j]), XR=float(lv["xr"][j]),
                             Zn_ohm=0.0,
                             ibr_pen=float(LEVELS["ibr_pen"][j % 5]),
                             ibr_mode=int(LEVELS["ibr_mod"][(j // 5) % 2]),
                             external=0, kind="ibr"))
    return recs
