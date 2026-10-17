#!/usr/bin/env python3
"""
spconst.py -- six-phase line constants, modal transform and network impedances.

Everything here is a deterministic function of the parameters declared in
Table I of the manuscript.  Nothing is measured.

Line constants are taken from sixphase_core.py (already verified against the
manuscript to full printed precision).  This module adds:

  * a real constant-parameter modal transformation (Wedepohl eigenvector
    method, real-rotated), with a quantified decoupling residual;
  * modal surge impedance, travel time and series resistance per section;
  * the six-phase Thevenin equivalent of the local source + phase-conversion
    transformer, and of the remote equivalent;
  * the constant-impedance load.
"""
import numpy as np
import sixphase_core as core

TWO_PI = 2.0 * np.pi

# ---------------------------------------------------------------- Table I ---
TABLE_I = dict(
    f0            = 50.0,        # Hz
    n_phase       = 6,
    V_phg_kV      = 93.0,        # conductor-to-ground rms, kV
    V_src_kV      = 115.0,       # three-phase source nominal, kV
    S_base_MVA    = 100.0,
    S_sc_MVA      = 3500.0,      # local source short-circuit level (base case)
    XR_local      = 15.0,
    S_scR_MVA     = 1500.0,      # remote equivalent short-circuit level  [ADDED]
    XR_remote     = 15.0,        #                                        [ADDED]
    Zn_ohm        = 0.0,         # source neutral earthing (0 = solid)
    Zt_pct        = 10.0,        # transformer leakage, % on S_T
    S_T_MVA       = 100.0,       # per bank
    n_banks       = 2,           # banks per terminal (each feeds 3 conductors)
    Rw_pu         = 0.004,       # winding resistance, pu on S_T
    Rm_pu         = 500.0,       # magnetising branch
    Xm_pu         = 300.0,
    length_km     = 100.0,
    load_pf       = 0.95,        # lagging                                [ADDED]
    dt_s          = 10.0e-6,     # base integration step                  [AMENDED 20->10]
    Tsim_s        = 0.400,
    t_fault_s     = 0.100,
    fs_Hz         = 5000.0,
    fc_Hz         = 1800.0,
    CT_ratio      = 200.0,       # 200 : 1
    CT_knee_V     = 150.0,
    CT_burden_ohm = 0.5,
    VT_ratio      = 845.0,
    n_bits        = 16,
    master_seed   = 20260829,
)

# six-phase bases (power-invariant, per conductor)
def bases(t=TABLE_I):
    V = t["V_phg_kV"] * 1e3                       # V, rms phase-to-ground
    S_per_phase = t["S_base_MVA"] * 1e6 / t["n_phase"]
    I = S_per_phase / V                            # A rms
    Z = V / I
    return dict(V_base=V, I_base=I, Z_base=Z, S_phase=S_per_phase)


# --------------------------------------------------------- line constants ---
def line_constants(t=TABLE_I):
    """R [ohm/km], L [H/km], C [F/km], G [S/km], Z [ohm/km], Y [S/km]."""
    cfg = dict(core.CONFIG); cfg["f"] = t["f0"]; cfg["length_km"] = t["length_km"]
    Z_km, R, L_mH = core.series_impedance(cfg)
    C_nF, _ = core.shunt_capacitance(cfg)
    L = L_mH * 1e-3
    C = C_nF * 1e-9
    G = np.zeros((6, 6))
    w = TWO_PI * t["f0"]
    Y_km = G + 1j * w * C
    return dict(R=R, L=L, C=C, G=G, Z=Z_km, Y=Y_km, w=w)


def modal_transform(lc, t=TABLE_I):
    """
    Real constant-parameter modal transformation (Wedepohl eigenvector method).

    Tv  : voltage transformation, v_phase = Tv @ v_modal
    Ti  : current transformation, i_phase = Ti @ i_modal,  Ti = inv(Tv).T
    Returns modal r [ohm/km], l [H/km], c [F/km] and the decoupling residual.
    """
    Z, Y, w = lc["Z"], lc["Y"], lc["w"]
    ev, V = np.linalg.eig(Z @ Y)
    # rotate each eigenvector so that its largest-magnitude entry is real +ve,
    # then take the real part -> real transformation matrix
    Tv = np.empty((6, 6))
    for k in range(6):
        v = V[:, k]
        j = int(np.argmax(np.abs(v)))
        v = v * np.exp(-1j * np.angle(v[j]))
        v = v.real
        Tv[:, k] = v / np.linalg.norm(v)
    Ti = np.linalg.inv(Tv).T
    Tv_inv = np.linalg.inv(Tv)
    Ti_inv = np.linalg.inv(Ti)
    Zm = Tv_inv @ lc["Z"] @ Ti
    Ym = Ti_inv @ lc["Y"] @ Tv
    r = np.real(np.diag(Zm))
    l = np.imag(np.diag(Zm)) / w
    c = np.imag(np.diag(Ym)) / w
    off_Z = np.abs(Zm - np.diag(np.diag(Zm))).max() / np.abs(np.diag(Zm)).min()
    off_Y = np.abs(Ym - np.diag(np.diag(Ym))).max() / np.abs(np.diag(Ym)).min()
    order = np.argsort(np.sqrt(l * c))                 # fastest mode first
    return dict(Tv=Tv, Ti=Ti, Tv_inv=Tv_inv, Ti_inv=Ti_inv,
                r=r, l=l, c=c, resid_Z=off_Z, resid_Y=off_Y,
                Zc=np.sqrt(l / c),
                vel_km_s=1.0 / np.sqrt(l * c),      # km/s
                tau_per_km=np.sqrt(l * c),          # s/km
                order=order)


# ------------------------------------------------------ network impedances ---
def S6():
    """Power-invariant six-phase Fortescue matrix, Eq. (4).

    S6[i, j] = a**(i*j) / sqrt(6)  with  a = exp(+j 2*pi/6).

    HANDEDNESS.  The conjugate form a = exp(-j 2*pi/6) is equally unitary and
    power-invariant and gives the identical sequence impedances, because the
    conjugate pairs Z1 = Z5 and Z2 = Z4 coincide, so no quantity quoted in
    Section 2.3 distinguishes the two.  They are NOT interchangeable for
    currents: under the conjugate form a balanced set with the conventional
    60-degree progressive lag maps to index 5, not index 1, which reverses the
    index convention declared in Section 2.3 ("k = 1 the positive sequence;
    k = 5 the negative sequence") and makes Eq. (18) read the whole
    positive-sequence load current as an unbalance quantity.  An earlier
    revision of sixphase_core.py used the conjugate form by mistake; the error
    was invisible in everything that module printed.  validate.py checks the
    convention explicitly against S6_conjugate() below.
    """
    a = np.exp(2j * np.pi / 6.0)
    return np.array([[a ** (i * j) for j in range(6)] for i in range(6)]) / np.sqrt(6.0)


def S6_conjugate():
    """The conjugate form a = exp(-j 2*pi/6), kept only so that the index
    convention can be checked against it.  It is unitary, power-invariant and
    gives the identical sequence impedances, and it reverses the sequence
    order; an earlier revision of sixphase_core.py used it by mistake."""
    a = np.exp(-2j * np.pi / 6.0)
    return np.array([[a ** (i * j) for j in range(6)] for i in range(6)]) / np.sqrt(6.0)


def seq_to_phase(diag6):
    """Build the phase-domain matrix of a balanced element from its six
    power-invariant sequence impedances."""
    S = S6()
    return np.conj(S).T @ np.diag(diag6) @ S


def source_impedance(S_sc_MVA, XR, Zt_pct, S_T_MVA, Rw_pu, Zn_ohm, t=TABLE_I,
                     include_transformer=True):
    """
    Six-phase Thevenin impedance matrix [ohm] of a three-phase source behind a
    delta / grounded-wye phase-conversion transformer bank.

    Non-zero six-phase sequences see  Z_source + Z_transformer.
    The zero sequence sees the transformer only (the delta primary blocks it),
    plus 6 Zn from the common earthed neutral.
    """
    b = bases(t)
    Zb = b["Z_base"]
    # source, referred to the six-phase side, on the 100 MVA six-phase base
    z_src_pu = t["S_base_MVA"] / S_sc_MVA
    ang = np.arctan(XR)
    Z_src = z_src_pu * Zb * (np.cos(ang) + 1j * np.sin(ang))
    if include_transformer:
        # each bank feeds three of the six conductors, so on the six-phase
        # per-conductor base the leakage is Zlk * S_base / (n_banks * S_T)
        k = t["S_base_MVA"] / (t["n_banks"] * S_T_MVA)
        z_t_pu = (Zt_pct / 100.0) * k
        Z_t = Zb * (Rw_pu * k + 1j * z_t_pu)
    else:
        Z_t = 0.0
    Z_pos = Z_src + Z_t
    Z_zero = Z_t + 6.0 * Zn_ohm if np.isfinite(Zn_ohm) else 1e9 + 0j
    if not np.isfinite(Zn_ohm):
        Z_zero = 1e9 + 0j
    return seq_to_phase(np.array([Z_zero] + [Z_pos] * 5, dtype=complex))


def load_impedance(level_pu, t=TABLE_I, pf=None):
    """Six-phase impedance of the remote terminal: a constant-impedance load
    behind the remote phase-conversion transformer bank (Yg on the six-phase
    side, delta on the three-phase side).  Non-zero sequences see the
    transformer leakage in series with the load; the zero sequence sees the
    transformer alone, the delta providing the circulating path."""
    b = bases(t)
    Zb = b["Z_base"]
    k = t["S_base_MVA"] / (t["n_banks"] * t["S_T_MVA"])
    Z_t = Zb * (t["Rw_pu"] * k + 1j * (t["Zt_pct"] / 100.0) * k)
    if level_pu <= 0:
        Z_ld = 1e9 + 0j
    else:
        pf = t["load_pf"] if pf is None else pf
        S_ph = level_pu * t["S_base_MVA"] * 1e6 / t["n_phase"]
        ang = np.arccos(pf)
        Z_ld = (b["V_base"] ** 2 / S_ph) * (np.cos(ang) + 1j * np.sin(ang))
    return seq_to_phase(np.array([Z_t] + [Z_t + Z_ld] * 5, dtype=complex))


def source_emf(t=TABLE_I):
    """Peak conductor e.m.f. and the six phase angles [rad] of the ideal source."""
    Vpk = np.sqrt(2.0) * t["V_phg_kV"] * 1e3
    ang = -TWO_PI / 6.0 * np.arange(6)
    return Vpk, ang


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True, linewidth=170)
    t = TABLE_I
    b = bases(t)
    print(f"V_base {b['V_base']:.1f} V   I_base {b['I_base']:.2f} A   Z_base {b['Z_base']:.2f} ohm")
    lc = line_constants()
    md = modal_transform(lc)
    print("\nmodal r [ohm/km]:", md["r"])
    print("modal l [mH/km] :", md["l"] * 1e3)
    print("modal c [nF/km] :", md["c"] * 1e9)
    print("modal Zc [ohm]  :", md["Zc"])
    print("modal v/c       :", md["vel_km_s"] * 1e3 / 2.998e8)
    print("tau over 100 km [us]:", md["tau_per_km"] * 100 * 1e6)
    print(f"\ndecoupling residual  Z {md['resid_Z']:.3e}   Y {md['resid_Y']:.3e}")
    Zs = source_impedance(t["S_sc_MVA"], t["XR_local"], t["Zt_pct"], t["S_T_MVA"],
                          t["Rw_pu"], t["Zn_ohm"])
    print("\nsix-phase source+transformer Thevenin, sequence diagonal [ohm]:")
    S = S6(); print(np.diag(S @ Zs @ np.conj(S).T))
