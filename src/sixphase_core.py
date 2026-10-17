#!/usr/bin/env python3
"""
sixphase_core.py -- authoritative line-constant and transform computations
=========================================================================
Reproduces Tables I, II and III of

  "Adaptive Sequence-Modal Protection of Six-Phase Transmission Lines:
   Formulation, Fault Taxonomy, and a Reproducible Validation Benchmark"

directly from the conductor geometry declared in Table I.  Nothing in this
module is measured or simulated: every quantity is a deterministic function
of the declared geometry, the declared earth resistivity, and 50 Hz.

Formulation (as declared in Table I of the manuscript)
------------------------------------------------------
Series impedance -- Deri complex-depth earth return:

    p        = sqrt(rho_e / (j*omega*mu0))                       [m]
    Z_ii     = R_dc + j*omega*mu0/(2*pi) * ln((h_i + h_i + 2p)/GMR_i)
    Z_ik     =        j*omega*mu0/(2*pi) * ln(D'_ik / d_ik)
    D'_ik    = sqrt((x_i-x_k)^2 + (h_i + h_k + 2p)^2)
    d_ik     = sqrt((x_i-x_k)^2 + (h_i - h_k)^2)

Shunt capacitance -- Maxwell potential coefficients, perfect-ground images:

    P_ii = ln(2 h_i / r_i) / (2 pi eps0)
    P_ik = ln(D0_ik / d_ik) / (2 pi eps0),  D0_ik with the real image at -h_k
    C    = inv(P)

Power-invariant six-phase Fortescue transform:

    S6[i,k] = a^(i k) / sqrt(6),   a = exp(+j 2 pi / 6)

CORRECTED 3 Sep 2026.  The previous revision of this file used a = exp(-j 2 pi/6),
the complex conjugate.  Both forms are unitary and power-invariant and both give
the identical sequence impedances, because the conjugate pairs Z1 = Z5 and
Z2 = Z4 coincide, so nothing this module printed distinguished them.  They are
NOT interchangeable for currents: under the conjugate form a balanced set with
the conventional 60-degree progressive lag maps to k = 5, which reverses the
index convention declared in Section 2.3 of the manuscript ("k = 1 the positive
sequence; k = 5 the negative sequence") and makes Eq. (18) read the entire
positive-sequence load current as an unbalance quantity.  The convention is now
checked explicitly in check_transform() below.

Verified against an independent reimplementation; see reproducibility/README.md.
"""

import numpy as np

MU0  = 4e-7 * np.pi
EPS0 = 8.8541878128e-12
C_LIGHT = 2.998e8            # m/s, as used in the manuscript

# ----------------------------------------------------------------------
# Declared configuration -- Table I
# ----------------------------------------------------------------------
CONFIG = dict(
    f          = 50.0,       # Hz
    rho_e      = 100.0,      # Ohm.m
    length_km  = 100.0,      # km
    n_cond     = 6,
    spacing_m  = 1.0,        # horizontal spacing between adjacent conductors
    height_m   = 10.0,       # mean height above ground
    r_cond     = 0.015,      # outer radius, m
    gmr_factor = 0.7788,     # GMR = gmr_factor * r  (solid round conductor)
    r_dc       = 0.0500,     # Ohm/km at 20 C, skin effect neglected
    tan_delta  = 0.0,        # -> G = 0
)


def geometry(cfg=CONFIG):
    """Return (x, h) conductor coordinates in metres for the flat array."""
    n = cfg["n_cond"]
    s = cfg["spacing_m"]
    x = (np.arange(n) - (n - 1) / 2.0) * s
    h = np.full(n, cfg["height_m"])
    return x, h


def complex_depth(cfg=CONFIG):
    """Deri complex penetration depth p = sqrt(rho / (j w mu0))  [m]."""
    w = 2 * np.pi * cfg["f"]
    return np.sqrt(cfg["rho_e"] / (1j * w * MU0))


def series_impedance(cfg=CONFIG):
    """Return Z [Ohm/km], R [Ohm/km], L [mH/km]."""
    x, h = geometry(cfg)
    n = cfg["n_cond"]
    w = 2 * np.pi * cfg["f"]
    p = complex_depth(cfg)
    gmr = cfg["gmr_factor"] * cfg["r_cond"]
    k = 1j * w * MU0 / (2 * np.pi) * 1000.0          # Ohm/km per ln unit

    Z = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j:
                Z[i, j] = cfg["r_dc"] + k * np.log((2 * h[i] + 2 * p) / gmr)
            else:
                dx = x[i] - x[j]
                d = np.hypot(dx, h[i] - h[j])
                D = np.sqrt(dx ** 2 + (h[i] + h[j] + 2 * p) ** 2)
                Z[i, j] = k * np.log(D / d)
    R = Z.real
    L = Z.imag / w * 1000.0                          # mH/km
    return Z, R, L


def shunt_capacitance(cfg=CONFIG):
    """Return C [nF/km] and the potential-coefficient matrix P."""
    x, h = geometry(cfg)
    n = cfg["n_cond"]
    c = 1.0 / (2 * np.pi * EPS0)
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = c * np.log(2 * h[i] / cfg["r_cond"])
            else:
                dx = x[i] - x[j]
                d = np.hypot(dx, h[i] - h[j])
                D = np.hypot(dx, h[i] + h[j])
                P[i, j] = c * np.log(D / d)
    C = np.linalg.inv(P) * 1e9 * 1000.0              # nF/km
    return C, P


def fortescue6():
    """Power-invariant six-phase Fortescue matrix S6, Eq. (4).

    S6[i, j] = a**(i*j) / sqrt(6)  with  a = exp(+j 2*pi/6), so that a balanced
    set with the conventional 60-degree progressive lag maps to k = 1, as the
    index convention of Section 2.3 requires.  See the module docstring: the
    conjugate form passes every check on impedances and fails on currents.
    """
    a = np.exp(2j * np.pi / 6)
    return np.array([[a ** (i * j) for j in range(6)] for i in range(6)]) / np.sqrt(6)


def check_transform():
    """Unitarity, power invariance and the declared index convention.

    Returns (unitarity_error, power_error, index_of_positive_sequence).  The
    third must be 1; it is the check that the conjugate form fails.
    """
    S = fortescue6()
    uni = float(np.abs(S @ np.conj(S).T - np.eye(6)).max())
    x = np.array([np.exp(-1j * np.pi / 3.0) ** k for k in range(6)])
    y = S @ x
    pw = abs(float((np.abs(x) ** 2).sum() - (np.abs(y) ** 2).sum()))
    return uni, pw, int(np.argmax(np.abs(y)))


def sequence_impedance(Z_per_km, cfg=CONFIG):
    """Sequence-domain impedance of the whole line, S6 (L*Z) S6^H  [Ohm]."""
    S = fortescue6()
    return S @ (cfg["length_km"] * Z_per_km) @ np.conj(S).T


def modal_propagation(Z_per_km, C_nF_per_km, cfg=CONFIG):
    """Return gamma [1/km], alpha, beta, velocity fraction v/c, travel time [us]."""
    w = 2 * np.pi * cfg["f"]
    Y = 1j * w * C_nF_per_km * 1e-9                  # S/km  (G = 0)
    ev = np.linalg.eigvals(Z_per_km @ Y)
    g = np.sqrt(ev)
    g = np.where(g.real < 0, -g, g)                  # principal branch, Re>0
    order = np.argsort(g.imag)                       # aerial modes first
    g = g[order]
    alpha, beta = g.real, g.imag
    v = w / (beta * 1e-3) / C_LIGHT                  # beta in 1/km -> 1/m
    tau = cfg["length_km"] * 1e3 / (w / (beta * 1e-3)) * 1e6   # us
    return g, alpha, beta, v, tau


def summary(cfg=CONFIG):
    """Everything the manuscript quotes, as a dictionary."""
    Z, R, L = series_impedance(cfg)
    C, P = shunt_capacitance(cfg)
    S = fortescue6()
    Zseq = sequence_impedance(Z, cfg)
    g, alpha, beta, v, tau = modal_propagation(Z, C, cfg)
    n = cfg["n_cond"]
    return dict(
        p=complex_depth(cfg), Z=Z, R=R, L=L, C=C, G=np.zeros((n, n)),
        S6=S, Zseq=Zseq, gamma=g, alpha=alpha, beta=beta, vfrac=v, tau_us=tau,
        unitarity=np.abs(S @ np.conj(S).T - np.eye(n)).max(),
        C_posdef=bool(np.all(np.linalg.eigvalsh(C) > 0)),
        cfg=cfg,
    )


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=160)
    s = summary()
    cfg = s["cfg"]
    print("=" * 74)
    print("SIX-PHASE LINE CONSTANTS FROM THE TABLE I GEOMETRY")
    print(f"  f = {cfg['f']} Hz   rho_e = {cfg['rho_e']} Ohm.m   length = {cfg['length_km']} km")
    print(f"  {cfg['n_cond']} conductors, flat, spacing {cfg['spacing_m']} m, height {cfg['height_m']} m")
    print(f"  r = {cfg['r_cond']} m,  GMR = {cfg['gmr_factor']}r = {cfg['gmr_factor']*cfg['r_cond']:.6f} m")
    print("=" * 74)
    p = s["p"]
    print(f"\nComplex depth p = {p.real:.1f} {p.imag:+.1f}j m,  |p| = {abs(p):.1f} m")

    R, L, C = s["R"], s["L"], s["C"]
    print(f"\nR [Ohm/km]: diagonal {R[0,0]:.6f}, off-diagonal mean {R[0,1:].mean():.6f}")
    print(f"   -> R = {R[0,0]-R[0,1]:.4f} I + {R[0,1]:.5f} J  (max residual "
          f"{np.abs(R - ((R[0,0]-R[0,1])*np.eye(6) + R[0,1]*np.ones((6,6)))).max():.2e})")
    print("\nL [mH/km] by conductor separation:")
    for k in range(6):
        print(f"   |i-j| = {k}:  {L[0, k] if k else L[0,0]:.6f}")
    print("\nC [nF/km] distinct entries:")
    for lbl, val in (("C11 = C66", C[0,0]), ("C22 = C55", C[1,1]), ("C33 = C44", C[2,2]),
                     ("C12 = C56", C[0,1]), ("C23 = C45", C[1,2]), ("C34", C[2,3]),
                     ("C13 = C46", C[0,2]), ("C24 = C35", C[1,3]), ("C14 = C36", C[0,3]),
                     ("C25", C[1,4]), ("C15 = C26", C[0,4]), ("C16", C[0,5])):
        print(f"   {lbl:<10} {val:+.4f}")
    print(f"\n   C positive definite: {s['C_posdef']}")
    print(f"   L Toeplitz residual: "
          f"{max(abs(L[i,j]-L[0,abs(i-j)]) for i in range(6) for j in range(6)):.2e}")
    print(f"   C Toeplitz residual: "
          f"{max(abs(C[i,j]-C[0,abs(i-j)]) for i in range(6) for j in range(6)):.4f} nF/km")

    print(f"\nS6 unitarity error max|S6 S6^H - I| = {s['unitarity']:.3e}")
    rng = np.random.default_rng(20260829)            # master seed, Table I
    xv = rng.normal(size=6) + 1j * rng.normal(size=6)
    e_ph = float(np.vdot(xv, xv).real)
    e_sq = float(np.vdot(s["S6"] @ xv, s["S6"] @ xv).real)
    print(f"Power invariance (master seed 20260829): {e_ph:.10f} phase vs "
          f"{e_sq:.10f} sequence, difference {abs(e_ph-e_sq):.3e}")

    Zs = s["Zseq"]
    print("\nSequence impedances of the 100 km line [Ohm]:")
    for k in range(6):
        print(f"   Z{k} = {Zs[k,k].real:8.3f} {Zs[k,k].imag:+9.3f}j")
    off = np.abs(Zs - np.diag(np.diag(Zs)))
    print(f"   max off-diagonal magnitude = {off.max():.3f} Ohm "
          f"({100*off.max()/min(abs(np.diag(Zs))):.1f}% of the smallest diagonal, "
          f"{20*np.log10(off.max()/abs(Zs).max()):.1f} dB re largest element)")
    print(f"   X0/X1 = {Zs[0,0].imag/Zs[1,1].imag:.3f}   R0/R1 = {Zs[0,0].real/Zs[1,1].real:.3f}")
    xs = sorted({round(Zs[k,k].imag, 6) for k in range(1, 6)})
    print(f"   non-zero-sequence reactance spread = {100*(xs[-1]-xs[0])/xs[0]:.1f}%")

    print("\nModal propagation constants (1e-4 /km), velocities and travel times:")
    for k in range(6):
        kind = "earth-return" if k == 5 else "aerial"
        print(f"   mode {k}: {s['alpha'][k]*1e4:6.3f} {s['beta'][k]*1e4:+8.3f}j   "
              f"v/c = {s['vfrac'][k]:.3f}   tau = {s['tau_us'][k]:6.1f} us   {kind}")
    a_aer = s["alpha"][:5]
    print(f"   aerial attenuation spread (max/min - 1) = {100*(a_aer.max()/a_aer.min()-1):.1f}%")
