#!/usr/bin/env python3
"""
compute_rlcg.py
===============
Computes the 6x6 series impedance (R, L) and shunt admittance (C, G)
matrices for a six-phase transmission line from conductor geometry, using
Carson's earth-return correction and Maxwell's potential coefficients.

This script does NOT contain the authors' line data. It is the tool that
turns geometry into the matrices required by Table I of the manuscript.
Supply the geometry in the CONFIG block, run, and paste the output.

Method
------
Series impedance (Carson, first-order approximation valid for
D_ik * sqrt(f/rho_e) << 1, which holds for power-frequency spacings):

    Z_ii = (R_ac,i + dR) + j*omega*(mu0/2pi)*ln(D_e / GMR_i)      Ohm/km
    Z_ik =            dR  + j*omega*(mu0/2pi)*ln(D_e / d_ik)      Ohm/km

    dR  = omega*(mu0/8)        (Carson earth-resistance term, ~9.869e-4*f
                                Ohm/km)
    D_e = 658.87*sqrt(rho_e/f) (equivalent earth-return depth, m)

Shunt capacitance (Maxwell potential coefficients, with image charges):

    P_ii = (1/(2*pi*eps0))*ln(2*h_i / r_i)                        m/F
    P_ik = (1/(2*pi*eps0))*ln(D'_ik / d_ik)                       m/F
    C    = inv(P)                                                 F/m

where D'_ik is the distance from conductor i to the image of conductor k.

Bundled conductors are handled by replacing r and GMR with their
geometric-mean-bundle equivalents.

Verification built in
---------------------
  * symmetry of all matrices
  * positive-definiteness of C
  * comparison of the derived positive-sequence impedance against a
    user-supplied expected value, if given
"""

import numpy as np

# ======================================================================
# CONFIG  --  REPLACE EVERY VALUE BELOW WITH THE ACTUAL LINE DATA
# ======================================================================
CONFIG = dict(
    f=50.0,              # system frequency, Hz          [AUTHORS: confirm 50 or 60]
    rho_e=100.0,         # earth resistivity, Ohm-m      [AUTHORS: site value]
    length_km=100.0,     # line length, km

    # Conductor positions (x horizontal, y height above ground), metres.
    # Order MUST match the phase order [a, b, c, a', b', c'] = c0..c5 used
    # in the manuscript's Section 2.3 index convention.
    #                x       y
    coords=np.array([[-6.0, 20.0],     # c0
                     [-3.0, 24.0],     # c1
                     [ 3.0, 24.0],     # c2
                     [ 6.0, 20.0],     # c3
                     [ 3.0, 16.0],     # c4
                     [-3.0, 16.0]]),   # c5

    r_cond=0.0102,       # conductor outer radius, m
    gmr=0.00812,         # conductor geometric mean radius, m
    r_ac=0.0721,         # conductor a.c. resistance at f, Ohm/km
    n_bundle=1,          # sub-conductors per phase
    bundle_spacing=0.45, # m (ignored if n_bundle == 1)

    expected_z1=None,    # e.g. 0.03 + 0.30j Ohm/km, or None to skip check
)
# ======================================================================

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


def bundle_equivalent(r, gmr, n, s):
    """GMR and radius of an n-conductor bundle on a circle of spacing s."""
    if n == 1:
        return r, gmr
    R = s / (2 * np.sin(np.pi / n)) if n > 2 else s / 2
    gmr_b = (n * gmr * R ** (n - 1)) ** (1.0 / n)
    r_b = (n * r * R ** (n - 1)) ** (1.0 / n)
    return r_b, gmr_b


def distances(coords):
    """Physical distance matrix d_ik and image distance matrix D'_ik."""
    n = len(coords)
    d = np.zeros((n, n))
    D = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for k in range(n):
            xk, yk = coords[k]
            if i == k:
                D[i, k] = 2 * yi
            else:
                d[i, k] = np.hypot(xi - xk, yi - yk)
                D[i, k] = np.hypot(xi - xk, yi + yk)
    return d, D


def series_impedance(cfg, gmr_eq):
    f, rho_e = cfg["f"], cfg["rho_e"]
    coords = cfg["coords"]
    n = len(coords)
    w = 2 * np.pi * f

    d, _ = distances(coords)
    De = 658.87 * np.sqrt(rho_e / f)          # m
    dR = w * MU0 / 8 * 1000.0                 # Ohm/km  (earth resistance term)
    k = w * MU0 / (2 * np.pi) * 1000.0        # Ohm/km per ln unit

    Z = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            if i == j:
                Z[i, j] = (cfg["r_ac"] + dR) + 1j * k * np.log(De / gmr_eq)
            else:
                Z[i, j] = dR + 1j * k * np.log(De / d[i, j])
    R = Z.real
    L = Z.imag / w * 1000.0                   # mH/km
    return Z, R, L


def shunt_capacitance(cfg, r_eq):
    coords = cfg["coords"]
    n = len(coords)
    d, D = distances(coords)
    P = np.zeros((n, n))
    c = 1.0 / (2 * np.pi * EPS0)
    for i in range(n):
        for j in range(n):
            if i == j:
                P[i, j] = c * np.log(D[i, j] / r_eq)
            else:
                P[i, j] = c * np.log(D[i, j] / d[i, j])
    C = np.linalg.inv(P) * 1e9 * 1000.0       # nF/km
    return C, P


def sequence_check(Z, cfg):
    """Positive-sequence impedance via the power-invariant six-phase transform."""
    a = np.exp(-2j * np.pi / 6)
    S = np.array([[a ** (i * j) for j in range(6)] for i in range(6)]) / np.sqrt(6)
    Zs = S @ Z @ np.conj(S).T
    return np.diag(Zs)


def main():
    cfg = CONFIG
    r_eq, gmr_eq = bundle_equivalent(cfg["r_cond"], cfg["gmr"],
                                     cfg["n_bundle"], cfg["bundle_spacing"])

    Z, R, L = series_impedance(cfg, gmr_eq)
    C, P = shunt_capacitance(cfg, r_eq)
    G = np.zeros((6, 6))    # dielectric loss normally negligible for OHL

    np.set_printoptions(precision=5, suppress=True, linewidth=150)

    print("=" * 72)
    print("SIX-PHASE LINE MATRICES  (per unit length)")
    print(f"f = {cfg['f']} Hz,  rho_e = {cfg['rho_e']} Ohm-m,  "
          f"length = {cfg['length_km']} km")
    print(f"bundle: n = {cfg['n_bundle']},  r_eq = {r_eq:.5f} m,  "
          f"GMR_eq = {gmr_eq:.5f} m")
    print("=" * 72)
    for name, M, unit in (("R", R, "Ohm/km"), ("L", L, "mH/km"),
                          ("C", C, "nF/km"), ("G", G, "uS/km")):
        print(f"\n{name}  [{unit}]")
        print(M)

    # ---- verification -------------------------------------------------
    print("\n" + "-" * 72)
    print("VERIFICATION")
    for name, M in (("R", R), ("L", L), ("C", C)):
        sym = np.allclose(M, M.T, atol=1e-9)
        print(f"  {name} symmetric:            {sym}")
    eig = np.linalg.eigvalsh(C)
    print(f"  C positive definite:     {np.all(eig > 0)}  "
          f"(min eig {eig.min():.4g})")

    zs = sequence_check(Z, cfg)
    print("\n  Modal/sequence diagonal of S6 Z S6^H  [Ohm/km]:")
    for i, z in enumerate(zs):
        print(f"    mode {i}:  {z.real:+.5f} {z.imag:+.5f}j")
    print("    (mode 0 = zero sequence, 1 = positive, 5 = negative,")
    print("     3 = anti-phase, 2/4 = higher-order pair)")

    if cfg["expected_z1"] is not None:
        err = abs(zs[1] - cfg["expected_z1"]) / abs(cfg["expected_z1"]) * 100
        print(f"\n  Positive-sequence deviation from expected: {err:.2f} %")

    print("\n" + "-" * 72)
    print("NOTE: values above derive from the PLACEHOLDER geometry in CONFIG.")
    print("They are NOT the authors' line data and must not be published")
    print("until CONFIG is replaced with the real conductor geometry.")


if __name__ == "__main__":
    main()
