#!/usr/bin/env python3
"""
hif_arc_regime.py -- fault-current regime of the declared HIF arc parameters.

Analytical evaluation of the peak-current envelope and half-cycle imbalance
implied by the two-branch asymmetric arc parameters declared in Section 2.5.
This is a CALCULATION from declared parameters, not a simulation of the arc
and not a measurement.  It reproduces every number quoted in Section 2.5.

Convention.  Imbalance is reported as (I_larger - I_smaller)/I_larger, the
same definition used for both the breakdown-voltage-only and the
path-resistance-only figures, so that the two are directly comparable.
"""
import numpy as np
import sixphase_core as core

V_PHG_KV   = 93.0                 # Table I, phase-to-ground rms
S_BASE_MVA = 100.0                # Table I
N_PHASE    = 6
VS_PLUS_KV, VS_MINUS_KV = 6.15, 5.50          # Section 2.5
RP_MIN_KOHM, RP_MAX_KOHM = 2.5, 6.0           # Section 2.5
I0_PLUS_A, I0_MINUS_A = 10.0, 8.0             # Section 2.5
SSC_MVA    = 3500.0               # Table I, base case source short-circuit level
XR_RATIO   = 15.0                 # Table I, base case
ZLK_PCT    = 10.0                 # Table I, transformer leakage, per bank
N_BANKS    = 2                    # Table I, two banks per terminal


def report():
    Vpk = V_PHG_KV * np.sqrt(2) * 1e3
    Ibase = S_BASE_MVA * 1e6 / (N_PHASE * V_PHG_KV * 1e3)
    out = {}
    out["V_peak_kV"] = Vpk / 1e3
    out["I_base_A"] = Ibase

    ip = lambda R, Vs: (Vpk - Vs * 1e3) / (R * 1e3)
    i_hi = ip(RP_MIN_KOHM, VS_PLUS_KV)        # smallest R -> largest current
    i_lo = ip(RP_MAX_KOHM, VS_PLUS_KV)
    out["I_peak_min_A"], out["I_peak_max_A"] = i_lo, i_hi
    out["I_peak_min_pct_base"] = 100 * i_lo / Ibase
    out["I_peak_max_pct_base"] = 100 * i_hi / Ibase

    # breakdown-voltage asymmetry alone, path resistance held equal
    a, b = Vpk - VS_PLUS_KV * 1e3, Vpk - VS_MINUS_KV * 1e3
    out["imbalance_breakdown_only_pct"] = 100 * (max(a, b) - min(a, b)) / max(a, b)
    # path-resistance asymmetry alone, breakdown voltage held equal
    c, d = 1 / RP_MIN_KOHM, 1 / RP_MAX_KOHM
    out["imbalance_resistance_only_pct"] = 100 * (c - d) / c

    # ---- bolted single-conductor-to-ground reference ------------------
    # For an m-phase system faulted on one conductor, the power-invariant
    # transform gives I_k = I_f/sqrt(m) in every sequence, hence
    #     I_f = m * V_phi_g / sum_k Z_k .
    # Two limiting fault positions bracket the answer:
    #   remote terminal -> full line impedance in every sequence
    #   local bus       -> source + transformer impedance only
    S = core.summary()
    Zline = np.diag(S["Zseq"])                       # six sequence impedances, Ohm
    Zb = V_PHG_KV * 1e3 / Ibase                      # six-phase base impedance, Ohm
    Zsrc = (S_BASE_MVA / SSC_MVA) * Zb * (1j * XR_RATIO + 1) / np.hypot(XR_RATIO, 1)
    Ztr = (ZLK_PCT / 100.0) / N_BANKS * Zb * 1j      # two banks in parallel
    Zsys = Zsrc + Ztr

    If_remote = N_PHASE * V_PHG_KV * 1e3 / abs(np.sum(Zline + Zsys))
    If_local = N_PHASE * V_PHG_KV * 1e3 / abs(N_PHASE * Zsys)
    out["Z_base_ohm"] = Zb
    out["I_bolted_remote_A"] = If_remote
    out["I_bolted_local_A"] = If_local
    out["hif_over_bolted_min_pct"] = 100 * i_hi / If_local
    out["hif_over_bolted_max_pct"] = 100 * i_hi / If_remote
    out["I0_plus_A"], out["I0_minus_A"] = I0_PLUS_A, I0_MINUS_A
    return out


if __name__ == "__main__":
    r = report()
    print("HIF ARC CURRENT REGIME  (calculated from declared parameters)")
    print("-" * 62)
    print(f"  phase-to-ground peak voltage        {r['V_peak_kV']:.1f} kV")
    print(f"  six-phase base current              {r['I_base_A']:.0f} A")
    print(f"  peak fault-current envelope         {r['I_peak_min_A']:.0f} to "
          f"{r['I_peak_max_A']:.0f} A")
    print(f"    as a fraction of base current     {r['I_peak_min_pct_base']:.0f}% to "
          f"{r['I_peak_max_pct_base']:.0f}%")
    print(f"  half-cycle imbalance, breakdown-voltage asymmetry alone   "
          f"{r['imbalance_breakdown_only_pct']:.2f}%")
    print(f"  half-cycle imbalance, path-resistance asymmetry alone     "
          f"{r['imbalance_resistance_only_pct']:.2f}%")
    print(f"  saturation currents I0+ / I0-       {r['I0_plus_A']:.0f} / "
          f"{r['I0_minus_A']:.0f} A")
    print()
    print(f"  six-phase base impedance            {r['Z_base_ohm']:.1f} Ohm")
    print(f"  bolted 1-conductor-to-ground current, m*V/sum(Z_k):")
    print(f"    fault at the local bus            {r['I_bolted_local_A']:.0f} A")
    print(f"    fault at the remote terminal      {r['I_bolted_remote_A']:.0f} A")
    print(f"  HIF peak as a fraction of it        "
          f"{r['hif_over_bolted_min_pct']:.1f}% to {r['hif_over_bolted_max_pct']:.1f}%")
    print()
    print("  STATUS: CALCULATED from declared parameters (Tables I and III).")
    print("  The bolted-current bracket is an ESTIMATE: it is a phasor")
    print("  calculation with no arc, CT saturation, converter or transformer")
    print("  saturation representation, and it presumes the base-case source")
    print("  strength of Table I.  It is not a simulated fault current and")
    print("  must not be reported as one.")
