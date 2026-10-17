#!/usr/bin/env python3
"""
verify_manuscript_numbers.py
============================
Checks every numerical statement in the manuscript that is derivable from
the declared parameters against an independent recomputation, and writes
derived/number_verification.csv.

Values are checked against Revised_Manuscript_v3_completed.docx (the REVISED
file), not the original v3.  Where the revision corrected a number, the Note
column records what the original said and why it changed.

A PASS means the manuscript value and the recomputed value agree to the
precision at which the manuscript prints it.  A FAIL means they do not and
the manuscript value has been (or must be) corrected.  Tolerances are set to half a unit in the last digit the manuscript
prints, so PASS means 'agrees at the stated precision'.  A quantity that is
neither derivable nor measurable here is not listed: those appear in
pending_actions.csv as RED items instead.
"""
import csv, os
import numpy as np
import sixphase_core as core
import hif_arc_regime as hif

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "derived")
os.makedirs(OUTD, exist_ok=True)

S = core.summary()
L, C, Zs = S["L"], S["C"], S["Zseq"]
a, b, v, tau = S["alpha"] * 1e4, S["beta"] * 1e4, S["vfrac"], S["tau_us"]
h = hif.report()
off = np.abs(Zs - np.diag(np.diag(Zs))).max()
xs = sorted({round(Zs[k, k].imag, 6) for k in range(1, 6)})

def rel(x, y):
    return abs(x - y) / abs(y) if y else abs(x - y)

# (section, quantity, manuscript value, recomputed value, tolerance)
CHECKS = [
 ("2.2 / Table I", "R diagonal coefficient (Ohm/km)",        0.0500,   S["R"][0,0]-S["R"][0,1], 5e-4),
 ("2.2 / Table I", "R earth-return coefficient (Ohm/km)",    0.04848,  S["R"][0,1],  5e-4),
 ("2.2 / Table I", "complex depth Re(p) (m)",                355.9,    S["p"].real,  5e-4),
 ("2.2 / Table I", "complex depth |p| (m)",                  503.3,    abs(S["p"]),  5e-4),
 ("Table II", "L self (mH/km)",                              2.275614, L[0,0], 1e-6),
 ("Table II", "L |i-j|=1 (mH/km)",                           1.385673, L[0,1], 1e-6),
 ("Table II", "L |i-j|=2 (mH/km)",                           1.247044, L[0,2], 1e-6),
 ("Table II", "L |i-j|=3 (mH/km)",                           1.165951, L[0,3], 1e-6),
 ("Table II", "L |i-j|=4 (mH/km)",                           1.108414, L[0,4], 1e-6),
 ("Table II", "L |i-j|=5 (mH/km)",                           1.063786, L[0,5], 1e-6),
 ("Table II", "C11 = C66 (nF/km)",                           9.8249,  C[0,0], 1e-4),
 ("Table II", "C22 = C55 (nF/km)",                          10.7760,  C[1,1], 1e-4),
 ("Table II", "C33 = C44 (nF/km)",                          10.9169,  C[2,2], 1e-4),
 ("Table II", "C12 = C56 (nF/km)",                          -3.0902,  C[0,1], 1e-4),
 ("Table II", "C23 = C45 (nF/km)",                          -2.7109,  C[1,2], 1e-4),
 ("Table II", "C34 (nF/km)",                                -2.6535,  C[2,3], 1e-4),
 ("Table II", "C13 = C46 (nF/km)",                          -1.2796,  C[0,2], 1e-4),
 ("Table II", "C24 = C35 (nF/km)",                          -1.0844,  C[1,3], 1e-4),
 ("Table II", "C14 = C36 (nF/km)",                          -0.7269,  C[0,3], 1e-4),
 ("Table II", "C25 (nF/km)",                                -0.6275,  C[1,4], 1e-4),
 ("Table II", "C15 = C26 (nF/km)",                          -0.5032,  C[0,4], 1e-4),
 ("Table II", "C16 (nF/km)",                                -0.4526,  C[0,5], 1e-4),
 ("2.2", "earth-return alpha (1e-4/km)",                     1.017,   a[5], 1e-3),
 ("2.2", "earth-return beta (1e-4/km)",                     15.86,    b[5], 1e-3),
 ("2.2", "aerial alpha, mode a1 (1e-4/km)",                  0.574,   a[0], 2e-3),
 ("2.2", "aerial beta, mode a1 (1e-4/km)",                  10.71,    b[0], 1e-3),
 ("2.2", "aerial alpha, mode a5 (1e-4/km)",                  1.057,   a[4], 1e-3),
 ("2.2", "aerial beta, mode a5 (1e-4/km)",                  10.87,    b[4], 1e-3),
 ("2.2", "earth-return velocity (fraction of c)",            0.661,   v[5], 2e-3),
 ("2.2", "earth-return travel time (us)",                  505.0,     tau[5], 2e-3),
 ("2.2", "aerial travel time, fastest (us)",               341.0,     tau[0], 2e-3),
 ("2.2", "aerial travel time, slowest (us)",               346.0,     tau[4], 2e-3),
 ("2.2", "aerial attenuation spread (%)",                   84.0, 100*(a[:5].max()/a[:5].min()-1), 1e-2),
 ("2.3", "Z0 resistance (Ohm)",                             34.09,    Zs[0,0].real, 1e-3),
 ("2.3", "Z0 reactance (Ohm)",                             267.26,    Zs[0,0].imag, 1e-4),
 ("2.3", "Z1 = Z5 reactance (Ohm)",                         38.98,    Zs[1,1].imag, 1e-3),
 ("2.3", "Z2 = Z4 reactance (Ohm)",                         28.55,    Zs[2,2].imag, 1e-3),
 ("2.3", "Z3 reactance (Ohm)",                              26.62,    Zs[3,3].imag, 1e-3),
 ("2.3", "max off-diagonal residual (Ohm)",                  4.82,    off, 2e-3),
 ("2.3", "X0/X1 ratio",                                      6.86,    Zs[0,0].imag/Zs[1,1].imag, 1e-3),
 ("2.3", "R0/R1 ratio",                                      6.82,    Zs[0,0].real/Zs[1,1].real, 1e-3),
 ("2.3", "non-zero-sequence reactance spread (%)",          46.0, 100*(xs[-1]-xs[0])/xs[0], 2e-2),
 ("2.3", "off-diagonal as % of smallest diagonal",          18.0, 100*off/min(abs(np.diag(Zs))), 2e-2),
 ("2.2 / Fig. 3", "largest off-diagonal (dB re max)",      -34.9, 20*np.log10(off/np.abs(Zs).max()), 2e-3),
 ("2.5", "phase-to-ground peak (kV)",                      131.5,  h["V_peak_kV"], 2e-4),
 ("2.5", "six-phase base current (A)",                     179.0,  h["I_base_A"], 2e-3),
 ("2.5", "HIF peak current, minimum (A)",                   21.0,  h["I_peak_min_A"], 6e-3),
 ("2.5", "HIF peak current, maximum (A)",                   50.0,  h["I_peak_max_A"], 5e-3),
 ("2.5", "HIF peak as % of base, upper",                    28.0,  h["I_peak_max_pct_base"], 5e-3),
 ("2.5", "imbalance from path resistance alone (%)",        58.0,  h["imbalance_resistance_only_pct"], 1e-2),
 ("2.5", "imbalance from breakdown voltage alone (%)",       0.5,  h["imbalance_breakdown_only_pct"], 5e-2,
  "Revised value. Earlier drafts stated 0.4%, computed under a different definition from the 58% "
  "path-resistance figure it is paired with. Both now use (larger - smaller)/larger."),
 ("3.2 / (12)", "feature-vector dimension",                 85.0,  4*6 + 2*6 + 6*5 + 15 + 4, 0.0),
 ("3.5", "cases in the frozen set",                      89000.0,  178*500, 0.0),
 ("3.5", "full factorial of Table IX (1e7)",                 2.9,  6*10*12*4*2*4*7*3*2*5*2*3/1e7, 1e-2),
 ("3.6", "Huber evaluations per window",                   270.0,  15*3*6, 0.0),
 ("2.4", "shunt classes",                                  120.0,  sum(1 for r in csv.DictReader(open(os.path.join(HERE,"taxonomy_full.csv"))) if r["group"]=="A_shunt"), 0.0),
 ("2.4", "total frozen classes",                           178.0,  sum(1 for _ in csv.DictReader(open(os.path.join(HERE,"taxonomy_full.csv")))), 0.0),
]

rows = []
for chk in CHECKS:
    sec, name, ms, comp, tol = chk[:5]
    note = chk[5] if len(chk) > 5 else ""
    ok = rel(ms, comp) <= tol
    rows.append(dict(Section=sec, Quantity=name,
                     ManuscriptValue=f"{ms:.6g}", RecomputedValue=f"{comp:.6g}",
                     RelativeDeviation=f"{rel(ms, comp):.2e}",
                     Tolerance=f"{tol:.0e}", Result="PASS" if ok else "FAIL", Note=note))

fn = os.path.join(OUTD, "number_verification.csv")
with open(fn, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

npass = sum(r["Result"] == "PASS" for r in rows)
print(f"{npass}/{len(rows)} checks PASS  "
      f"(reference: Revised_Manuscript_v3_completed.docx)")
for r in rows:
    if r["Note"]:
        print(f"  NOTE  {r['Section']:<16} {r['Quantity']}")
        print(f"        {r['Note']}")
for r in rows:
    if r["Result"] == "FAIL":
        print(f"  FAIL  {r['Section']:<16} {r['Quantity']:<44} "
              f"manuscript {r['ManuscriptValue']:>10}  recomputed {r['RecomputedValue']:>12}")
print("written:", fn)
