#!/usr/bin/env python3
"""
make_tables.py -- emits the machine-readable source of every computed table.

  derived/table_II_line_matrices.csv    entries of Table II
  derived/table_III_modal_sequence.csv  entries of Table III (new)
  derived/table_IV_shunt_taxonomy.csv   the shunt-class counts of Table IV
Nothing here is measured; every row is a deterministic function of Table I.
"""
import csv, os
import numpy as np
import sixphase_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "derived"); os.makedirs(OUTD, exist_ok=True)
S = core.summary(); L, C, Zs = S["L"], S["C"], S["Zseq"]

# ---- Table II -------------------------------------------------------
rows = [dict(Entry=f"L, |i-j| = {k}" if k else "L, self (Lii)",
             Value=f"{L[0,k]:.6f}", Unit="mH/km", Status="Calculated") for k in range(6)]
for lbl, (i, j) in [("C11 = C66",(0,0)),("C22 = C55",(1,1)),("C33 = C44",(2,2)),
                    ("C12 = C56",(0,1)),("C23 = C45",(1,2)),("C34",(2,3)),
                    ("C13 = C46",(0,2)),("C24 = C35",(1,3)),("C14 = C36",(0,3)),
                    ("C25",(1,4)),("C15 = C26",(0,4)),("C16",(0,5))]:
    rows.append(dict(Entry=lbl, Value=f"{C[i,j]:.4f}", Unit="nF/km", Status="Calculated"))
with open(os.path.join(OUTD,"table_II_line_matrices.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,["Entry","Value","Unit","Status"]); w.writeheader(); w.writerows(rows)

# ---- Table III (new) ------------------------------------------------
# IMPORTANT: the modal basis (eigenvectors of ZY) and the sequence basis
# (columns of S6) are DIFFERENT bases on an untransposed line -- see
# Section 2.2.  They are therefore reported as two separate blocks and
# their indices must never be paired row-by-row.
a, b, v, tau = S["alpha"] * 1e4, S["beta"] * 1e4, S["vfrac"], S["tau_us"]
mode_name = ["a1", "a2", "a3", "a4", "a5", "e"]
mode_kind = ["aerial"] * 5 + ["earth-return"]
t3 = []
for k in range(6):
    t3.append(dict(Block="Propagation mode (eigenbasis of ZY)",
                   Index=mode_name[k], Descriptor=mode_kind[k],
                   Quantity1_label="alpha (1e-4/km)", Quantity1=f"{a[k]:.3f}",
                   Quantity2_label="beta (1e-4/km)",  Quantity2=f"{b[k]:.2f}",
                   Quantity3_label="v/c",             Quantity3=f"{v[k]:.3f}",
                   Quantity4_label="travel time (us)",Quantity4=f"{tau[k]:.0f}",
                   Status="Calculated"))
seq_desc = ["zero sequence", "positive sequence",
            "higher-order pair", "anti-phase mode",
            "higher-order pair", "negative sequence"]
for k in range(6):
    t3.append(dict(Block="Sequence component (columns of S6)",
                   Index=f"k = {k}", Descriptor=seq_desc[k],
                   Quantity1_label="R (ohm, 100 km)", Quantity1=f"{Zs[k,k].real:.2f}",
                   Quantity2_label="X (ohm, 100 km)", Quantity2=f"{Zs[k,k].imag:.2f}",
                   Quantity3_label="|Z| (ohm)",       Quantity3=f"{abs(Zs[k,k]):.2f}",
                   Quantity4_label="angle (deg)",     Quantity4=f"{np.degrees(np.angle(Zs[k,k])):.2f}",
                   Status="Calculated"))
with open(os.path.join(OUTD, "table_III_modal_sequence.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(t3[0].keys())); w.writeheader(); w.writerows(t3)

# ---- Table IV shunt counts -----------------------------------------
rows2=[]; src=list(csv.DictReader(open(os.path.join(HERE,"taxonomy_full.csv"))))
for k in range(1,7):
    g=sum(1 for r in src if r["group"]=="A_shunt" and int(r["k"])==k and r["grounded"]=="1")
    u=sum(1 for r in src if r["group"]=="A_shunt" and int(r["k"])==k and r["grounded"]=="0")
    rows2.append(dict(Conductors=k, Grounded=g, Ungrounded=u, Total=g+u))
rows2.append(dict(Conductors="Subtotal",
                  Grounded=sum(r["Grounded"] for r in rows2),
                  Ungrounded=sum(r["Ungrounded"] for r in rows2),
                  Total=sum(r["Total"] for r in rows2)))
with open(os.path.join(OUTD,"table_IV_shunt_taxonomy.csv"),"w",newline="") as f:
    w=csv.DictWriter(f,["Conductors","Grounded","Ungrounded","Total"]); w.writeheader(); w.writerows(rows2)

print("wrote table_II_line_matrices.csv, table_III_modal_sequence.csv, table_IV_shunt_taxonomy.csv")
for r in t3:
    print(f"  {r['Block'][:34]:<34} {r['Index']:<5} {r['Descriptor']:<18} "
          f"{r['Quantity1']:>8} {r['Quantity2']:>8} {r['Quantity3']:>8} {r['Quantity4']:>8}")
