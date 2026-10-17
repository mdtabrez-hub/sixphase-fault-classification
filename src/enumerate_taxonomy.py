#!/usr/bin/env python3
"""
enumerate_taxonomy.py
=====================
Generates the complete, frozen six-phase fault-class enumeration for the
adaptive sequence-modal protection benchmark.

Every class is derived combinatorially. NO measured or simulated data is
involved -- this script produces the case LIST, not any result.

Outputs
-------
  taxonomy_full.csv    one row per fault class, with a stable case_id
  taxonomy_summary.csv group-level counts

Symmetry convention
-------------------
Conductors are indexed 0..5 around the six-phase arrangement, so that
conductor k sits at 60k electrical degrees. The relay observes the system
through six-phase symmetric measurements, so open-conductor topologies
related by a rotation or reflection are indistinguishable and are counted
once. That symmetry group is the dihedral group D6 (order 12).

Author: (see manuscript)
"""

from itertools import combinations
import csv

N = 6
POSITIONS = [("near", 10), ("mid", 50), ("remote", 90)]   # % of line length


# ----------------------------------------------------------------------
# Symmetry group
# ----------------------------------------------------------------------
def dihedral_group(n=N):
    rots = [tuple((i + r) % n for i in range(n)) for r in range(n)]
    refs = [tuple((r - i) % n for i in range(n)) for r in range(n)]
    return rots + refs


D6 = dihedral_group()


def canonical(subset):
    """Lexicographically smallest image of `subset` under D6."""
    return min(tuple(sorted(g[i] for i in subset)) for g in D6)


def orbit_reps(k):
    """One representative per D6-orbit of k-subsets."""
    seen, reps = set(), []
    for s in combinations(range(N), k):
        c = canonical(s)
        if c not in seen:
            seen.add(c)
            reps.append(c)
    return reps


# ----------------------------------------------------------------------
# GROUP A -- shunt faults (no symmetry reduction: every subset is a class)
# ----------------------------------------------------------------------
def group_a_shunt():
    """
    All non-empty conductor subsets, grounded; plus all subsets of size >= 2,
    ungrounded. A single ungrounded conductor is not a shunt fault.
    """
    rows = []
    for k in range(1, N + 1):
        for s in combinations(range(N), k):
            rows.append(dict(group="A_shunt", k=k, conductors="".join(map(str, s)),
                             grounded=1, topology="", position="", detail=""))
            if k >= 2:
                rows.append(dict(group="A_shunt", k=k, conductors="".join(map(str, s)),
                                 grounded=0, topology="", position="", detail=""))
    return rows


# ----------------------------------------------------------------------
# GROUP B -- open-conductor (series) faults
# ----------------------------------------------------------------------
def group_b_open():
    """
    Open-conductor topologies of size 1..3, reduced under D6, each
    instantiated at three line positions.
    """
    rows = []
    for k in (1, 2, 3):
        for rep in orbit_reps(k):
            for pname, ppct in POSITIONS:
                rows.append(dict(group="B_open", k=k,
                                 conductors="".join(map(str, rep)),
                                 grounded=0,
                                 topology="".join(map(str, rep)),
                                 position=f"{pname}({ppct}%)",
                                 detail="open conductor(s)"))
    return rows


# ----------------------------------------------------------------------
# GROUP C -- open conductor combined with a shunt fault
# ----------------------------------------------------------------------
def group_c_open_shunt():
    """
    For each open topology, add a single phase-to-ground fault on one of three
    physically distinct targets:
        t1  the lowest-indexed OPEN conductor  (fault on the severed conductor)
        t2  the nearest HEALTHY conductor going clockwise from t1
        t3  the conductor diametrically opposite t1  (index +3 mod 6)

    A target is dropped when it is degenerate:
        - t3 is dropped if it is itself open (the pairing then duplicates a
          same-conductor case already produced by t1)
        - any pairing whose (topology, target) canonical form has already been
          produced is dropped, preventing double counting under D6.
    """
    rows, seen = [], set()
    for k in (1, 2, 3):
        for rep in orbit_reps(k):
            openset = set(rep)
            t1 = min(rep)
            # nearest healthy conductor clockwise from t1
            t2 = next(((t1 + d) % N for d in range(1, N) if (t1 + d) % N not in openset), None)
            t3 = (t1 + 3) % N
            for label, tgt in (("on-open", t1), ("adjacent", t2), ("opposite", t3)):
                if tgt is None:
                    continue
                if label == "opposite" and tgt in openset:
                    continue                      # degenerate with 'on-open'
                key = (canonical(rep), label, (tgt - t1) % N)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(group="C_open_shunt", k=k,
                                 conductors="".join(map(str, rep)),
                                 grounded=1,
                                 topology="".join(map(str, rep)),
                                 position="mid(50%)",
                                 detail=f"open {rep} + LG on c{tgt} ({label})"))
    return rows


# ----------------------------------------------------------------------
# GROUP D -- evolving faults
# ----------------------------------------------------------------------
def group_d_evolving():
    """
    Transitions between shunt FAMILIES, a family being (k conductors, grounded).
    A single conductor must be grounded to constitute a fault, so (1,0) is not
    a family.

    A transition is admissible when it is a physically monotone escalation:
        (i)  ground never clears:            g' >= g
        (ii) conductor count never falls:    k' >= k
        (iii) escalation is single-step:     k' - k in {0, 1}
        (iv) something actually changes:     (k',g') != (k,g)
    Condition (i) is what rules out a grounded fault becoming ungrounded, which
    is not a physical evolution of an existing arc.
    """
    fams = [(1, 1)] + [(k, g) for k in range(2, N + 1) for g in (0, 1)]
    rows = []
    for (k, g) in fams:
        for (kp, gp) in fams:
            if (kp, gp) == (k, g):
                continue
            if gp < g:            # (i)
                continue
            if kp < k:            # (ii)
                continue
            if kp - k > 1:        # (iii)
                continue
            rows.append(dict(group="D_evolving", k=k,
                             conductors="", grounded=g,
                             topology="", position="mid(50%)",
                             detail=f"({k}C,{'G' if g else 'ung'}) -> "
                                    f"({kp}C,{'G' if gp else 'ung'})"))
    return rows


# ----------------------------------------------------------------------
def main():
    groups = [("A", group_a_shunt()), ("B", group_b_open()),
              ("C", group_c_open_shunt()), ("D", group_d_evolving())]

    allrows, summary = [], []
    for tag, rows in groups:
        for i, r in enumerate(rows, 1):
            r["case_id"] = f"{tag}{i:03d}"
        allrows += rows
        summary.append((tag, rows[0]["group"], len(rows)))

    cols = ["case_id", "group", "k", "conductors", "grounded",
            "topology", "position", "detail"]
    with open("taxonomy_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)

    with open("taxonomy_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "group", "count"])
        w.writerows(summary)
        w.writerow(["", "TOTAL", len(allrows)])

    print(f"{'tag':<4}{'group':<16}{'count':>6}")
    print("-" * 26)
    for tag, g, c in summary:
        print(f"{tag:<4}{g:<16}{c:>6}")
    print("-" * 26)
    print(f"{'':<4}{'TOTAL':<16}{len(allrows):>6}")

    # ---- verification of the shunt sub-counts against binomial identities ----
    from math import comb
    gnd = sum(1 for r in allrows if r["group"] == "A_shunt" and r["grounded"] == 1)
    ung = sum(1 for r in allrows if r["group"] == "A_shunt" and r["grounded"] == 0)
    assert gnd == sum(comb(6, k) for k in range(1, 7)) == 63, "grounded count"
    assert ung == sum(comb(6, k) for k in range(2, 7)) == 57, "ungrounded count"
    assert gnd + ung == 120
    print("\nshunt check: 63 grounded + 57 ungrounded = 120  OK")
    print(f"open topologies (D6-reduced, sizes 1-3): "
          f"{sum(len(orbit_reps(k)) for k in (1,2,3))}")


if __name__ == "__main__":
    main()
