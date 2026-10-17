#!/usr/bin/env python3
"""
classify.py -- the three-family classifier benchmark of Table VII.

Partitioning is at scenario level, as Table VIII requires: every distinct
(fault location, fault resistance) operating point is assigned wholly to the
development, validation or test partition, so no window drawn from a scenario
seen in training can appear in the test partition.

Model families
  * gradient-boosted decision trees  (HistGradientBoostingClassifier)
  * support-vector machine, RBF kernel, one-vs-rest
  * a temporal model over the three-window feature stack

The temporal model of Table VII is specified as a 1-D convolutional network.
No deep-learning framework is installed in the reproducibility environment, so
a multilayer perceptron over the same three-window stack is trained in its
place and is reported as such; the 1-D CNN remains outstanding and is recorded
in RED_Actions.md.

Reported: macro-F1 with bootstrap confidence intervals, balanced accuracy,
per-family and per-conductor confusion matrices, and permutation importance in
the sequence-modal domain.
"""
import os, json, time
# Pin the BLAS/OpenMP thread count before importing anything that spawns a pool.
# Oversubscription against a concurrently running simulation makes the tree
# ensembles roughly two orders of magnitude slower.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "out", "Data", "main")
TBL = os.path.join(ROOT, "out", "Tables")
SEED = 20260829
os.makedirs(TBL, exist_ok=True)

FEAT_BLOCKS = [("|I_k|", 6), ("angle I_k", 6), ("|V_k|", 6), ("angle V_k", 6),
               ("E_k^I", 6), ("E_k^V", 6), ("R_kl", 30), ("dtheta_kl", 15),
               ("H2", 1), ("H3", 1), ("eta_tr", 1), ("eta_CT", 1)]


def block_names():
    n = []
    for b, k in FEAT_BLOCKS:
        n += [f"{b}[{i}]" for i in range(k)] if k > 1 else [b]
    return n


def load():
    d = pd.read_csv(os.path.join(DATA, "cases.csv"))
    X = np.load(os.path.join(DATA, "features.npy"))          # (N, 3, 85)
    assert len(d) == len(X), (len(d), len(X))
    m = d.kind == "internal"
    d = d[m].reset_index(drop=True); X = X[m.values]
    X = X.reshape(len(X), -1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # scale-invariant transform of the magnitude blocks
    return d, X


def splits(d, seed=SEED):
    """
    Scenario-level three-way split, as Table VIII requires: a distinct
    (fault location, fault resistance) operating point is assigned wholly to
    one partition, so no window drawn from a scenario seen in training can
    appear in the test partition.

    The assignment is made WITHIN each taxonomy group.  The open-conductor,
    open-plus-shunt and evolving classes are instantiated at three fixed line
    positions, so a split over the pooled operating points would place whole
    groups in a single partition and leave the test set without them.
    """
    rng = np.random.default_rng(seed)
    a = np.full(len(d), -1, np.int64)
    nd = nv = nt = 0
    for g in sorted(d.group.unique()):
        m = (d.group.values == g)
        sub = d.loc[m, ["d_km", "Rf_ohm"]]
        uniq = [tuple(x) for x in sorted({tuple(r) for r in sub.to_numpy()})]
        n = len(uniq)
        order = rng.permutation(n)
        i1 = max(1, int(round(0.6 * n)))
        i2 = max(i1 + 1, int(round(0.8 * n))) if n >= 3 else i1
        i2 = min(i2, n - 1) if n >= 3 else i2
        lab = {}
        for j, o in enumerate(order):
            lab[uniq[o]] = 0 if j < i1 else (1 if j < i2 else 2)
        nd += sum(v == 0 for v in lab.values())
        nv += sum(v == 1 for v in lab.values())
        nt += sum(v == 2 for v in lab.values())
        a[m] = [lab[tuple(r)] for r in sub.to_numpy()]
    assert (a >= 0).all()
    return a, nd, nv, nt


def boot_f1(y, p, B=300, seed=SEED, cap=8000):
    """Bootstrap interval on macro-F1.  The resample is capped so the interval
    costs seconds rather than minutes; the point estimate uses the whole test
    partition."""
    rng = np.random.default_rng(seed)
    n = min(len(y), cap)
    s = [f1_score(y[i], p[i], average="macro", zero_division=0)
         for i in (rng.integers(0, len(y), n) for _ in range(B))]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    d, X = load()
    part, nd, nv, nt = splits(d)
    print(f"operating points: dev {nd}  val {nv}  test {nt}")
    # The 178-class label is a product, not 178 independent categories: a class
    # is (taxonomy group) x (fault family) x (which conductors are involved).
    # It is therefore classified as that product, and the exact-match accuracy of
    # the composition is reported.  A flat 178-way tree ensemble fits one tree
    # per class per boosting iteration and does not finish in a defensible time.
    tasks = {
        "family": d.apply(lambda r: f"{int(r.nc)}C-{'G' if r.grounded else 'U'}"
                          if r.nc > 0 else ("OPEN" if r.nopen > 0 else "NONE"),
                          axis=1).values,
        "group": d.group.values,
    }
    # per-conductor involvement, six binary tasks
    bits = np.array([[(int(b) >> (5 - k)) & 1 for k in range(6)] for b in d.mask_bits])

    models = {
        "gradient-boosted trees": lambda: HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.1, max_iter=300, random_state=SEED,
            class_weight="balanced"),
        "SVM (RBF, one-vs-rest)": lambda: make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced",
                decision_function_shape="ovr", cache_size=512, random_state=SEED)),
        "temporal MLP (CNN substitute)": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=250,
                          early_stopping=True, n_iter_no_change=15,
                          random_state=SEED)),
    }
    rows = []
    best = {}
    for task, y_raw in tasks.items():
        # integer-encode the labels: sklearn's MLP early-stopping path does not
        # accept string targets
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)
        tr = part == 0; te = part == 2
        # SVM and MLP are cubic / slow: cap the development set for them
        for name, mk in models.items():
            cap = 4000 if name.startswith("SVM") else 30000
            itr = np.nonzero(tr)[0]
            if len(itr) > cap:
                itr = np.random.default_rng(SEED).choice(itr, cap, replace=False)
            t0 = time.time()
            m = mk(); m.fit(X[itr], y[itr])
            fit_s = time.time() - t0
            t0 = time.time(); p = m.predict(X[te]); pred_s = (time.time() - t0) / te.sum()
            f1 = f1_score(y[te], p, average="macro", zero_division=0)
            lo, hi = boot_f1(y[te], p)
            # Model SELECTION uses the validation partition; the test partition
            # is scored once and never used to choose between candidates.
            # Table VIII declares three partitions and Section 3.4 says the
            # models are scored once on the held-out test set: selecting on the
            # test set would make that statement false and would bias the
            # reported figure upward by the maximum over the candidates.
            va = part == 1
            pv = m.predict(X[va])
            f1_val = f1_score(y[va], pv, average="macro", zero_division=0)
            rows.append(dict(task=task, model=name, n_train=len(itr),
                             n_val=int(va.sum()), n_test=int(te.sum()),
                             macro_F1_val=f1_val,
                             macro_F1=f1, F1_lo=lo, F1_hi=hi,
                             balanced_accuracy=balanced_accuracy_score(y[te], p),
                             accuracy=accuracy_score(y[te], p),
                             fit_seconds=fit_s, predict_ms_per_window=pred_s * 1e3))
            print(f"  {task:9s} {name:32s} val {f1_val:.4f} | test {f1:.4f} "
                  f"[{lo:.4f},{hi:.4f}]")
            if task not in best or f1_val > best[task][1]:
                best[task] = (name, f1_val, m, le.inverse_transform(p), le)
    pd.DataFrame(rows).to_csv(os.path.join(TBL, "classifier_benchmark.csv"), index=False)

    # confusion matrix of the selected model on the family task
    name, f1, m, p, _le = best["family"]
    te = part == 2
    y = tasks["family"][te]
    lab = sorted(set(tasks["family"]))
    cm = confusion_matrix(y, p, labels=lab)
    pd.DataFrame(cm, index=lab, columns=lab).to_csv(
        os.path.join(TBL, "confusion_family.csv"))
    pr, rc, f1s, sup = precision_recall_fscore_support(y, p, labels=lab, zero_division=0)
    pd.DataFrame(dict(family=lab, precision=pr, recall=rc, f1=f1s, support=sup)).to_csv(
        os.path.join(TBL, "per_family_scores.csv"), index=False)

    # per-conductor identification with the selected family
    pc = []
    tr = part == 0
    cmats = {}
    bitpreds = []
    for k in range(6):
        mk = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1,
                                            max_iter=150, random_state=SEED,
                                            class_weight="balanced")
        itr6 = np.nonzero(tr)[0]
        if len(itr6) > 30000:
            itr6 = np.random.default_rng(SEED).choice(itr6, 30000, replace=False)
        mk.fit(X[itr6], bits[itr6, k])
        q = mk.predict(X[te])
        bitpreds.append(q)
        pr_, rc_, f_, _ = precision_recall_fscore_support(bits[te, k], q,
                                                          average="binary", zero_division=0)
        pc.append(dict(conductor=f"c{k}", precision=pr_, recall=rc_, f1=f_,
                       accuracy=accuracy_score(bits[te, k], q),
                       n_positive=int(bits[te, k].sum())))
        cmats[f"c{k}"] = confusion_matrix(bits[te, k], q).ravel()
    pd.DataFrame(pc).to_csv(os.path.join(TBL, "per_conductor_scores.csv"), index=False)
    pd.DataFrame(cmats, index=["TN", "FP", "FN", "TP"]).T.to_csv(
        os.path.join(TBL, "per_conductor_confusion.csv"))

    # permutation importance in the sequence-modal domain, on the family task
    names = block_names()
    full = [f"{n}@w{w}" for w in range(3) for n in names]
    gb = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1, max_iter=200,
                                        random_state=SEED, class_weight="balanced")
    sub = np.random.default_rng(SEED).choice(np.nonzero(tr)[0],
                                             min(8000, tr.sum()), replace=False)
    gb.fit(X[sub], tasks["family"][sub])
    subte = np.random.default_rng(SEED).choice(np.nonzero(te)[0],
                                               min(2000, te.sum()), replace=False)
    r = permutation_importance(gb, X[subte], tasks["family"][subte], n_repeats=3,
                               random_state=SEED, scoring="f1_macro", n_jobs=1)
    imp = pd.DataFrame(dict(feature=full, importance=r.importances_mean,
                            sd=r.importances_std)).sort_values("importance",
                                                               ascending=False)
    imp.to_csv(os.path.join(TBL, "permutation_importance.csv"), index=False)

    # ablation: drop whole feature blocks
    off = 0; blocks = {}
    for b, k in FEAT_BLOCKS:
        blocks[b] = [off + i + 85 * w for w in range(3) for i in range(k)]
        off += k
    ab = []
    base = None
    for drop in [None] + list(blocks):
        cols = np.ones(X.shape[1], bool)
        if drop is not None:
            cols[blocks[drop]] = False
        g = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.1, max_iter=120,
                                           random_state=SEED, class_weight="balanced")
        g.fit(X[sub][:, cols], tasks["family"][sub])
        f = f1_score(tasks["family"][te], g.predict(X[te][:, cols]),
                     average="macro", zero_division=0)
        if drop is None:
            base = f
        ab.append(dict(removed=("none (all features)" if drop is None else drop),
                       macro_F1=f, delta=f - base))
        print(f"  ablation drop {str(drop):12s} macro-F1 {f:.4f}")
    pd.DataFrame(ab).to_csv(os.path.join(TBL, "ablation.csv"), index=False)

    # Composed class decision: the taxonomy class is recovered only when the
    # group, the fault family and every one of the six conductor decisions are
    # simultaneously correct.  This is the strictest reading of "178-class
    # accuracy" and is the one reported.
    comp = None
    try:
        bitpred = np.stack(bitpreds, 1)
        ok = ((best["group"][3] == d.group.values[te])
              & (best["family"][3] == tasks["family"][te])
              & (bitpred == bits[te]).all(1))
        comp = float(ok.mean())
        pd.DataFrame([dict(metric="composed taxonomy-class exact-match accuracy",
                           value=comp, n=int(te.sum()),
                           definition="group AND family AND all six conductor "
                                      "decisions simultaneously correct")]).to_csv(
            os.path.join(TBL, "composed_class_accuracy.csv"), index=False)
        print(f"  composed taxonomy-class exact-match accuracy {comp:.4f}")
    except Exception as e:
        print("  composed accuracy skipped:", e)

    json.dump(dict(dev_points=nd, val_points=nv, test_points=nt,
                   best_family_model=best["family"][0],
                   best_family_F1=best["family"][1],
                   best_group_model=best["group"][0],
                   best_group_F1=best["group"][1],
                   composed_class_accuracy=comp,
                   sklearn=__import__("sklearn").__version__,
                   ablation_base_F1=base),
              open(os.path.join(ROOT, "out", "Data", "classify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
