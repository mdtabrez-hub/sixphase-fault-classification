#!/usr/bin/env bash
# =====================================================================
# reproduce.sh -- regenerate the case set from the master seed and
# reproduce every table and figure of the paper.
#
#   ./reproduce.sh check      line constants, transform, taxonomy and
#                             solver verification only (seconds)
#   ./reproduce.sh cases      generate and score the frozen case set
#                             (resumable; several hours)
#   ./reproduce.sh analyse    statistics, classifiers, figures, tables
#   ./reproduce.sh verify     check outputs against SHA256SUMS.txt
#   ./reproduce.sh all        everything, in order   [default]
#
# Requires Python 3.10+ and the packages in requirements.txt. No MATLAB
# and no Simulink: every result in the paper is produced by the Python
# modules in src/.
#
# Determinism: the master seed is 20260829 (Table 13). Every chunk is
# seeded from the master seed and the chunk identifier alone, so a run
# resumed after an interruption gives the same case set as one pass.
# =====================================================================
set -e
cd "$(dirname "$0")"
REPO="$PWD"
export PYTHONPATH="$REPO/src:$PYTHONPATH"
# Pin thread pools: oversubscription makes the tree ensembles far slower.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

N_MAIN=${N_MAIN:-256}      # scenarios per internal taxonomy class
N_EXT=${N_EXT:-256}        # external / reverse / no-fault scenarios
N_HIF=${N_HIF:-128}
N_IBR=${N_IBR:-64}
BATCH=${BATCH:-400}
OUT=${OUT:-$REPO/out/Data/main}

mkdir -p "$OUT" "$REPO/out/Logs" "$REPO/out/Tables" "$REPO/out/Figures"

step_check() {
  echo "== line constants, transform, taxonomy =========================="
  ( cd src && python3 sixphase_core.py )
  ( cd src && python3 spconst.py )
  ( cd src && python3 enumerate_taxonomy.py )
  echo "== solver verification =========================================="
  ( cd src && python3 validate.py )
  echo "== representative waveform records =============================="
  ( cd src && python3 spwave.py )
}

step_cases() {
  echo "== frozen case set =============================================="
  echo "   resumable: each chunk checkpoints into $OUT/parts/"
  cd src
  until python3 -u runsim.py "$N_MAIN" "$N_EXT" "$N_HIF" "$N_IBR" "$OUT" "$BATCH" 1e9 \
        | tee -a "$REPO/out/Logs/runsim.log" | grep -q FINALISED; do
    echo "   ... interrupted, resuming"
    sleep 2
  done
  cd "$REPO"
  echo "   wrote $OUT/cases.csv and $OUT/features.npy"
}

step_analyse() {
  echo "== statistics ==================================================="
  ( cd src && python3 analyse.py )
  echo "== classifier benchmark ========================================="
  ( cd src && python3 classify.py )
  echo "== high-impedance-fault index: ablation and symmetry control ===="
  ( cd src && python3 hif_ablation.py )
  echo "== statistics, second pass (picks up the ablation keys) ========="
  ( cd src && python3 analyse.py )
  echo "== figures ======================================================"
  ( cd src && python3 make_figures_v3.py ) || true
  ( cd src && python3 makefigs_results.py )
  echo "== manuscript number check ======================================"
  ( cd src && python3 verify_manuscript_numbers.py ) || true
}

step_verify() {
  if [ ! -f SHA256SUMS.txt ]; then
    echo "No SHA256SUMS.txt in this release; nothing to verify against."
    return 0
  fi
  echo "== checksum verification ========================================"
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -c SHA256SUMS.txt
  else shasum -a 256 -c SHA256SUMS.txt; fi
}

case "${1:-all}" in
  check)   step_check ;;
  cases)   step_cases ;;
  analyse) step_analyse ;;
  verify)  step_verify ;;
  all)     step_check; step_cases; step_analyse; step_verify ;;
  *) echo "usage: $0 {check|cases|analyse|verify|all}"; exit 1 ;;
esac
echo "done."
