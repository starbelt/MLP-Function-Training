#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Get the directory of this script (smart-energy root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_DIR_TRN="$SCRIPT_DIR/06-trn-val-mlp"
BASE_DIR_TST="$SCRIPT_DIR/08-tst-mlp"

SCRIPT_TRN="$BASE_DIR_TRN/trn_val_mlp.py"
SCRIPT_TST="$BASE_DIR_TST/tst_mlp.py"

SPLIT="$SCRIPT_DIR/05-split-data"
SRC_DIR="$SCRIPT_DIR/05-split-data"   # must contain tst/ inside it

# Activate the virtual environment
source "$SCRIPT_DIR/.p3-env/bin/activate"

# -----------------------------
# Helpers
# -----------------------------
cfg_to_results_dir_trn() {
  # input: cfg dir path (.../mlp-XXXX-cfg/)
  # output: training results dir path (.../06-trn-val-mlp/mlp-XXXX-results)
  local cfg_dir="$1"
  local cfg_base
  cfg_base="$(basename "${cfg_dir%/}")"          # mlp-0256-cfg
  local results_base="${cfg_base%-cfg}-results"  # mlp-0256-results
  echo "$BASE_DIR_TRN/$results_base"
}

cfg_to_results_dir_tst() {
  # input: cfg dir path (.../mlp-XXXX-cfg/)
  # output: testing results dir path (.../08-tst-mlp/mlp-XXXX-results)
  local cfg_dir="$1"
  local cfg_base
  cfg_base="$(basename "${cfg_dir%/}")"          # mlp-0256-cfg
  local results_base="${cfg_base%-cfg}-results"  # mlp-0256-results
  echo "$BASE_DIR_TST/$results_base"
}

find_model_and_norm() {
  # Args: RESULTS_DIR stem
  # Prints: "model_pt|norm_pt" if found, returns 0; else returns 1
  local results_dir="$1"
  local stem="$2"

  # Preferred layout: .../mlp-XXXX-results/<stem>/<stem>.pt
  local model_nested="$results_dir/$stem/$stem.pt"
  local norm_nested="$results_dir/$stem/$stem-norm.pt"

  # Fallback layout: .../mlp-XXXX-results/<stem>.pt
  local model_flat="$results_dir/$stem.pt"
  local norm_flat="$results_dir/$stem-norm.pt"

  if [[ -f "$model_nested" && -f "$norm_nested" ]]; then
    echo "$model_nested|$norm_nested"
    return 0
  fi
  if [[ -f "$model_flat" && -f "$norm_flat" ]]; then
    echo "$model_flat|$norm_flat"
    return 0
  fi
  return 1
}

# -----------------------------
# Optional data generation
# -----------------------------
read -p "Generate Data? Enter your choice (y/n): " choice
if [[ "$choice" == "y" ]]; then
  echo "Generating Data..."
  rm -rf "$SCRIPT_DIR/03-data/"*
  rm -f  "$SCRIPT_DIR/02-gen-data/data-cfg.json"
  python3 "$SCRIPT_DIR/02-gen-data/gen_data.py" "$SCRIPT_DIR/02-gen-data/test.csv" "$SCRIPT_DIR/03-data"
else
  echo "Skipping Data Generation."
fi

# -----------------------------
# Split data
# -----------------------------
rm -rf "$SCRIPT_DIR/05-split-data/trn"* "$SCRIPT_DIR/05-split-data/tst"* "$SCRIPT_DIR/05-split-data/val"*
python3 "$SCRIPT_DIR/05-split-data/split_data.py" "$SCRIPT_DIR/03-data/" "$SCRIPT_DIR/05-split-data/"

# -----------------------------
# Choose a config folder
# -----------------------------
echo "Available config folders:"
select CFG_DIR in "$BASE_DIR_TRN"/mlp-*-cfg/; do
  if [[ -n "${CFG_DIR:-}" ]]; then
    echo "Selected folder: $CFG_DIR"
    break
  else
    echo "Invalid selection, try again."
  fi
done

# -----------------------------
# Training loop
# -----------------------------
TRN_RESULTS_DIR="$(cfg_to_results_dir_trn "$CFG_DIR")"
mkdir -p "$TRN_RESULTS_DIR"

MAX_RUNS=6
count=0

trained_stems=()

for cfg in "$CFG_DIR"/mlp-*-*.json; do
  echo "-> Training config: $cfg"
  python3 "$SCRIPT_TRN" "$cfg" "$SPLIT" "$TRN_RESULTS_DIR"

  trained_stems+=( "$(basename "$cfg" .json)" )

  ((++count))
  if (( count >= MAX_RUNS )); then
    echo "Reached limit of $MAX_RUNS configs. Stopping."
    break
  fi
done

# -----------------------------
# Testing loop
#   - stores test CSV outputs in: 08-tst-mlp/<mlp-XXXX-results>/
#   - tests ONLY the models trained above (trained_stems)
# -----------------------------
[[ -d "$SRC_DIR/tst" ]] || { echo "Missing test dir: $SRC_DIR/tst"; exit 1; }

TST_RESULTS_DIR="$(cfg_to_results_dir_tst "$CFG_DIR")"
mkdir -p "$TST_RESULTS_DIR"

echo "=== TESTING selected folder only: $CFG_DIR ==="
echo "Test outputs -> $TST_RESULTS_DIR"

for stem in "${trained_stems[@]}"; do
  cfg="$CFG_DIR/$stem.json"

  if [[ ! -f "$cfg" ]]; then
    echo "Skipping (missing cfg): $cfg"
    continue
  fi

  if ! pair="$(find_model_and_norm "$TRN_RESULTS_DIR" "$stem")"; then
    echo "Skipping (no model/norm found): $stem"
    echo "  looked for:"
    echo "    $TRN_RESULTS_DIR/$stem/$stem.pt"
    echo "    $TRN_RESULTS_DIR/$stem/$stem-norm.pt"
    echo "    $TRN_RESULTS_DIR/$stem.pt"
    echo "    $TRN_RESULTS_DIR/$stem-norm.pt"
    continue
  fi

  model_pt="${pair%%|*}"
  norm_pt="${pair##*|}"

  echo "▶ Testing: $stem"
  python3 "$SCRIPT_TST" "$cfg" "$model_pt" "$norm_pt" "$SRC_DIR" "$TST_RESULTS_DIR"
done

# -----------------------------
# Visualization loop (TRAINING ONLY)
#   - saves plots next to each losses CSV (inside the results folders)
# -----------------------------
VIZ_SCRIPT="$SCRIPT_DIR/07-viz-trn-val/viz_trn_val.py"
TRAIN_RESULTS_ROOT="$SCRIPT_DIR/06-trn-val-mlp"

run_dirs=( "$TRAIN_RESULTS_ROOT"/mlp-*-results/ )
if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "No training results dirs found in $TRAIN_RESULTS_ROOT"
  exit 1
fi

for run_dir in "${run_dirs[@]}"; do
  echo "=== TRAIN RESULTS dir: $run_dir ==="

  # Find losses CSVs up to 2 levels deep inside each results folder
  mapfile -t losses < <(find "$run_dir" -maxdepth 2 -type f -name "*losses*.csv" -print)

  if [[ ${#losses[@]} -eq 0 ]]; then
    echo "No losses CSVs found in $run_dir"
    continue
  fi

  for csv in "${losses[@]}"; do
    csv_dir="$(cd "$(dirname "$csv")" && pwd)"
    echo "Visualizing: $csv -> $csv_dir"
    python3 "$VIZ_SCRIPT" "$csv" "$csv_dir" || { echo "Viz failed on: $csv"; exit 1; }
  done
done

deactivate
