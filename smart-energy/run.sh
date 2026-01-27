#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Get the directory of this script (smart-energy root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_DIR_TRN="$SCRIPT_DIR/06-trn-val-mlp"
BASE_DIR_TST="$SCRIPT_DIR/08-tst-mlp"

SCRIPT_TRN="$BASE_DIR_TRN/trn_val_mlp.py"
SCRIPT_TST="$BASE_DIR_TST/tst_mlp.py"

SPLIT="$SCRIPT_DIR/05-split-data/"

# Activate the virtual environment
source "$SCRIPT_DIR/p3-env/bin/activate"

# ---- optional training pipeline (as you had it) ----
read -p "Generate Data? Enter your choice (y/n): " choice
if [ "$choice" = "y" ]; then
    echo "Generating Data..."
    rm -rf "$SCRIPT_DIR/03-data/"*
    rm -f  "$SCRIPT_DIR/02-gen-data/data-cfg.json"
    python3 "$SCRIPT_DIR/02-gen-data/gen_data.py" "$SCRIPT_DIR/02-gen-data/test.csv" "$SCRIPT_DIR/03-data"
else
    echo "Skipping Data Generation."
fi


rm -rf "$SCRIPT_DIR/05-split-data/trn"* "$SCRIPT_DIR/05-split-data/tst"* "$SCRIPT_DIR/05-split-data/val"*
python3 "$SCRIPT_DIR/05-split-data/split_data.py" "$SCRIPT_DIR/03-data/" "$SCRIPT_DIR/05-split-data/"


echo "Available config folders:"
select CFG_DIR in "$BASE_DIR_TRN"/mlp-*-cfg/; do
  if [[ -n "$CFG_DIR" ]]; then
    echo "Selected folder: $CFG_DIR"
    break
  else
    echo "Invalid selection, try again."
  fi
done

MAX_RUNS=6
count=0

for cfg in "$CFG_DIR"/mlp-*-*.json; do
  echo "-> Training config: $cfg"
  python3 "$SCRIPT_TRN" "$cfg" "$SPLIT" "$BASE_DIR_TRN"

  ((++count))
  if (( count >= MAX_RUNS )); then
    echo "Reached limit of $MAX_RUNS configs. Stopping."
    break
  fi
done


# ---- testing loop (NEW, matches your required call) ----
# ---- testing loop (FIXED) ----
# ---- testing loop (CSV files all go directly into 08-tst-mlp) ----
mkdir -p "$BASE_DIR_TST"

SRC_DIR="$SCRIPT_DIR/05-split-data"   # must contain tst/ inside it

cfg_dirs=( "$BASE_DIR_TRN"/mlp-*-cfg/ )
if [[ ${#cfg_dirs[@]} -eq 0 ]]; then
  echo "No cfg dirs found at: $BASE_DIR_TRN/mlp-*-cfg/"
  exit 1
fi

for cfg_dir in "${cfg_dirs[@]}"; do
  echo "=== CFG dir: $cfg_dir ==="

  cfgs=( "$cfg_dir"/mlp-*-*.json )
  if [[ ${#cfgs[@]} -eq 0 ]]; then
    echo "No json configs in: $cfg_dir"
    continue
  fi

  for cfg in "${cfgs[@]}"; do
    stem="$(basename "$cfg" .json)"

    run_dir="$BASE_DIR_TRN/$stem"
    model_pt="$run_dir/$stem.pt"
    norm_pt="$run_dir/$stem-norm.pt"

    # Put ALL outputs directly into 08-tst-mlp/
    dst="$BASE_DIR_TST"

    [[ -f "$cfg" ]]        || { echo "Missing cfg: $cfg"; exit 1; }
    [[ -f "$model_pt" ]]   || { echo "Missing model: $model_pt"; exit 1; }
    [[ -f "$norm_pt" ]]    || { echo "Missing norm: $norm_pt"; exit 1; }
    [[ -d "$SRC_DIR/tst" ]]|| { echo "Missing test dir: $SRC_DIR/tst"; exit 1; }

    echo "▶ Testing: $stem"
    python3 "$SCRIPT_TST" "$cfg" "$model_pt" "$norm_pt" "$SRC_DIR" "$dst"
  done
done


# ---- visualization loop ----
VIZ_SCRIPT="$SCRIPT_DIR/07-viz-trn-val/viz_trn_val.py"
VIZ_DST="$SCRIPT_DIR/07-viz-trn-val"
BASE_DIR_TRN="$SCRIPT_DIR/06-trn-val-mlp"

mkdir -p "$VIZ_DST"

run_dirs=( "$BASE_DIR_TRN"/mlp-*-*-*-* )
if [[ ${#run_dirs[@]} -eq 0 ]]; then
  echo "No training run dirs found in $BASE_DIR_TRN"
  exit 1
fi

for run_dir in "${run_dirs[@]}"; do
  losses=( "$run_dir"/*-losses.csv )
  if [[ ${#losses[@]} -eq 0 ]]; then
    echo "No losses CSV in $run_dir"
    continue
  fi

  for csv in "${losses[@]}"; do
    echo "Visualizing: $csv"
    python3 "$VIZ_SCRIPT" "$csv" "$VIZ_DST"
  done
done

deactivate
