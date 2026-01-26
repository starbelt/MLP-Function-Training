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

: <<"comment"
# ---- optional training pipeline (as you had it) ----
rm -rf "$SCRIPT_DIR/03-data/"*
rm -f  "$SCRIPT_DIR/02-gen-data/data-cfg.json"
python3 "$SCRIPT_DIR/02-gen-data/gen_data.py" "$SCRIPT_DIR/02-gen-data/test.csv" "$SCRIPT_DIR/03-data"

rm -rf "$SCRIPT_DIR/05-split-data/trn"* "$SCRIPT_DIR/05-split-data/tst"* "$SCRIPT_DIR/05-split-data/val"*
python3 "$SCRIPT_DIR/05-split-data/split_data.py" "$SCRIPT_DIR/03-data/" "$SCRIPT_DIR/05-split-data/"

for cfg_dir in "$BASE_DIR_TRN"/mlp-*-cfg/; do
  echo "=== Training folder: $cfg_dir ==="
  for cfg in "$cfg_dir"/mlp-*-*.json; do
    echo "-> Training config: $cfg"
    python3 "$SCRIPT_TRN" "$cfg" "$SPLIT" "$BASE_DIR_TRN"
  done
done
comment

# ---- testing loop (NEW, matches your required call) ----
# ---- testing loop (FIXED) ----
# ---- testing loop (CSV files all go directly into 08-tst-mlp) ----
mkdir -p "$BASE_DIR_TST"

SRC_DIR="$SCRIPT_DIR/05-split-data"   # must contain tst/ inside it

cfg_dirs=( "$BASE_DIR_TRN"/mlp-*-cfg/ )
if [[ ${#cfg_dirs[@]} -eq 0 ]]; then
  echo "❌ No cfg dirs found at: $BASE_DIR_TRN/mlp-*-cfg/"
  exit 1
fi

for cfg_dir in "${cfg_dirs[@]}"; do
  echo "=== CFG dir: $cfg_dir ==="

  cfgs=( "$cfg_dir"/mlp-*-*.json )
  if [[ ${#cfgs[@]} -eq 0 ]]; then
    echo "⚠️  No json configs in: $cfg_dir"
    continue
  fi

  for cfg in "${cfgs[@]}"; do
    stem="$(basename "$cfg" .json)"

    run_dir="$BASE_DIR_TRN/$stem"
    model_pt="$run_dir/$stem.pt"
    norm_pt="$run_dir/$stem-norm.pt"

    # Put ALL outputs directly into 08-tst-mlp/
    dst="$BASE_DIR_TST"

    [[ -f "$cfg" ]]        || { echo "❌ Missing cfg: $cfg"; exit 1; }
    [[ -f "$model_pt" ]]   || { echo "❌ Missing model: $model_pt"; exit 1; }
    [[ -f "$norm_pt" ]]    || { echo "❌ Missing norm: $norm_pt"; exit 1; }
    [[ -d "$SRC_DIR/tst" ]]|| { echo "❌ Missing test dir: $SRC_DIR/tst"; exit 1; }

    echo "▶ Testing: $stem"
    python3 "$SCRIPT_TST" "$cfg" "$model_pt" "$norm_pt" "$SRC_DIR" "$dst"
  done
done



deactivate
