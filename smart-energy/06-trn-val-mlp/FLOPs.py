import json
import csv
import sys
from pathlib import Path

def analyze_mlp_config(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)

    curr_in = cfg["in_features"]
    n_linear = 0
    adds = muls = divs = 0

    for layer in cfg["layers"]:
        if layer["class"] == "Linear":
            out = layer["out_features"]

            adds += curr_in * out
            muls += curr_in * out
            n_linear += 1

            curr_in = out

    return {
        "config": json_path.name,
        "in_features": cfg["in_features"],
        "out_features": curr_in,
        "n_linear_layers": n_linear,
        "adds": adds,
        "muls": muls,
        "divs": divs,
    }


def analyze_folder(cfg_dir, csv_out):
    cfg_dir = Path(cfg_dir)
    json_files = sorted(cfg_dir.glob("*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in {cfg_dir}")

    rows = []
    for json_path in json_files:
        rows.append(analyze_mlp_config(json_path))

    # Write CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    analyze_folder(
        cfg_dir="mlp-1024-cfg",
        csv_out="mlp_ops_summary.csv"
    )

if len(sys.argv) != 3:
    print("Usage: python3 analyze_mlp_folder.py CFG_DIR OUT.csv")
    sys.exit(1)

analyze_folder(sys.argv[1], sys.argv[2])