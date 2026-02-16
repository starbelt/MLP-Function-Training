import json
import csv
import sys
from pathlib import Path

def analyze_mlp_config(json_path):
    with open(json_path, "r") as f:
        cfg = json.load(f)

    curr_in = int(cfg["in_features"])

    n_linear = 0
    n_relu = 0

    muls = 0
    adds = 0
    relu_comps = 0

    for layer in cfg["layers"]:
        cls = layer.get("class")

        if cls == "Linear":
            out = int(layer["out_features"])

            # Dense layer y = W x + b
            # muls: in*out
            # adds: (in-1)*out for dot-products + out for bias
            muls += curr_in * out
            adds += (curr_in - 1) * out + out  # bias add

            n_linear += 1
            curr_in = out

        elif cls == "ReLU":
            # elementwise max(0, x): ~1 comparison per element
            relu_comps += curr_in
            n_relu += 1

        else:
            # ignore other layer types for now (or raise if you prefer)
            pass

    total_flops = muls + adds + relu_comps

    return {
        "config": json_path.name,
        "in_features": int(cfg["in_features"]),
        "out_features": curr_in,
        "n_linear_layers": n_linear,
        "n_relu_layers": n_relu,
        "muls": muls,
        "adds": adds,
        "relu_comps": relu_comps,
        "total_flops": total_flops,
    }

def analyze_folder(cfg_dir, csv_out):
    cfg_dir = Path(cfg_dir)
    json_files = sorted(cfg_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {cfg_dir}")

    rows = [analyze_mlp_config(p) for p in json_files]

    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_mlp_folder.py CFG_DIR OUT.csv")
        sys.exit(1)

    analyze_folder(sys.argv[1], sys.argv[2])
