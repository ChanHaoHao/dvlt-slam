"""Consolidate TUM ATE results across configs into one recorded table.

Reads every logs/<config>/tum_results.txt produced by eval_tum_backbone.sh and
writes a combined CSV plus a Markdown table to outputs/tum_metrics.{csv,md}.

Failed sequences are recorded as NaN and excluded from means, with the count of
scored sequences reported alongside — a mean over 7 of 9 sequences is not
comparable to a mean over 9, so the denominator is always shown.

Usage:  python evals/summarize_results.py [--logs-dir logs] [--out-dir outputs]
"""

import argparse
import csv
import math
from pathlib import Path


SHORT = "rgbd_dataset_freiburg1_"


def load_config(results_file: Path) -> dict[str, float]:
    """Return {sequence: rmse}; non-numeric (NaN/failed) becomes float('nan')."""
    out: dict[str, float] = {}
    with results_file.open() as fh:
        for row in csv.DictReader(fh):
            try:
                val = float(row["RMSE"])
            except (ValueError, KeyError, TypeError):
                val = float("nan")
            out[row["Dataset"]] = val
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    logs = Path(args.logs_dir)
    configs = {p.parent.name: load_config(p) for p in sorted(logs.glob("*/tum_results.txt"))}
    if not configs:
        print(f"no results found under {logs}/*/tum_results.txt")
        return

    sequences = sorted({s for c in configs.values() for s in c})
    names = list(configs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "tum_metrics.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", *names])
        for seq in sequences:
            w.writerow([seq, *[configs[n].get(seq, float("nan")) for n in names]])

    def fmt(v: float) -> str:
        return "FAILED" if math.isnan(v) else f"{v:.4f}"

    lines = [
        "# TUM freiburg1 — ATE RMSE (m), evo_ape -as (aligned + scale corrected)",
        "",
        "| sequence | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for seq in sequences:
        short = seq[len(SHORT):] if seq.startswith(SHORT) else seq
        lines.append(f"| {short} | " + " | ".join(fmt(configs[n].get(seq, float('nan'))) for n in names) + " |")

    means, counts = [], []
    for n in names:
        vals = [v for v in configs[n].values() if not math.isnan(v)]
        means.append(f"{sum(vals)/len(vals):.4f}" if vals else "n/a")
        counts.append(f"{len(vals)}/{len(sequences)}")
    lines += [
        "| **mean** | " + " | ".join(f"**{m}**" for m in means) + " |",
        "| *scored* | " + " | ".join(f"*{c}*" for c in counts) + " |",
    ]

    # Only compare means computed over the same set of sequences.
    if len(names) > 1 and len(set(counts)) > 1:
        lines += ["", "> Means cover different sequence counts — not directly comparable."]

    md_path = out_dir / "tum_metrics.md"
    md_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
