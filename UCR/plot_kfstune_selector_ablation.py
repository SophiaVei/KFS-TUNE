import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from UCR.run_kfstune_selector_ablation import (
    SELECTOR_ORDER,
    complete_selector_cases,
    plot_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create transparent-background plots and a LaTeX table from existing "
            "KFS-TUNE selector-ablation results."
        )
    )
    parser.add_argument(
        "--results-csv",
        default="results/kfstune_selector_ablation/selector_ablation_results.csv",
        help="Path to selector_ablation_results.csv from run_kfstune_selector_ablation.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots/table. Defaults to the parent folder of --results-csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_csv = Path(args.results_csv)
    if not results_csv.exists():
        raise SystemExit(f"Results CSV not found: {results_csv}")

    output_dir = Path(args.output_dir) if args.output_dir else results_csv.parent
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise SystemExit(f"No ok rows found in {results_csv}")

    complete = complete_selector_cases(df)
    if complete.empty:
        raise SystemExit(
            "No complete datasets found. Each plotted dataset must have ok rows for "
            f"all selectors: {', '.join(SELECTOR_ORDER)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(results_csv, output_dir)

    print(f"Wrote summary: {output_dir / 'selector_ablation_summary.csv'}")
    print(f"Wrote complete cases: {output_dir / 'selector_ablation_complete_cases.csv'}")
    print(f"Wrote table: {output_dir / 'selector_ablation_table.tex'}")
    print(f"Wrote transparent PNG/PDF plots to: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
