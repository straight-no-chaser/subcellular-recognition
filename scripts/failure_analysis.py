import argparse
from pathlib import Path

import pandas as pd

from src.analysis.failure import save_failure_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--fields", nargs="+", default=["source", "acquisition_batch", "quality_flags", "organelle_label"])
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.pred_csv)
    out_dir = args.out_dir or str(Path(args.pred_csv).resolve().parent / "failure_analysis")
    save_failure_report(predictions, out_dir, top_k=args.top_k, fields=args.fields)
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
