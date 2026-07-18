"""Command-line interface.

    ionpred species <checkpoint>
        List the species fields a checkpoint offers as targets.

    ionpred run <checkpoint> --species si3p [--model gbm] [--out results/]
        Train and evaluate on one checkpoint with a spatial hold-out.
"""

from __future__ import annotations

import argparse
import json
import sys

from .io import discover_species
from .pipeline import run_single_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ionpred",
        description="Predict ion abundances in FLASH MHD simulations")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_species = sub.add_parser(
        "species", help="list species fields in a checkpoint")
    p_species.add_argument("checkpoint")

    p_run = sub.add_parser(
        "run", help="train/evaluate on one checkpoint (spatial hold-out)")
    p_run.add_argument("checkpoint")
    p_run.add_argument("--species", required=True,
                       help="FLASH field name, e.g. 'si3p' or 'sip '")
    p_run.add_argument("--model", default="gbm",
                       choices=["ridge", "gbm", "nn"])
    p_run.add_argument("--split", default="median",
                       choices=["median", "percentile", "thirds"])
    p_run.add_argument("--hurdle", default="auto",
                       choices=["auto", "on", "off"])
    p_run.add_argument("--sample", type=int, default=500_000,
                       help="max cells to use (0 = all)")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--out", default=None,
                       help="output directory (must not already contain "
                            "results; nothing is ever overwritten)")

    args = parser.parse_args(argv)

    if args.cmd == "species":
        for field, label in discover_species(args.checkpoint).items():
            print(f"  {field!r}  {label}")
        return 0

    if args.cmd == "run":
        results = run_single_checkpoint(
            args.checkpoint,
            species=args.species if len(args.species) == 4
            else args.species.ljust(4),
            model=args.model, split=args.split, out_dir=args.out,
            sample=args.sample, hurdle=args.hurdle, seed=args.seed)
        json.dump(results, sys.stdout, indent=2, default=float)
        print()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
