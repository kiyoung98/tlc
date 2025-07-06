#!/usr/bin/env python3
"""
plot_mlcv_hist.py

Load a PLUMED COLVAR file and plot the histogram of your chosen collective variable (MLCV).
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def load_colvar(path):
    """
    Read the 'FIELDS' header from a PLUMED COLVAR file, then load the data into a DataFrame.
    """
    fields = None
    with open(path) as f:
        for line in f:
            # look for the FIELDS directive
            if line.startswith("#! FIELDS"):
                # split: "#!","FIELDS", then the names
                fields = line.strip().split()[2:]
                break
    if fields is None:
        raise ValueError(f"No '#! FIELDS' line found in {path}")
    # now use pandas to read the file, skipping all '#' lines
    df = pd.read_csv(
        path,
        delim_whitespace=True,
        comment="#",
        names=fields,
        header=None,
    )
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--colvar",
                   help="path to PLUMED COLVAR file")
    p.add_argument("--cv", default="deep.node-0",
                   help="name of the CV column to histogram (default: deep.node-0)")
    p.add_argument("--bins", type=int, default=50,
                   help="number of histogram bins")
    p.add_argument("--out", default="./test.png",
                   help="output PNG file (if omitted, shows interactively)")
    args = p.parse_args()

    if not os.path.isfile(args.colvar):
        print(f"Error: file not found: {args.colvar}", file=sys.stderr)
        sys.exit(1)

    df = load_colvar(args.colvar)
    print(df.columns)

    values = df[args.cv].values

    # plot
    plt.figure(figsize=(6,4))
    plt.hist(values, bins=args.bins, density=True, alpha=0.7, edgecolor="k")
    plt.xlabel(args.cv)
    plt.ylabel("Probability density")
    plt.title(f"Histogram of {args.cv}")
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300)
        print(f"Saved histogram to {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()