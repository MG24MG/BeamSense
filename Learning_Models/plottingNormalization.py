import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataGenerator_CNN import load_npy, normalize_arr, compute_channel_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="train_set.csv (used both to compute stats and pick the file)")
    p.add_argument("--index", type=int, default=0, help="which row of the csv to plot")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    path = df["filename"].iloc[args.index]

    # channel stats computed from the whole training csv, same as train_gen would do
    channel_mean, channel_std = compute_channel_stats(df["filename"])

    raw = load_npy(path)
    norm = normalize_arr(raw, channel_mean, channel_std)

    print(f"File: {os.path.basename(path)}")
    print(f"RAW        min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f} std={raw.std():.4f}")
    print(f"NORMALIZED min={norm.min():.4f} max={norm.max():.4f} mean={norm.mean():.4f} std={norm.std():.4f}")

    n_channels = raw.shape[-1]
    time_steps = np.arange(raw.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for c in range(n_channels):
        axes[0].plot(time_steps, raw[:, :, c].mean(axis=1), label=f"channel {c}")
    axes[0].set_title("Raw values (before normalization)")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("value (avg over subcarriers)")
    axes[0].legend()

    for c in range(n_channels):
        axes[1].plot(time_steps, norm[:, :, c].mean(axis=1), label=f"channel {c}")
    axes[1].set_title("Normalized values (after z-score)")
    axes[1].set_xlabel("time step")
    axes[1].legend()

    plt.suptitle(os.path.basename(path))
    plt.tight_layout()
    plt.savefig("raw_vs_normalized_line_graph.png", dpi=200)
    print("Saved raw_vs_normalized_line_graph.png")
    plt.show()


if __name__ == "__main__":
    main()
