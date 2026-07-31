"""
plot_raw_vs_normalized_lines.py

Plots raw vs. normalized line graphs for EVERY file listed in the csv below,
using the actual normalization equation from dataGenerator_CNN.py:
    arr / np.abs(np.max(arr))

Just hit Run in PyCharm -- no configuration/arguments needed.
Edit CSV_PATH below to point at your train/val/test csv.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataGenerator_CNN import load_npy, normalize_arr, parse_label

CSV_PATH = "/home/maria/Documents/BeamSense/Data/BFI/New_Processed/Livingroom/train_set.csv"
OUTPUT_DIR = "raw_vs_normalized_plots_L"

def plot_one_file(path, out_dir):
    raw = load_npy(path)
    norm = normalize_arr(raw)
    label = parse_label(path) #+ 1?
    name = os.path.basename(path)

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
    axes[1].set_title("Normalized values (arr / |max(arr)|)")
    axes[1].set_xlabel("time step")
    axes[1].legend()

    plt.suptitle(f"{name}  (label {label})")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)  # close instead of show, since we're doing many files
    return out_path


def main():
    df = pd.read_csv(CSV_PATH)
    paths = df["filename"].tolist()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Plotting {len(paths)} files from {CSV_PATH}")
    for i, path in enumerate(paths):
        try:
            out_path = plot_one_file(path, OUTPUT_DIR)
            print(f"[{i + 1}/{len(paths)}] saved {out_path}")
        except Exception as e:
            print(f"[{i + 1}/{len(paths)}] FAILED on {path}: {e}")

    print(f"\nDone. All plots saved under: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

