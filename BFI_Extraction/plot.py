#creates charts based on amount of angles, one subplot per angle
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

#where finds the file to be plotted
files = sorted(glob.glob("/home/Documents/BeamSense/Data/BFI/Processed/Livingroom/Shahriar/*_vmatrices.npy"))

#where the plots will be saved
output_dir = Path("/home/Documents/BeamSense/Data/BFI/Plots")
output_dir.mkdir(parents=True, exist_ok=True)

for file in files:
    print(f"Processing {file}")

    v = np.load(file, allow_pickle=True)
    v = np.array(v)

    print("Shape:", v.shape)

    if v.ndim == 4:
        v = v[:, :, :, 0]  # remove receiver dim -> (time, freq, N)

    if v.ndim != 3:
        print("Skipping unexpected shape:", v.shape)
        continue

    v = np.abs(v)  #fixes complex values

    time_steps, num_freq, num_angles = v.shape
    print(f"Time: {time_steps}, Freq: {num_freq}, Angles: {num_angles}")

    x = np.arange(num_freq)
    step = max(1, time_steps // 20)

    angle_labels = [f' Receiving antenna {i+1}' for i in range(num_angles)]  # dynamic, works for any count

    fig, axes = plt.subplots(num_angles, 1, figsize=(8, 2.5 * num_angles), sharex=True)

    if num_angles == 1:
        axes = [axes]  # ensure iterable if only 1 angle

    for i, ax in enumerate(axes):
        for t in range(0, time_steps, step):
            ax.plot(x, abs(v[t, :, i]), linewidth=0.6, alpha=0.6, color='black')
        ax.set_title(angle_labels[i], fontsize=10)
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=8)

    axes[-1].set_xlabel('Sub-channel', fontsize=10)

    plt.suptitle(Path(file).stem, fontsize=9, y=1.01)
    plt.tight_layout()

    save_path = output_dir / f"{Path(file).stem}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")