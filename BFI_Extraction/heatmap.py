import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

files = sorted(glob.glob("/home/maria/Documents/BeamSense/Data/BFI/Processed/Livingroom/Shahriar/*_vmatrices.npy"))
output_dir = Path("/home/maria/Documents/BeamSense/Data/BFI/Heatmaps")
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

    v = np.abs(v)  # amplitude
    time_steps, num_freq, num_angles = v.shape
    print(f"Time: {time_steps}, Freq: {num_freq}, Angles: {num_angles}")

    angle_labels = [f'Receiving antenna {i+1}' for i in range(num_angles)]

    fig, axes = plt.subplots(num_angles, 1, figsize=(10, 3 * num_angles), sharex=True)
    if num_angles == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # v shape: (time, freq, angle) → heatmap needs (freq, time) so y=freq, x=time
        heatmap_data = v[:, :, i].T  # shape: (num_freq, time_steps), makes transposed

        im = ax.imshow(
            heatmap_data,

            # aspect='auto',

            origin='lower',          # freq axis starts from 0 at bottom

            extent=[0, time_steps, 0, num_freq],  # x=time, y=freq

            cmap='hot',
            interpolation='nearest'
        )

        plt.colorbar(im, ax=ax, label='Amplitude')
        ax.set_title(angle_labels[i], fontsize=10)
        ax.set_ylabel('Frequency (sub-channel)', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

    axes[-1].set_xlabel('Time (steps)', fontsize=10)
    plt.suptitle(Path(file).stem, fontsize=9, y=1.01)
    plt.tight_layout()

    save_path = output_dir / f"{Path(file).stem}_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
