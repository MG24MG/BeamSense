import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

files = sorted(glob.glob("/home/maria/Documents/BeamSense/Data/BFI/Processed/Livingroom/Shahriar/*_vmatrices.npy"))

output_dir = Path("/home/maria/Documents/BeamSense/Data/BFI/Plots")
output_dir.mkdir(parents=True, exist_ok=True)

for file in files:
    print(f"Processing {file}")

    v = np.load(file, allow_pickle=True)
    v = np.array(v)

    print("Shape:", v.shape)

    # Expect: (time, frequency, 1) OR (time, frequency)
    if v.ndim == 4:
        v = v[:, :, :, 0]  # remove receiver dim

    if v.ndim != 3:
        print("Skipping unexpected shape")
        continue

    # maybe include maybe not
    v = np.abs(v)

    time_steps, num_freq = v.shape

    plt.figure(figsize=(10, 5))

    x = range(num_freq)

    # subsample time to avoid clutter
    step = max(1, time_steps // 20)

    for t in range(0, time_steps, step):
        plt.plot(x, v[t, :], linewidth=0.8, alpha=0.7)

    plt.title(Path(file).stem)
    plt.xlabel("Subcarrier (Frequency)")
    plt.ylabel("Value")
    plt.grid()

    save_path = output_dir / f"{Path(file).stem}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import glob
#
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
#
# files = sorted(glob.glob("/home/maria/Documents/BeamSense/Data/BFI/Processed/Livingroom/Shahriar/*_vmatrices.npy")) # something wrong here prob
#
# all_angles = []
#
# for file in files:
#     # try:
#     #
#     # except:
#     #     break
#     print(f"Loading {file}")
#     angles = np.load(file, allow_pickle=True)
#
#     print("Shape:", angles.shape)
#
#     all_angles.append(angles)
#
# angle = np.stack(all_angles, axis=0)
#
# print("Combined shape:", angle.shape)
#
#
# angles_mean = np.mean(angle, axis=1)
#
# x = range(angles_mean.shape[0])
#
# fig, ax = plt.subplots(5, 1, figsize=(10, 10))
#
# labels = ['Angle 1', 'Angle 2', 'Angle 3', 'Angle 4']
#
# for i in range(4):
#     ax[i].plot(x, angles_mean[:, i], linewidth=1)
#     ax[i].set_title(labels[i], fontsize=10)
#     ax[i].set_ylabel('Value', fontsize=8)
#     ax[i].grid()
#     ax[i].tick_params(axis='both', labelsize=8)
#     ax[i].set_xticklabels([])
#
# subchannel = np.std(angle, axis=2).mean(axis=1)
#
# ax[4].plot(x, subchannel, linewidth=1)
# ax[4].set_title("Sub-channel", fontsize=10)
# ax[4].set_xlabel('Packet Index (Time)', fontsize=10)
# ax[4].set_ylabel('Value', fontsize=8)
# ax[4].grid()
# ax[4].tick_params(axis='both', labelsize=8)
#
# plt.tight_layout()
# plt.savefig('beamSense_graph.png', dpi=300)
# plt.show()

## og code
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
#
#
# angles= np.load("/home/maria/Documents/BeamSense/Data/BFI/Processed/Livingroom/Shahriar/*.npy")
# #angles= np.load("bfa/bfa_ax_su_4x2_160.npy")
# #print(np.shape(angles))
#
# angle = angles[:1, :, :]
#
#
# x = range(angle.shape[1])
#
# # Create a figure and axis object using matplotlib
# fig, ax = plt.subplots(2, 1, figsize=(8,8))
#
# # Loop through the first dimension of the array (axis 0)
# for i in range(angle.shape[0]):
#     # Plot the slice angle[i, :, j]
#     ax[0].plot(x, angle[i, :, 0].flatten(), label='$\phi$\u2081\u2081', linewidth=4.0)
#
# for i in range(angle.shape[0]):
#     # Plot the slice angle[i, :, j]
#     ax[1].plot(x, angle[i, :, 1].flatten(), label='$\phi$\u2082\u2081', linewidth=4.0)
#
# # Add labels and title to the plot
# ax[0].set_xlabel('Sub-channel', fontsize=40)
# ax[1].set_xlabel('Sub-channel', fontsize=40)
#
# ax[0].set_ylabel('Value', fontsize=40)
# ax[1].set_ylabel('Value', fontsize=40)
#
# #ax.set_title('Plotting a numpy array')
# ax[0].grid()
# ax[1].grid()
#
# # Add a legend to differentiate the slices
# ax[0].legend(fontsize='35', loc='lower left')
# ax[1].legend(fontsize='35', loc='lower left')
#
# ax[0].tick_params(axis='x', labelsize=35)  # Fontsize of x-axis tick labels
# ax[1].tick_params(axis='x', labelsize=35)  # Fontsize of x-axis tick labels
#
# ax[0].tick_params(axis='y', labelsize=35)  # Fontsize of y-axis tick labels
# ax[1].tick_params(axis='y', labelsize=35)  # Fontsize of y-axis tick labels
#
# plt.tight_layout()
#
# plt.savefig('angle.png', dpi=300, format='png')
# # Show the plot
# plt.show()