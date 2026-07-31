"""
    Copyright (C) 2023 Khandaker Foysal Haque
    contact: haque.k@northeastern.edu
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from tensorflow import keras
import pandas as pd
import os

# I keep the default temporal window size used by the data generator.
window_size = 10

def load_npy(path):
    # I load a single .npy capture and normalize it to a (time, 234, 4) array.
    angle_data = np.load(path, allow_pickle=True)
    if angle_data.ndim == 4:
        angle_data = np.squeeze(angle_data[0, :, :, :])
    return angle_data.astype(np.float32)

def normalize_arr(arr):
    """Per-file max-abs scaling: divides the whole array by the absolute
    value of its own max. Scales values to roughly [-1, 1] (or [0, 1] if
    all values are non-negative)."""
    return arr / np.abs(np.max(arr))

def parse_label(path):
    # I parse the angle class (A_01..A_20) from the basename and make it 0-indexed.
    base = os.path.basename(path)
    return int(base.split("_")[3]) - 1

class DataGenerator(keras.utils.Sequence):
    """Data generator that yields sliding windows over each capture."""

    def __init__(
        self,
        dataset_path,
        dataset_csv,
        num_classes=20,
        chunk_shape=(window_size, 234, 4),
        batchsize=64,
        shuffle=True,
        to_categorical=True,
        window_stride=2,
    ):
        """Initialization
        param:
            dataset_path: the directory the .npy files live under
            dataset_csv: the csv file listing files and labels
            num_classes: number of classes
            chunk_shape: shape of one window (time x subcarriers x channels)
            window_stride: step (in time steps) between consecutive windows
        """
        df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]
        self.to_categorical = to_categorical
        self.window_stride = window_stride

        # I load each capture once and enumerate every sliding window as its own sample.
        # This multiplies the usable sample count from time steps that were previously discarded.
        self.cache = {}
        self.samples = []  # list of (path, window_start)
        sample_labels = []
        for path in df["filename"]:
            try:
                arr = load_npy(path)
            except Exception:
                continue
            self.cache[path] = normalize_arr(arr)
            label = parse_label(path)
            T = arr.shape[0]
            if T < self.windowsize:
                starts = [0]  # short captures are zero-padded into a single window
            else:
                starts = list(range(0, T - self.windowsize + 1, self.window_stride))
            for s in starts:
                self.samples.append((path, s))
                sample_labels.append(label)

        # I keep `labels`/`indexes` so downstream code (e.g. confusion matrix) stays aligned.
        self.labels = np.array(sample_labels, dtype=int)
        self.indexes = np.arange(len(self.labels))
        np.random.shuffle(self.indexes)
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch."""
        return int(np.floor(len(self.labels) / self.batchsize))

    def __getitem__(self, idx):
        """Generate one batch of data."""
        indexes = self.indexes[idx * self.batchsize:(idx + 1) * self.batchsize]
        return self.__load_batch(indexes)

    def on_epoch_end(self):
        """Reshuffle window order after each epoch when shuffling is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __load_batch(self, indexes):
        """Read one batch of windows from the in-memory cache."""
        batch_data = np.empty((self.batchsize, self.windowsize, self.length, self.height), dtype=np.float32)
        batch_label = np.empty(self.batchsize, dtype=int)
        for i, k in enumerate(indexes):
            path, start = self.samples[k]
            arr = self.cache[path]
            window = arr[start:start + self.windowsize]
            if window.shape[0] < self.windowsize:
                pad = np.zeros((self.windowsize - window.shape[0], self.length, self.height), dtype=np.float32)
                window = np.concatenate([window, pad])
            batch_data[i] = window
            batch_label[i] = self.labels[k]
        if self.to_categorical:
            batch_label = keras.utils.to_categorical(batch_label, num_classes=self.num_classes)
        return batch_data, batch_label