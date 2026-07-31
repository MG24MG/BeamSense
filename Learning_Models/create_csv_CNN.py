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
import os
import csv

# Set the station, will go through train, val, and test subfolders within station's folder
Test = "Livingroom"

data_pa = "/home/maria/Documents/BeamSense/Data/BFI/New_Processed"

data_path = os.path.join(data_pa, Test)

train_csv = os.path.join(data_path, "train_set.csv")
val_csv = os.path.join(data_path, "val_set.csv")
test_csv = os.path.join(data_path, "test_set.csv")

# I open output CSV files and write headers for each split.
train_csv = open(train_csv, "w", newline="")
val_csv = open(val_csv, "w", newline="")
test_csv = open(test_csv, "w", newline="")
fieldnames = ["filename", "label"]
writer_train = csv.DictWriter(train_csv, fieldnames=fieldnames)
writer_train.writeheader()
writer_val = csv.DictWriter(val_csv, fieldnames=fieldnames)
writer_val.writeheader()
writer_test = csv.DictWriter(test_csv, fieldnames=fieldnames)
writer_test.writeheader()

#sorts files within each csv based on label number
for split, writer in [("train", writer_train), ("val", writer_val), ("test", writer_test)]:
    split_path = os.path.join(data_path, split)
    for root, dirs, files in os.walk(split_path):
        for file in sorted(files, key=lambda x: int(x.split("_")[3])):
            filename = os.path.join(root, file)
            label = int(file.split("_")[3])
            writer.writerow({"filename": filename, "label": label})

train_csv.close()
val_csv.close()
test_csv.close()
