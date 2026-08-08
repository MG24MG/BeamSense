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

import argparse
import os


# I keep a single window-size constant so the model input size stays consistent.
WINDOW = 10
WINDOW_SIZE = WINDOW

# The station folder names under data_path, each containing train_set.csv /
# val_set.csv / test_set.csv. Edit this list (or pass --stations) if your
# folder names differ.
DEFAULT_STATIONS = ["Classroom", "Kitchen", "Livingroom"]


def parse_args():
    # I parse the runtime parameters for a combined-station training run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_save", help="Name of the model")
    parser.add_argument(
        "--stations",
        nargs="+",
        default=DEFAULT_STATIONS,
        help="Station folder names to pool together for training and to test individually (default: %(default)s)",
    )
    return parser.parse_args()


def getBaselineModel2D(slice_size=WINDOW_SIZE, classes=20):
    from tensorflow.keras import layers, models

    # I define the baseline 2D CNN used for beamforming-angle classification.
    # Two paired conv layers per block with decreasing width, then Flatten into the
    # dense head. Pooling is (2, 1) so it only shrinks the time axis and preserves the
    # per-subcarrier resolution that distinguishes beamforming angles; Flatten keeps
    # that full spatial detail feeding the classifier.
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(slice_size, 234, 4),
        )
    )
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation="softmax"))

    model.summary()
    return model


if __name__ == "__main__":
    args = parse_args()

    stations = args.stations
    model_save = args.model_save
    window_size = WINDOW

    # I define dataset and output locations. Everything for this combined run
    # lives under a "Combined" folder instead of a single station's folder.
    data_path = "/home/maria/Documents/BeamSense/Data/BFI/New_Processed"
    data_proc = "Model"

    combined_out_dir = os.path.join(data_path, "Combined", data_proc)
    model_dir = os.path.join(combined_out_dir, model_save)
    os.makedirs(combined_out_dir, exist_ok=True)

    from tensorflow import keras

    # I build the CNN model for training. Same architecture as before -- only
    # the data feeding it changes.
    model = getBaselineModel2D(slice_size=window_size)
    model.summary()

    import numpy as np

    from dataGenerator_CNN import DataGenerator

    # I point each station at its own train/val/test csvs, then pool the
    # train and val csvs across all stations into single generators.
    train_csvs = [os.path.join(data_path, s, "train_set.csv") for s in stations]
    val_csvs = [os.path.join(data_path, s, "val_set.csv") for s in stations]
    test_csvs = {s: os.path.join(data_path, s, "test_set.csv") for s in stations}

    # One model trained on all stations combined.
    train_gen = DataGenerator(data_path, train_csvs, batchsize=32)
    val_gen = DataGenerator(data_path, val_csvs, batchsize=32)

    # Separate test generators per station -- this is what lets us see how
    # the single combined model performs in each individual environment.
    test_gens = {
        s: DataGenerator(data_path, csv, batchsize=32, shuffle=False)
        for s, csv in test_csvs.items()
    }

    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping

    # I configure training callbacks for LR scheduling, checkpointing, and early stopping.
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_loss",
        patience=6,
        verbose=1,
        factor=0.5,
        min_lr=0.00001,
    )

    checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)

    # can add early stopping if desired
    earlystopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=20,
        verbose=1,
        restore_best_weights=True,
    )

    # I compile and train the model.
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # to see, enter: tensorboard --logdir <combined_out_dir>/logs
    # histogram_freq=1 recomputes weight histograms for every layer every epoch
    # and is a major per-epoch slowdown; 0 disables it (loss/accuracy scalars are
    # still logged). write_graph is also disabled to skip serializing the graph.
    tensorboard = TensorBoard(
        log_dir=os.path.join(combined_out_dir, "logs"),
        histogram_freq=0,
        write_graph=False,
        update_freq="epoch",
    )

    history = model.fit(
        x=train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=[learning_rate_reduction, checkpoint, tensorboard, earlystopping],
        verbose=1,
    )

    from matplotlib import pyplot as plt

    # I plot and save training/validation loss over epochs.
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join(combined_out_dir, "train_val_loss.png"), dpi=300)
    plt.close()

    # I plot and save training/validation accuracy over epochs.
    plt.plot(history.history["accuracy"], label="Training acc")
    plt.plot(history.history["val_accuracy"], label="Validation acc")
    plt.legend()
    plt.savefig(os.path.join(combined_out_dir, "train_val_accuracy.png"), dpi=300)
    plt.close()

    print("The validation accuracy is :", history.history["val_accuracy"])
    print("The training accuracy is :", history.history["accuracy"])
    print("The validation loss is :", history.history["val_loss"])
    print("The training loss is :", history.history["loss"])

    from tensorflow.keras.models import load_model

    # I reload the best checkpoint (trained on all stations combined).
    model = load_model(model_dir)

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    labels = list(range(20))

    all_true = []
    all_pred = []

    # I evaluate the single combined model separately on each station's test
    # set, so we can see per-station generalization, not just an average.
    from sklearn.metrics import accuracy_score

    for station, gen in test_gens.items():
        # Single inference pass per station: get predictions once and derive the
        # accuracy from them, instead of running model.evaluate() and then
        # model.predict() (two full passes over the same test set).
        Y_pred = model.predict(gen)
        Y_pred = np.argmax(Y_pred, axis=1)

        Y_true = gen.labels[gen.indexes].astype(int)
        Y_true = Y_true[: len(Y_pred)]

        final_accuracy = accuracy_score(Y_true, Y_pred)
        print(f"[{station}] Final Accuracy: {final_accuracy}")

        all_true.append(Y_true)
        all_pred.append(Y_pred)

        cm = confusion_matrix(Y_true, Y_pred, normalize="true")

        plt.figure(figsize=(24, 24))
        ax = sns.heatmap(
            cm,
            cmap=plt.cm.Greens,
            annot=True,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_ylabel("Actual", fontsize=20)
        ax.set_xlabel("Predicted", fontsize=20)
        ax.set_title(f"Confusion Matrix - {station} (combined model)", fontsize=24)

        plt.savefig(os.path.join(combined_out_dir, f"confusion_matrix_{station}.png"), dpi=300)
        plt.close()

    # I also build one overall confusion matrix + accuracy across all three
    # stations' test sets combined, to see aggregate performance of the
    # single model (not just an average of the per-station numbers).
    Y_true_all = np.concatenate(all_true)
    Y_pred_all = np.concatenate(all_pred)

    overall_accuracy = accuracy_score(Y_true_all, Y_pred_all)
    print(f"[ALL STATIONS COMBINED] Overall Accuracy: {overall_accuracy}")

    cm_all = confusion_matrix(Y_true_all, Y_pred_all, normalize="true")

    plt.figure(figsize=(24, 24))
    ax = sns.heatmap(
        cm_all,
        cmap=plt.cm.Greens,
        annot=True,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_ylabel("Actual", fontsize=20)
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_title("Confusion Matrix - All Stations Combined", fontsize=24)

    plt.savefig(os.path.join(combined_out_dir, "confusion_matrix_all_stations.png"), dpi=300)
    plt.close()
