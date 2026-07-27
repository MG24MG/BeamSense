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

import tensorboard

# I keep a single window-size constant so the model input size stays consistent.
WINDOW = 10
WINDOW_SIZE = WINDOW


def parse_args():
    # I parse the three required runtime parameters for this training run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("Test", help="Testing Location")
    parser.add_argument("station", help="Name of the Station")
    parser.add_argument("model_save", help="Name of the model")
    return parser.parse_args()


def getBaselineModel2D(slice_size=WINDOW_SIZE, classes=20):
    from tensorflow.keras import layers, models, regularizers

    # Three conv blocks with growing width give the model enough capacity to fit the
    # data now that overfitting is under control. Pooling is (2, 1) so it only shrinks
    # the time axis and preserves subcarrier resolution, which is what distinguishes
    # beamforming angles. GlobalAveragePooling keeps the dense head tiny.
    l2 = regularizers.l2(1e-4)
    model = models.Sequential()
    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=l2,
            input_shape=(slice_size, 234, 4),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(classes, activation="softmax"))

    model.summary()
    return model


if __name__ == "__main__":
    # I read runtime arguments and keep local names used throughout the script.
    args = parse_args()

    station = args.station
    Test = args.Test #BFI
    model_save = args.model_save
    window_size = WINDOW

    import os

    # I define dataset and output locations based on the selected test/station setup.
    data_path = "/home/maria/Documents/BeamSense/Data/BFI/New_Processed"
    data_proc = "Model"

    model_dir = os.path.join(data_path, Test, data_proc, model_save)
    data_dir = os.path.join(data_path, Test)

    from tensorflow import keras

    # I build the CNN model for training.
    model = getBaselineModel2D(slice_size=window_size)
    model.summary()
    import numpy as np

    from dataGenerator_CNN import DataGenerator

    # I load train/validation/test splits through the custom data generator.
    tr_csv = os.path.join(data_dir, "train_set.csv")
    val_csv = os.path.join(data_dir, "val_set.csv")
    test_csv = os.path.join(data_dir, "test_set.csv")

    train_gen = DataGenerator(data_dir, tr_csv, batchsize=32)
    val_gen = DataGenerator(data_dir, val_csv, batchsize=32)
    test_gen = DataGenerator(data_dir, test_csv, batchsize=32, shuffle=False)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    # I configure training callbacks for LR scheduling, checkpointing, and early stopping.
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_loss",
        patience=6,
        verbose=1,
        factor=0.5,
        min_lr=0.00001,
    )

    checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)
   # earlystopping = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1) # change to 0.001 for overfitting

    # I compile and train the model.
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # to see, enter: tensorboard --logdir /home/maria/Documents/BeamSense/Data/BFI/New_Processed/Classroom/Model/logs
    tensorboard = TensorBoard(
        log_dir=os.path.join(data_path, Test, data_proc, "logs"),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch"
    )

    history = model.fit(
        x=train_gen,
        epochs=100,
        validation_data=val_gen,
        callbacks=[learning_rate_reduction, checkpoint, tensorboard], #include earlystopping when putting back
        verbose=1,
    )



    from matplotlib import pyplot as plt

    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.legend()
    plt.savefig(os.path.join(data_path, Test, data_proc, "train_val_loss.png"),dpi=300)
    plt.show()


    # I plot and save training/validation accuracy over epochs.
    plt.plot(history.history["accuracy"], label="Training acc")
    plt.plot(history.history["val_accuracy"], label="Validation acc")
    plt.legend()
    plt.savefig(os.path.join(data_path, Test, data_proc, "train_val_accuracy.png"), dpi=300)
    plt.show()

    print("The validation accuracy is :", history.history["val_accuracy"])
    print("The training accuracy is :", history.history["accuracy"])
    print("The validation loss is :", history.history["val_loss"])
    print("The training loss is :", history.history["loss"])

    from tensorflow.keras.models import load_model

    # I reload the best checkpoint and evaluate it on the held-out test set.
    model = load_model(model_dir)
    final_loss, final_accuracy = model.evaluate(test_gen)
    print("Final Loss: {}, Final Accuracy: {}".format(final_loss, final_accuracy))

    labels = list(range(20))

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # I generate predictions and convert labels to numeric indices for the confusion matrix.
    Y_pred = model.predict(test_gen)
    Y_pred = np.argmax(Y_pred, axis=1)

    Y = test_gen.labels[test_gen.indexes]

    Y_true = Y.astype(int)
    print(Y_true)

    cm = confusion_matrix(Y_true[: len(Y_pred)], Y_pred, normalize="true")

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
    # I save the normalized confusion matrix for this run.
    plt.savefig(os.path.join(data_path, Test, data_proc, "confusion_matrix.png"), dpi=300)
