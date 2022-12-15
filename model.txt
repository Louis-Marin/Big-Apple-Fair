from colorama import Fore, Style

import time
start = time.perf_counter()

from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

end = time.perf_counter()

from typing import Tuple

import numpy as np


def initialize_model(X: np.ndarray) -> Model:

    reg = regularizers.l1_l2(l2=0.005)

    model = Sequential()
    model.add(layers.BatchNormalization(input_shape=X.shape[1:]))
    model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(50, activation="relu", kernel_regularizer=reg))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.BatchNormalization(momentum=0.99))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Dense(1, activation="linear"))

    return model


def compile_model(model: Model, learning_rate: float) -> Model:

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])
    return model


def train_model(model: Model,
                X: np.ndarray,
                y: np.ndarray,
                batch_size=256,
                validation_split=0.3,
                validation_data=None) -> Tuple[Model, dict]:

    print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

    es = EarlyStopping(monitor="val_loss",
                       patience=2,
                       restore_best_weights=True,
                       verbose=0)

    history = model.fit(X,
                        y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=0)

    return model, history

