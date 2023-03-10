from tests.test_base import write_result

from taxifare_model.ml_logic.data import clean_data

from taxifare_model.ml_logic.params import (CHUNK_SIZE,
                                            DATA_RAW_DTYPES_OPTIMIZED,
                                            DATA_PROCESSED_DTYPES_OPTIMIZED,
                                            DATA_RAW_COLUMNS,
                                            DATASET_SIZE,
                                            VALIDATION_DATASET_SIZE,
                                            ROOT_PATH)

from taxifare_model.ml_logic.preprocessor import preprocess_features

from taxifare_model.ml_logic.model import (initialize_model,
                                           compile_model,
                                           train_model)

from taxifare_model.ml_logic.registry import (save_model,
                                              load_model)

import numpy as np
import pandas as pd
import os


def preprocess_and_train():

    data_raw_path = os.path.join(ROOT_PATH, "data", "raw", f"train_{DATASET_SIZE}.csv")
    data = pd.read_csv(data_raw_path, dtype=DATA_RAW_DTYPES_OPTIMIZED)

    data_cleaned = clean_data(data)

    X = data_cleaned.drop("fare_amount", axis=1)
    y = data_cleaned[["fare_amount"]]


    X_processed = preprocess_features(X)

    model = None
    learning_rate = 0.001
    batch_size = 256
    model = initialize_model(X_processed)
    model = compile_model(model, learning_rate)
    model, history = train_model(model, X_processed, y, batch_size, validation_split=0.3)


    metrics = dict(val_mae=None)
    metrics = dict(val_mae=np.min(history.history['val_mae']))


    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size)
    save_model(model, params=params, metrics=metrics)

    write_result(name="test_preprocess_and_train", subdir="train_at_scale", metrics=metrics)



def preprocess(training_set=True):

    if training_set:
        source_name = f"train_{DATASET_SIZE}.csv"
        destination_name = f"train_processed_{DATASET_SIZE}.csv"
    else:
        source_name = f"val_{VALIDATION_DATASET_SIZE}.csv"
        destination_name = f"val_processed_{VALIDATION_DATASET_SIZE}.csv"

    data_raw_path = os.path.abspath(os.path.join(
        ROOT_PATH, "data", "raw", source_name))
    data_processed_path = os.path.abspath(os.path.join(
        ROOT_PATH, "data", "processed", destination_name))

    chunk_id = 0

    while (True):
        print(f"processing chunk n??{chunk_id}...")


        one_if_first_chunk = 1 if chunk_id == 0 else 0

        try:
            data_chunk = pd.read_csv(
                    data_raw_path,
                    header=None, # ignore headers
                    skiprows=(chunk_id * CHUNK_SIZE) + one_if_first_chunk, # first chunk has headers
                    nrows=CHUNK_SIZE,
                    dtype=DATA_RAW_DTYPES_OPTIMIZED,
                    )

            data_chunk.columns = DATA_RAW_COLUMNS

        except pd.errors.EmptyDataError:
            data_chunk = None  


        if data_chunk is None:
            break

        data_chunk_cleaned = clean_data(data_chunk)
        if len(data_chunk_cleaned) ==0:
            break



        X_chunk = data_chunk_cleaned.drop("fare_amount", axis=1)
        y_chunk = data_chunk_cleaned[["fare_amount"]]


        X_processed_chunk = preprocess_features(X_chunk)
        data_processed_chunk = pd.DataFrame(
            np.concatenate((X_processed_chunk, y_chunk), axis=1))


        data_processed_chunk.to_csv(data_processed_path,
                mode="w" if chunk_id==0 else "a",
                header=chunk_id==0,
                index=False)

        chunk_id += 1

    if training_set:
        data_processed = pd.read_csv(data_processed_path, header=None, dtype=DATA_PROCESSED_DTYPES_OPTIMIZED).to_numpy()
        write_result(name="test_preprocess", subdir="train_at_scale",
                    data_processed_head=data_processed[0:2])



def train():

 
    path = os.path.abspath(os.path.join(
        ROOT_PATH, "data", "processed", f"val_processed_{VALIDATION_DATASET_SIZE}.csv"))

    data_val_processed = pd.read_csv(
        path,
        header=None,
        dtype=DATA_PROCESSED_DTYPES_OPTIMIZED
        ).to_numpy()

    X_val = data_val_processed[:, :-1]
    y_val = data_val_processed[:, -1]


    model = None
    chunk_id = 0
    metrics_val_list = []  # store each metrics_val_chunk

    while (True):
        print(f"loading and training on preprocessed chunk n??{chunk_id}...")



        path = os.path.abspath(os.path.join(
            ROOT_PATH, "data", "processed", f"train_processed_{DATASET_SIZE}.csv"))

        try:
            data_processed_chunk = pd.read_csv(
                    path,
                    header=None,
                    skiprows=(chunk_id * CHUNK_SIZE),
                    nrows=CHUNK_SIZE,
                    dtype=DATA_PROCESSED_DTYPES_OPTIMIZED,
                    ).to_numpy()

        except pd.errors.EmptyDataError:
            data_processed_chunk = None  # end of data


        if data_processed_chunk is None:
            break

        X_train_chunk = data_processed_chunk[:, :-1]
        y_train_chunk = data_processed_chunk[:, -1]



        learning_rate = 0.001
        batch_size = 256

        if model is None:
            model = initialize_model(X_train_chunk)
            model = compile_model(model, learning_rate)

        model, history = train_model(model,
                                     X_train_chunk,
                                     y_train_chunk,
                                     batch_size,
                                     validation_data=(X_val, y_val))
        metrics_val_chunk = np.min(history.history['val_mae'])
        metrics_val_list.append(metrics_val_chunk)
        print(metrics_val_chunk)


        chunk_id += 1

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        incremental=True,
        chunk_size=CHUNK_SIZE)


    metrics_val_mean_all_chunks = None

    metrics_val_mean_all_chunks = np.mean(np.array(metrics_val_list))

    metrics = dict(mean_val=metrics_val_mean_all_chunks)

    save_model(model, params=params, metrics=metrics)

  
    write_result(name="test_train", subdir="train_at_scale",
                 metrics=metrics)



    pass


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    if X_pred is None:

        X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[-73.950655],
            pickup_latitude=[40.783282],
            dropoff_longitude=[-73.984365],
            dropoff_latitude=[40.769802],
            passenger_count=[1]))

    model = load_model()

    X_processed = preprocess_features(X_pred)


    y_pred = model.predict(X_processed)


    write_result(name="test_pred", subdir="train_at_scale", y_pred=y_pred)

    return y_pred
