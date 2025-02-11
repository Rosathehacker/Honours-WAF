import pandas as pd
import numpy as np
import pyarrow.parquet as pq

def open_training_file():
    training_file = pq.ParquetFile("train-00000-of-00001.parquet")
    for index in training_file.iter_batches(batch_size=8236):
        df_training = index.to_pandas()
    return df_training

def open_test_file():
    test_file = pq.ParquetFile("test-00000-of-00001.parquet")
    for index in test_file.iter_batches(batch_size=8236):
        df_test = index.to_pandas()
    return df_test

training_raw_dataset = open_training_file()
test_raw_dataset = open_test_file()
