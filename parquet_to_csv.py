import pyarrow.parquet as pq

def open_training_file():
    training_file = pq.ParquetFile("train-00000-of-00001-9564e8b05b4757ab.parquet")
    for index in training_file.iter_batches(batch_size=100000):
        df_training = index.to_pandas()
    return df_training

def open_test_file():
    test_file = pq.ParquetFile("test-00000-of-00001.parquet")
    for index in test_file.iter_batches(batch_size=8236):
        df_test = index.to_pandas()
    return df_test
def save_training_data(training_raw_dataset):
    training_raw_dataset.to_csv('trainingdata.csv')
    
def save_test_data(test_raw_dataset):
    test_raw_dataset.to_csv('testdata.csv')

training_raw_dataset = open_training_file()
test_raw_dataset = open_test_file()
save_training_data(training_raw_dataset)
save_test_data(test_raw_dataset)
