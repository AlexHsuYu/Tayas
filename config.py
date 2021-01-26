import os, keras
from keras.callbacks import EarlyStopping

class Config():
    
    earlyStopping = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=0, 
    verbose=0, 
    mode='auto')

    index_path = './datasets/index.csv'
    
    # test_size = 0.2

    value_start = 0
    value_end = 120000
    # value_resample = 10

    NN_epochs = 6
    NN_batch_size = 500

    sliding_window = 10000
    RNN_value_start = 10000
    RNN_value_end = 480000

    RNN_epochs = 5
    RNN_batch_size = 500
    # RNN_value_resample = 35