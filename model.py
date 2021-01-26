from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.layers import Dropout,Flatten
from keras.layers import Dropout,Flatten


def NN():
    model = Sequential()

    model.add(Dense(units=1000,input_dim=8, kernel_initializer='normal',activation='relu')) #輸入層跟隱藏層
    model.add(Dropout(0.25))  
    model.add(Dense(units=100,kernel_initializer='normal',activation='relu')) #輸入層跟隱藏層
    model.add(Dropout(0.25))  
    model.add(Dense(units=3 ,kernel_initializer='normal',activation='softmax'))#輸出層
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model 

def RNN():
    model = Sequential()
    
    model.add(LSTM(units = 500, return_sequences = True, input_shape=(1, 8)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 500, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 500, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(3))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

