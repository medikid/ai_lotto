
from imodel import Model

import keras as kr
from keras.models import Sequential
from keras.layers import LSTM, Dense

class KER_Model_Loader:
    
    def __init__(self, Model):
        Build = Model._INFO['BUILD'];
        Make = Model._INFO['MAKE']
        Version = Model._INFO['VERSION']
        k = Sequential();
        
        if(Build == 1):
            k.add(LSTM(units=100, activation='relu', input_shape=(dataset['X'].shape[1], dataset['X'].shape[2]) , return_sequences=True ));
            k.add(LSTM(units=100, activation='relu'))
            k.add(Dense(dataset['Y'].shape[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy']);
            
        elif (Build == 2):
            k.add(LSTM(units=100, activation='relu', input_shape=(dataset['X'].shape[1], dataset['X'].shape[2]) , return_sequences=True ));
            k.add(LSTM(units=100, activation='relu'))
            k.add(Dense(dataset['Y'].shape[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            
        elif (Build == 3):
            k.add(LSTM(units=80, activation='relu', input_shape=(dataset['X'].shape[1], dataset['X'].shape[2]) , return_sequences=True ));
            k.add(LSTM(units=160, activation='relu'))
            k.add(Dense(units=160, activation='relu'));
            k.add(Dense(dataset['Y'].shape[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            
        elif (Build == 4):
            k.add(LSTM(units=80, activation='relu', input_shape=(dataset['X'].shape[1], dataset['X'].shape[2]) , return_sequences=True ));
            k.add(LSTM(units=160, activation='relu'))
            k.add(Dense(units=160, activation='relu'));
            k.add(Dense(dataset['Y'].shape[1], activation='sigmoid'));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        
        return k;
            
                
        
