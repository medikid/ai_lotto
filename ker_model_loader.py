
from imodel import Model

import keras as kr
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

class KER_Model_Loader:
    
    def __init__(self, Model):
        Build = Model._INFO['BUILD'];
        Make = Model._INFO['MAKE']
        Version = Model._INFO['VERSION']
        k = Sequential();
        
        if(Build == '1'):
            k.add(LSTM(units=100, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) , return_sequences=True ));
            k.add(LSTM(units=100, activation='relu'))
            k.add(Dense(Model._D_Y_SHAPE[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy']);
            print("[KER_Model_loader: loaded build {0}]".format(Build))
        elif (Build == '2'):
            k.add(LSTM(units=100, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2])   ,  return_sequences=True ));
            k.add(LSTM(units=100, activation='relu'))
            k.add(Dense(Model._D_Y_SHAPE[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            print("[KER_Model_loader: loaded build {0}]".format(Build))
        elif (Build == '3'):
            k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2])   ,  return_sequences=True ));
            k.add(LSTM(units=160, activation='relu'))
            k.add(Dense(units=160, activation='relu'));
            k.add(Dense(Model._D_Y_SHAPE[1]));
            k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            print("[KER_Model_loader: loaded build {0}]".format(Build))
        elif (Build == '4'):
            if (Make == '1'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                print("[KER_Model_loader: loaded build {0}.{1}]".format(Build, Make))
            elif (Make == '2'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
                print("[KER_Model_loader: loaded build {0}.{1}]".format(Build, Make))
            elif (Make == '3'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
                #when u increase batchsize, decrease learning rate, so it learn slowly, take small step 
                print("[KER_Model_loader: loaded build {0}.{1}]".format(Build, Make))
        
        Model._M = k;
    
