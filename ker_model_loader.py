
import keras as kr
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from imodel import Model
from custom_functions import cf_metrics, cf_losses




class KER_Model_Loader:
    
    def __init__(self, Model):
        Build = Model._INFO['BUILD'];
        Make = Model._INFO['MAKE']
        Version = Model._INFO['VERSION']
        k = Sequential();
        
        if(Build == '1'):            
            if (Make == '1'):
                k.add(LSTM(units=100, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) , return_sequences=True ));
                k.add(LSTM(units=100, activation='relu'))
                k.add(Dense(Model._D_Y_SHAPE[1]));
                k.compile(optimizer='adam', loss='mse', metrics=['accuracy']);
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
        elif (Build == '2'):            
            if (Make == '1'):
                k.add(LSTM(units=100, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2])   ,  return_sequences=True ));
                k.add(LSTM(units=100, activation='relu'))
                k.add(Dense(Model._D_Y_SHAPE[1]));
                k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
        elif (Build == '3'):            
            if (Make == '1'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2])   ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1]));
                k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
        elif (Build == '4'):
            if (Make == '1'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '2'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '3'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '4'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['categorical_accuracy'])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '5'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['binary_accuracy'])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '6'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Recall(top_k=20, name='recall_top20')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '7'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Precision(top_k=20, name='precision_top20')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '8'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Recall(top_k=15, name='recall_top15')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '9'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Precision(top_k=15, name='precision_top15')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '10'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Recall(top_k=10, name='recall_top10')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '11'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[kr.metrics.Precision(top_k=10, name='precision_top10')])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
        elif (Build == '5'): #experimentation, will use custom functions            
            if (Make == '1'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[cf_metrics.recall_a])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '2'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[cf_metrics.precision_a])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))
            elif (Make == '3'):
                k.add(LSTM(units=80, activation='relu', input_shape=(Model._D_X_SHAPE[1],Model._D_X_SHAPE[2]) ,  return_sequences=True ));
                k.add(LSTM(units=160, activation='relu'))
                k.add(Dense(units=160, activation='relu'));
                k.add(Dense(Model._D_Y_SHAPE[1], activation='sigmoid'));
                k.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=[cf_metrics.f1_a])
                #when u increase batchsize, decrease learning rate, so it learn slowly, takes small step 
                print("[KER_Model_loader: loaded new build {0}.{1}]".format(Build, Make))                
        
        Model._M = k;
    
