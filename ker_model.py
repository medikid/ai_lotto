
#!python3 '../imodel.py' #from ipynb.fs.full.imodel 
from imodel import Model
import keras as kr
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

import numpy as np
import pandas as pd

class KER_Model(Model):
    _MODEL_PATH = None;
    
    def __init__(self, ModelID, Dataset):
        super().__init__(ModelID, Dataset)
        self._MODEL_PATH = self._FULL_PATH;
        print("[KER_Model:__init__]")
 
    def load(self):
        self._M = load_model(self._FULL_PATH)
        print("[KER_Model:load]: Loading {0}".format(self._FULL_PATH))
        pass;
    
    def load_untrained(self):
        untrained_file_path = self.get_untrained_folder_path() + self._GAME + "." + self._API + "." + self._BUILD + ".h5";
        self._M = load_model(untrained_file_path);
        print("[KER_Model:load_untrained] {0}".format(untrained_file_path));
        
    def save(self):
        self._M.save(self._FULL_PATH)
        print("[KER_Model:save]: Saving {0}".format(self._FULL_PATH))
    
    def load_best_version(self):
        pass;
    
    def load_latest_version(self):
        pass;
    
    def predict(self, X_test, isBatch=True):
        Y_hat = np.array()
        if(isBatch == True):
            Y_hat = self._M.predict_on_batch(X_test);
        else:
            Y_hat = self._M.predict(X_test);
            
        return Y_hat;     
    
    
