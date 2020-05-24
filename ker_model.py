
#!python3 '../imodel.py' #from ipynb.fs.full.imodel 
from imodel import Model
from ker_model_loader import KER_Model_Loader
import keras as kr
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

from custom_functions import cf_metrics, cf_losses
from keras.metrics import Recall, Precision

import numpy as np
import pandas as pd

class KER_Model(Model):
    _MODEL_PATH = None;
    _CHECKPOINTS=[];
    
    def __init__(self, ModelID, Dataset, FileFormat='h5'):
        super().__init__(ModelID, Dataset, FileFormat)
        self._MODEL_PATH = self._FULL_PATH;
        print("[KER_Model:__init__]")
 
    def load(self):        
        if(self._IS_UNTRAINED ==  True):            
            print("[KER_Model:load]: Loading untrained {0}".format(self._FULL_PATH))
            self.load_untrained();
        elif (self._IS_CHECKPOINT == True):            
            print("[KER_Model:load]: Loading checkpoint {0}".format(self._FULL_PATH))
            self.load_checkpoint();
        else:
            print("[KER_Model:load]: Loading new build {0}".format(self._FULL_PATH))
            KER_Model_Loader(self); #actually creates new model
        
        #compile once again
        #self._M.compile(optimizer='adam', loss='mse', metrics=['accuracy']);
        
        #print("[KER_Model:load]: Loaded {0}".format(self._M.summary()))
        pass;
    
    def load_checkpoint(self):    
        chkpnt_file_path = self._CHECKPOINTS_FOLDER +  self._FILE_NAME +'.' + self._FILE_FORMAT;
        try:
            self._M = load_model(chkpnt_file_path);
        except ValueError:
            print("Issue: Unable to load due to usage of unserializable custom model/layer/metrics")
            
            KER_Model_Loader(self);            
            print("Solution-Step#1: Built/compiled new model")
            
            self._M.load_weights(chkpnt_file_path)
            print("Solution-Step#2: Loaded weights into new model from {0} ".format(chkpnt_file_path))
       
        print('[KER_Model:load_checkpoint] Loaded {0}'.format(chkpnt_file_path))
    
    def load_untrained(self):
        untrained_file_path = self.get_untrained_folder_path() + self._GAME + "." + self._API + "." + self._BUILD  + "." + self._MAKE + ".0.h5";
        try:
            self._M = load_model(untrained_file_path);
        except ValueError:
            print("Unable to load due to usage of unserializable custom model/layer/metrics")
            print("Solution: Build/compile new model")
            KER_Model_Loader(self);
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
    
    
