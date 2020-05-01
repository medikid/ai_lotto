
!python3 '../ker_model.py' #from ipynb.fs.full.ker_model import KER_Model;
!python3 '../tfl_model.py' #from ipynb.fs.full.tfl_model import TFL_Model;
!python3 '../idataset.py' #from ipynb.fs.full.idataset import Dataset

import numpy as np
import pandas as pd

class AI:
    _MODEL =None;
    _DATASET =None;
    _MASTER=None;
    _GAME = None;
    _API = None;
    
    def __init__(self, ModelID, DatasetID):
        self.load_dataset(DatasetID)
        
        self.load_model(ModelID, self._DATASET); #you need x5/25k from datasetfolder
        
        print("[AI:__init__]")
        pass;

    
    def setup_model(self, ModelName):
        pass;
    
    def load_model(self, ModelID, Dataset):
        info = ModelID.split(".");
        self._GAME = info[0];
        self._API = info[1]
        
        print("Game: {0} API {1}".format(self._GAME, self._API))
        
        if (self._API == 'ker'):
            self._MODEL = KER_Model(ModelID, Dataset);
        elif(self._API == 'tfl'):
            self._MODEL = TFL_Model(ModelID, Dataset);
            
        self._MODEL.load();
        print("Loaded Model {0}".format(self._MODEL._FULL_PATH))
    
    def load_dataset(self, DatasetName):
        self._DATASET = Dataset(DatasetName); 
        self._DATASET.load();
        #Dataset is stored in self._D._DATASET[IDs,X,Y,IDs_test,X_test,Y_test]
        print("Loaded Dataset {0}".format(self._DATASET._FULL_PATH))
    
    def save_model(self):
        self._MODEL.save(self._FULL_PATH)
        pass;
    
    def print_model_summary(self):
        print(self._MODEL._M.summary())
        pass;
    
    def train_model(self):
        pass;
    
    def predict_model(self, X_test, IsBatch=True):
        Y_hat = self._MODEL.predict(X_test, IsBatch)
        return Y_hat;
    
    
