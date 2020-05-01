
%run '../imodel.py' #from ipynb.fs.full.imodel import Model
import tensorflow as tfl

class TFL_Model(Model):
    _MODEL_PATH = None;
    
    def __init__(self, ModelID, Dataset):
        super().__init__(ModelID, Dataset)
 
    def load(self):
        print("Loading {0}".format(self._FULL_PATH))
        pass;
    
    def save(self):
        print("Saving {0}".format(self._FULL_PATH))
    
    def load_best_version(self):
        pass;
    
    def load_latest_version(self):
        pass;
        
    
    
