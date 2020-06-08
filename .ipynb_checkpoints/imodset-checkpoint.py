
import os

from ifile import iFile
from idata import Data
from idataset import Dataset
from imodel import Model

class Modset:
    _MODSET_ID = None;
    _CHECKPOINT = None;
    _MODEL_ID = None;
    _DATASET_ID = None;
    _MODEL = None;
    _DATASET = None;
    _IS_ARCHIVED=False
    
    
    def __init__(self, ModsetID, CheckPoint=0, isArchivedModel=False):
        self._MODSET_ID = ModsetID;
        self._CHECKPOINT = CheckPoint;
        self._IS_ARCHIVED = isArchivedModel;
        self._MODEL_ID, self._DATASET_ID = self.parse_modset_id(self._MODSET_ID);
        self.setup_checkpoint(CheckPoint)
        
        self.load_dataset();
        self.load_model();
    
    def parse_modset_id(self, ModsetID):
        ids = ModsetID.split('[')
        mods = ids[0].split('.')
        dats = ids[1][:-1].split('_')
        model_id = '{0}.{1}.{2}.{3}.{4}'.format(mods[0],mods[1], mods[2], mods[3],mods[4])
        dataset_id = '{0}_{1}_{2}_{3}'.format(mods[0],dats[0], dats[1], dats[2])
        return model_id, dataset_id;
    
    def setup_checkpoint(self, Checkpoint):
        if (Checkpoint != 0):
            if (str(Checkpoint)[0] == 'e'):
                self._CHECKPOINT = str(Checkpoint)
                self._MODEL_ID += '.' + self._CHECKPOINT
            else:
                self._CHECKPOINT = 'e' + str(Checkpoint).zfill(4);
            
            self._MODEL_ID += '.' + self._CHECKPOINT
        else:
            self._CHECKPOINT = 0;

    def load_model(self):
        self._MODEL = Model(self._MODEL_ID, self._DATASET);
        if (self._IS_ARCHIVED == True):
            self._MODEL.load_archived();
        else:
            self._MODEL.load();
        
    def load_dataset(self):
        self._DATASET = Dataset(self._DATASET_ID);
        self._DATASET.load();
        
        
