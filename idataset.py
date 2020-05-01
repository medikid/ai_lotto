
import pandas as pd
import numpy as np

#!python3 '../ifile.py' #from ipynb.fs.full.ifile 
from ifile import iFile
#!python3 '../idata.py' #from ipynb.fs.full.idata 
from idata import Data

class Dataset(iFile):
    _ID = None;
    _GAME=None
    _D = None;
    _DATASET_PATH = None;
    _FOLDER_TYPE = 'datasets'
    _DF_MASTER = None;
    
    def __init__(self, DatasetID='', FolderPath= 'data/', FileFormat='npz'):
        self._ID = DatasetID;
        super().__init__(DatasetID, FolderPath, FileFormat)
        self.decipher_file_name();
        self.derive_folder_path(self._INFO); #you don't need this for model
        self.derive_file_path();
        self.derive_full_path();
    
    def decipher_file_name(self, delimiter="_"):
        #if .format is included, remove this first
        FN = self._FILE_NAME
        FN_PARTS = FN.split(delimiter);
        self._INFO['GAME'], self._GAME = FN_PARTS[0], FN_PARTS[0]
        self._INFO['xnINPUTS'] = FN_PARTS[1]
        self._INFO['nINPUTS'] = self.derive_xninputs(self._INFO['xnINPUTS'])
        self._INFO['xnDRAWS'] = FN_PARTS[2]
        self._INFO['nDRAWS'] = self.derive_xndraws(self._INFO['xnDRAWS'])
        self._INFO['DATA_TYPE'] = FN_PARTS[3] 
        #check if file format
        #self.set_file_format(self._FILE_NAME); #can't use this as we use decimal

        return self._FILE_NAME_DECIPHERS;
    
    #25000=>25k and 25k=>25000
    def derive_xndraws(self, xndraws):
        nDraw = 0; notation=''
        if (type(xndraws) == str):
            notation=xndraws[len(xndraws)-1];            
            if(type(notation) == int): multiple = 1;
            elif (notation=='k'): multiple = 1000;
            elif (notation=='m'): multiple = 1000000;
            new_st = float(xndraws.replace(notation,''))# change to float 2.5 first
            nDraw = int(new_st*multiple) #then covert back to int from float
            print("xndraw{0}, new_st{1}, multiple{2}, notation{3}".format(xndraws, new_st, multiple, notation))
        elif (type(xndraws) == int):
            if(xndraws > 1000000) :
                notation = 'm';
                nDraw = str(xndraws//1000000)+notation;
            elif (xndraws > 1000):
                notation = 'k';
                nDraw = str(xndraws//1000)+notation;
            else:
                nDraw = str(xndraws//1);
        return nDraw;
    
    #x15=>15 and 15=>x15
    def derive_xninputs(self, xnInputs):
        nInputs = 0;
        if (type(xnInputs) == str):
            #remove first character
            nInputs = int(xnInputs[1:]);
        elif (type(xnInputs) == int):
            nInputs = 'x'+str(xnInputs);        
        return nInputs;
    

    def set_file_format(self, FileName):
        #split name and see if file format is included
        Fs=FileName.split(".");
        try:
          self._FILE_FORMAT=Fs[1];
        except IndexError:
          print("File format not included in FIle Name")
        return self._FILE_FORMAT;

    def derive_file_path(self):
        #derive file name from decipher  
        FilePath = self._FOLDER_TYPE + '/';
        FilePath += self._FILE_NAME;
        FilePath += "." + self._FILE_FORMAT;
        self._FILE_PATH = FilePath;
        print("[iDataset:derive_file_path] {0}".format(self._FILE_PATH))
        
    def load(self):
        self._D = np.load(self._FULL_PATH)
        print("Loading dataset {0}".format(self._FULL_PATH))
        
    def load_master_df(self):
        if(self._DF_MASTER is None):
            masterPkl = self._GAME +"_master";
            masterDF = Data(masterPkl);
            self._DF_MASTER = masterDF._DF_MASTER;
            print("[iDataset:load_master_df] Loaded master DF")
            
    def derive_full_path_by_id(self, newDatasetID):
        self._FILE_NAME = newDatasetID; #this is the onlyl thing you need to change
        self.decipher_file_name();
        self.derive_folder_path(self._INFO); #you don't need this for model
        self.derive_file_path();
        self.derive_full_path();        
        self.ensure_dirs(); #ensure Full Path exists
        
    def create_new_by_id(self, DF_Master, newDatasetID, nTests=15):
        self.derive_full_path_by_id(newDatasetID);        
        #now use file info details self._INFO
        
        masterPkl = self._INFO['GAME'] +"_master";
        masterData = Data(masterPkl);
        ids, x, y, ids_test, x_test, y_test=masterData.create_supervised_package(DF_Master, 0, self._INFO['nDRAWS'], self._INFO['nINPUTS'], nTests);
        
        np.savez(self._FULL_PATH, IDs=ids, X=x, Y=y, IDs_test=ids_test, X_test=x_test, Y_test=y_test);
        
        print("Saved new datast file {0}".format(self._FULL_PATH))
        
    def derive_new_dataset_id(self, Game, nDraws, nInputs, DataType='dr'):
        newDatasetID = Game + "_";
        newDatasetID += self.derive_xninputs(nInputs) + "_";
        newDatasetID += self.derive_xndraws(nDraws) + "_";
        newDatasetID += DataType;
        return newDatasetID;
        
    def create_new(self, DF_Master, startID, nDraws, nInputs, nTests=15, DataType='dr'):        
        newDatasetID = self.derive_new_dataset_id(self._INFO['GAME'], nDraws, nInputs, DataType);
        
        self.create_new_by_id(DF_Master, newDatasetID, nTests)
