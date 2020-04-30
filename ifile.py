
from pathlib import Path
import pandas as pd
import numpy as np

class iFile:
    _FILE = None;
    _FILE_PATH=None;
    _FILE_NAME=None;
    _FILE_NAME_DECIPHERS={};
    _FILE_FORMAT=None;
    _FILE_TYPE=None;
    _FOLDERS_DECIPHERS={};
    _FOLDER_PATH=None; #
    _ROOT_FOLDER='../';
    _FULL_PATH = None; #[../][data/hotspot][/x5/25k][models][ker/1][hotspot_ker_1_2020040320300.h5]
    
    _GAME_PATH = None;
    _DATASET_PATH = None    
    _FOLDER_TYPE=None;
    
    
    _INFO = {};

    def __init__(self, FileName, FolderPath='', FileFormat=None):
        self._FILE_NAME = FileName;
        self._FOLDER_PATH = FolderPath;
        self._FILE_FORMAT = FileFormat;
        self.decipher_file_name();
        #self.derive_file_path();
        #self.derive_full_path();
        print("[iFile:__init__]")
        
    def set_root_folder(self, RootFolder):
        self._ROOT_FOLDER = RootFolder;
        print("[iFile:set_root_folder] {0}".format(RootFolder))
        
    def decipher_file_name(self, delimiter="_"):
        print("[iFile:decipher_file_name] Not Implemented");
        
    def derive_game_path(self, DatasetInfo):
        self._GAME_PATH = 'data/' + DatasetInfo['GAME'] + "/";
        
    def derive_dataset_path(self, DatasetInfo):
        self._DATASET_PATH = DatasetInfo['xnINPUTS']  + "/" + DatasetInfo['xnDRAWS'] + "/";
        
        
    def derive_folder_path(self, DatasetInfo):
        self.derive_game_path(DatasetInfo);
        self.derive_dataset_path(DatasetInfo)
        self._FOLDER_PATH = self._GAME_PATH + self._DATASET_PATH;
        print("[iFile:derive_folder_path] {0}".format(self._FOLDER_PATH))
       
   
    def set_file_format(self, FileName):
        #split name and see if file format is included
        Fs=FileName.split(".");
        try:
          self._FILE_FORMAT=Fs[1];
        except IndexError:
          print("File format not included in FIle Name")
        
        print("[iFile:set_file_format] {0}".format(self._FILE_FORMAT))
        return self._FILE_FORMAT;
    
   
    def derive_file_path(self):
        #derive file name from decipher  
        print("[iFile:derive_file_path] Not Implemented")

    def derive_full_path(self):
        FullPath = self._ROOT_FOLDER; #back to /
        FullPath += self._FOLDER_PATH; #derived from id
        FullPath += self._FILE_PATH;
        self._FULL_PATH = FullPath;
        
        self.ensure_dirs(); #ensure Full Path exists        
        print("[ifile:derive_full_path] {0}".format(self._FULL_PATH))
        
    def get_full_path(self):
        print("[iFile:get_full_path] {0}".format(self.__FULL_PATH))
        return self._FULL_PATH;
    
    def ensure_dirs(self):
        path = Path(self._FULL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        
    def getVarType(self, Var):
        varType = ''
        if (type(Var) == pd.core.frame.DataFrame):
            varType = "Dataframe";
        elif (type(Var) == type(None)):
            varType = "Nonetype";
        elif (type(Var) == np.ndarray):
            varType = "NumpyArray";
        #elif(type(df) == np.)
        
        return varType;
        
    def isDataframe(self, Var):
        return self.getVarType(Var) == "Dataframe";
    
    def isNonetype(self, Var):
        return self.getVarType(Var) == "Nonetype";
    
    def isNumpyArray(self, Var):
        return self.getVarType(Var) == "NumpyArray";
    
    def print_file_info(self):
        print("FILE NAME: {0}".format(self._FILE_NAME));
        print("FILE FORMAT(type): {0}[{1}]".format(self._FILE_FORMAT, self._FILE_TYPE));
        print("FILE PATH: {0}".format(self._FILE_PATH));
        print("FILE NAME PARTS: {0}".format(self._INFO));
