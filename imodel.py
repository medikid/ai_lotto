
import os
# %run '../ifile.py'
# %run '../idata.py'
# %run '../idataset.py'
# from ipynb.fs.full.ifile import iFile
# from ipynb.fs.full.idata import Data
# from ipynb.fs.full.idataset import Dataset

class Model(iFile):
    _M=None;
    _ID=None;
    _GAME=None;
    _API = None;
    _BUILD = '0'; #model architecture/layers
    _MAKE = '0'; #Same build, but differnet make parameters
    _VERSION = '0'; #training / epoch 
    
    #set default model
    _FOLDER_TYPE = 'models'
    _FOLDER_PATH = None;
    _FILE_FORMAT = None
    
    
    _IS_CHECKPOINT = False;
    _CHECKPOINTS_FOLDER=None;
    _CHECKPOINT_EPOCH = 0;
    
    def __init__(self, ModelID, Dataset, FileFormat='h5'):
        self._ID = ModelID;
        super().__init__(ModelID, Dataset._FOLDER_PATH, FileFormat);
        self.decipher_file_name();
        
        #we are not deriving folder_path, but using dataset folder
                
        self.derive_file_path();        
        self.derive_game_path(Dataset._INFO) #derives 'data/keno'
        self.derive_full_path();
        self.derive_checkpoints_folder();
        print("[iModel:__init__")
        
    def set_decipher_info(self):
        self._INFO['GAME'] = self._GAME;
        self._INFO['API'] = self._API;
        self._INFO['BUILD'] = self._BUILD;
        self._INFO['MAKE'] = self._MAKE;
        self._INFO['VERSION'] = self._VERSION;
        self._INFO['IS_CHECKPOINT'] = self._IS_CHECKPOINT;
        self._INFO['CHECKPOINT_EPOCH'] = self._CHECKPOINT_EPOCH;
        print("[iModel:set_decipher_info] {0}".format(self._INFO))
        
    def decipher_file_name(self, Delimiter="."):        
        ids = self._FILE_NAME.split(Delimiter);
        try:
            self._GAME = ids[0]
            self._API = ids[1]
            self._BUILD = ids[2]
            if (len(ids) > 3):
                self._MAKE = ids[3];
            else:
                self._MAKE= '0';
            if (len(ids) > 4):
                self._VERSION = ids[4]
                if(ids[4][0]=='e'):
                    self._IS_CHECKPOINT = True;
                    self._CHECKPOINT_EPOCH = int(ids[4][1:])
            else:
                self._VERSION = '0';
        except IndexError:
            self.LoadBestVersion();
        
        print("[iModel:decipher_file_name]")
        
        self.set_decipher_info();        
    
               
    def derive_file_path(self):
        #derive file name from decipher  
        #FilePath = self._FOLDER_PATH;
        FilePath = self._FOLDER_TYPE + "/";
        FilePath += self._INFO['API'] + "/";
        FilePath += str(self._INFO['BUILD']) + "/";
        FilePath += str(self._INFO['MAKE']) + "/";
        #FilePath += "/" + self._INFO['VERSION'];
        FilePath += self._FILE_NAME;
        FilePath += "." + self._FILE_FORMAT;
        
        self._FILE_PATH = FilePath;
        print("[iModel:derive_file_path] {0}".format(self._FILE_PATH))
        
    def derive_checkpoints_folder(self):
        self._CHECKPOINTS_FOLDER = os.path.dirname(self._FULL_PATH) + '/checkpoints';
        self.ensure_dirs(self._CHECKPOINTS_FOLDER)
        
    
    def load(self):
        print("[iModel:load] Not Implemented")
        pass;
    
    def get_untrained_folder_path(self):
        return self._ROOT_FOLDER + self._GAME_PATH +'untrained_models/';
        
 
    def load_untrained(self):
        untrained_file_path = self.get_untrained_folder_path() + self._GAME + "." + self._API + "." + self._BUILD + ".h5";
        self._M.load(untrained_file_path);
        print("Saved untrained model {0}".format(untrained_file_path));
       
    def save_untrained(self):
        untrained_file_path = self.get_untrained_folder_path()  + self._GAME + "." + self._API + "." + self._BUILD + ".h5";
        self._M.save(untrained_file_path);
        print("Saved untrained model {0}".format(untrained_file_path));
        
    def load_checkpoint(self, FileName):
        chkpnt_file_path = self._CHECKPOINTS_FOLDER + '/' + FileName;
        self._M.load(chkpnt_file_path);
        print('[iModel:load_checkpoint] Loaded {0}'.format(chkpnt_file_path))
        
    def save_checkpoint(self, FileName):
        chkpnt_file_path = self._CHECKPOINTS_FOLDER + '/' + FileName;
        print('[iModel:save_checkpoint] Saved {0}'.format(chkpnt_file_path))
    
    def load_best_version(self):
        print("[iModel:load_best_version] Not Implemented")
        pass;
    
    def load_latest_version(self):
        print("[iModel:load_latest_version] Not Implemented")
        pass;
           
             
    def get_model_summary(self):
        return self._M.summary();
    
    def train(self, x, y, Epochs=10, BatchSize=32):
        self._M.fit(x, y, epochs=Epochs, batch_size=BatchSize);        
        pass;
    
    def predict(self, x_test):
        return y_hat;
    
    def save(self):
        self._M.save(self._FULL_PATH)
        
        
    def print_summary(self):
        print(self._M.summary())
                
