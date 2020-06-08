
import os

from ifile import iFile
from idata import Data
from idataset import Dataset

class Model(iFile):
    _M=None;
    _ID=None;
    _MODSET_ID=None;
    _GAME=None;
    _API = None;
    _BUILD = '0'; #model architecture/layers
    _MAKE = '0'; #Same build, but differnet make parameters
    _VERSION = '0'; #hyperparameter    
    _CHECKPOINT = '0'; #string exxxx epoch
    
    #set Dataset IO Shape
    _D_X_SHAPE = (0,0,0);
    _D_Y_SHAPE = (0,0);
    
    #set default model
    _FOLDER_TYPE = 'models'
    _FOLDER_PATH = None;
    _FILE_FORMAT = None
    
    
    _IS_CHECKPOINT = False;
    _CHECKPOINTS_FOLDER=None;
    _CHECKPOINT_FORMAT = '.h5'
    _CHECKPOINT_EPOCH = 0; #int checkpoint#
    
    _IS_UNTRAINED = False;
    _IS_ARCHIVED = False;
    
    def __init__(self, ModelID, Dataset, FileFormat='.h5'):
        self._ID = ModelID;
        self._D_X_SHAPE = Dataset._D['X'].shape;
        self._D_Y_SHAPE = Dataset._D['Y'].shape;
        super().__init__(ModelID, Dataset._FOLDER_PATH, FileFormat);
        self.decipher_file_name();
        
        #we are not deriving folder_path, but using dataset folder
                
        self.derive_file_path();        
        self.derive_game_path(Dataset._INFO) #derives 'data/keno'
        self.derive_full_path();
        self.derive_checkpoints_folder();
        
        self.derive_modset_id(Dataset)
        print("[iModel:__init__")
        
    def set_decipher_info(self):
        self._INFO['GAME'] = self._GAME;
        self._INFO['API'] = self._API;
        self._INFO['BUILD'] = self._BUILD;
        self._INFO['MAKE'] = self._MAKE;
        self._INFO['VERSION'] = self._VERSION;
        self._INFO['CHECKPOINT']=self._CHECKPOINT;
        self._INFO['IS_CHECKPOINT'] = self._IS_CHECKPOINT;
        self._INFO['CHECKPOINT_EPOCH'] = self._CHECKPOINT_EPOCH;
        print("[iModel:set_decipher_info] {0}".format(self._INFO))
        
    def derive_modset_id(self, Dataset):
        self._MODSET_ID = self._INFO['GAME'] \
                            +'.'+  self._INFO['API'] \
                            +'.'+  self._INFO['BUILD'] \
                            +'.'+  self._INFO['MAKE'] \
                            +'.'+  self._INFO['VERSION'] \
                            + '[' + Dataset._INFO['xnINPUTS'] \
                            +'_'+  Dataset._INFO['xnDRAWS'] \
                            +'_'+  Dataset._INFO['DATA_TYPE'] \
                            + ']';
        print("Modset ID: {0}".format(self._MODSET_ID))
        return self._MODSET_ID;
        
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
            else:
                self._VERSION = '0';
                
            if (len(ids) > 5):
                self._CHECKPOINT = ids[5]
                if(ids[5][0]=='e'):
                    self._IS_CHECKPOINT = True;
                    self._CHECKPOINT_EPOCH = int(ids[5][1:])
                if (ids[5][0] == '0'):
                    self._IS_UNTRAINED = True;
            else:
                self._CHECKPOINT = '0'
                
        except IndexError:
            self.LoadBestVersion();
        
        print("[iModel:decipher_file_name] {0}".format(self._FILE_NAME))
        
        self.set_decipher_info();        
    
               
    def derive_file_path(self):
        #derive file name from decipher  
        #FilePath = self._FOLDER_PATH;
        FilePath = self._FOLDER_TYPE + "/";
        FilePath += self._INFO['API'] + "/";
        FilePath += str(self._INFO['BUILD']) + "/";
        FilePath += str(self._INFO['MAKE']) + "/";
        if (int(self._INFO['VERSION']) > 0):
            FilePath += self._INFO['VERSION'] + "/" ;
        FilePath += self._FILE_NAME;
        FilePath += self._FILE_FORMAT;
        
        self._FILE_PATH = FilePath;
        print("[iModel:derive_file_path] {0}".format(self._FILE_PATH))
    
    def load(self):
        #this is a generic model.Lets load API_specific model here
        if (self._API == 'ker'):
            from ker_model import KER_Model
            self.__class__ = KER_Model;
            print("[Model:loader] converted Model to KER_Model")
        elif (self._API == 'tfl'):
            from tfl_model import TFL_Model
            self.__class__ = TFL_Model;
            print("[Model:loader] converted Model to TFL_Model")        
        self.load();
        pass;
    
    def get_untrained_folder_path(self):
        return self._ROOT_FOLDER + self._GAME_PATH +'untrained_models/';
    
    def get_archived_folder_path(self):
        return self._ROOT_FOLDER + 'archive/' + self._GAME.lower() + "/"
    
    def load_archived(self, fileFormat='.h5'):
        archived_model_path = self.get_archived_folder_path() + 'models/' + self._MODSET_ID ;
        if (str(self._CHECKPOINT)[0] == 'e'):
            archived_model_path += '['+str(self._CHECKPOINT)+']'
        archived_model_path += fileFormat
        self._M.load(archived_model_path); 
        print("[iModel:load_archived] Loaded arhived path {0}".format(arhived_model_path))
 
    def load_untrained(self):
        untrained_file_path = self.get_untrained_folder_path() + self._GAME + "." + self._API + "." + self._BUILD + "." + self._MAKE + self._VERSION + ".0.h5";
        self._M.load(untrained_file_path);
        print("[iModel:load_untrained] Loaded untrained model {0}".format(untrained_file_path));   
       
    def save_untrained(self):
        untrained_file_path = self.get_untrained_folder_path()  + self._GAME + "." + self._API + "." + self._BUILD + "." + self._MAKE + self._VERSION + ".0.h5";
        self._M.save(untrained_file_path);
        print("[iModel:save_untrained] Saved untrained model {0}".format(untrained_file_path));
        
    def set_checkpoint_format(self, ChkptFormat='.h5'):
        self._CHECKPOINT_FORMAT = ChkptFormat
        
    def derive_checkpoints_folder(self):
        self._CHECKPOINTS_FOLDER = os.path.dirname(self._FULL_PATH) + '/checkpoints/';
        self.ensure_dirs(self._CHECKPOINTS_FOLDER)
        print('[iModel:derive_checkpoints_folder] Ensured {0}'.format(self._CHECKPOINTS_FOLDER))
        
    def load_latest_checkpoint(self):
        MAX_CHKPNT = 0; CHKPNT = 0;
        for f in os.listdir(self._CHECKPOINTS_FOLDER):
            f_splits = f.split('.');
            #print(f_splits)
            if(len(f_splits) > 6):
                print(f_splits,"-6-",f_splits[-2])
                CHKPNT = int(f_splits[-2][1:])
            elif(len(f_splits) > 5 ):
                print(f_splits,"-5-",f_splits[-1])
                CHKPNT = int(f_splits[-1][1:])

            if (CHKPNT > MAX_CHKPNT):
                MAX_CHKPNT = CHKPNT;
        print("Latest checkpoint is {0}".format(MAX_CHKPNT))
        
        self._CHECKPOINT = 'e' + str(MAX_CHKPNT).zfill(4);
        self._INFO['CHECKPOINT'] = self._CHECKPOINT;
        
        self._IS_CHECKPOINT = True;
        self._IS_UNTRAINED = False;
        self._CHECKPOINT_EPOCH = MAX_CHKPNT;
        
        #redireve file name
        self._FILE_NAME = self._GAME + '.' + self._API + '.'+ self._BUILD + '.' + self._MAKE + '.' + self._VERSION + '.' + self._CHECKPOINT;
        
        self.load_checkpoint();
        
    def load_checkpoint(self):
        chkpnt_file_path = self._CHECKPOINTS_FOLDER + self._FILE_NAME + self._CHECKPOINT_FORMAT;
        self._M.load(chkpnt_file_path);
        print('[iModel:load_checkpoint] Loaded {0}'.format(chkpnt_file_path))
        
    def save_checkpoint(self, FileName):
        chkpnt_file_path = self._CHECKPOINTS_FOLDER + self._FILE_NAME + self._CHECKPOINT_FORMAT;
        self._M.save(chkpnt_file_path)
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
        #convert 1/2-d to 3-d
        if (x_test.ndim == 1):
            x_test = x_test.reshape(1,1,x_test.shape[0])
        elif (x_test.ndim == 2):
            x_test = x_test.reshape(1,x_test.shape[0],x_test.shape[1])
        
        y_hat=self._M.predict(x_test)
        return y_hat;
    
    def save(self):
        self._M.save(self._FULL_PATH)
        
        
    def print_summary(self):
        print(self._M.summary())
                
