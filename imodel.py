
from ipynb.fs.full.ifile import iFile

class Model(iFile):
    _M=None;
    _ID=None;
    _GAME=None;
    _API = None;
    _BUILD = None;
    _VERSION = None;
    
    #set default model
    _FOLDER_TYPE = 'models'
    _FOLDER_PATH = None;
    _FILE_FORMAT = None
    
    def __init__(self, ModelID, Dataset, FileFormat='h5'):
        self._ID = ModelID;
        super().__init__(ModelID, Dataset._FOLDER_PATH, FileFormat);
        self.decipher_file_name();
        
        #we are not deriving folder_path, but using dataset folder
           
        self.derive_file_path();
        self.derive_full_path();
        print("[iModel:__init__")
        
    def set_decipher_info(self):
        self._INFO['GAME'] = self._GAME;
        self._INFO['API'] = self._API;
        self._INFO['BUILD'] = self._BUILD;
        self._INFO['VERSION'] = self._VERSION;
        print("[iModel:set_decipher_info] {0}".format(self._INFO))
        
    def decipher_file_name(self, Delimiter="."):        
        ids = self._FILE_NAME.split(Delimiter);
        try:
            self._GAME = ids[0]
            self._API = ids[1]
            self._BUILD = ids[2]
            self._VERSION = ids[3]
        except IndexError:
            self.LoadBestVersion();
        
        print("[iModel:decipher_file_name]")
        
        self.set_decipher_info();        
    
    def derive_game_path(self):
        self._GAME_PATH = self._INFO['GAME']
        
            
    def derive_file_path(self):
        #derive file name from decipher  
        #FilePath = self._FOLDER_PATH;
        FilePath = self._FOLDER_TYPE + "/";
        FilePath += self._INFO['API'] + "/";
        FilePath += self._INFO['BUILD'] + "/";
        #FilePath += "/" + self._INFO['VERSION'];
        FilePath += self._FILE_NAME;
        FilePath += "." + self._FILE_FORMAT;
        self._FILE_PATH = FilePath;
        print("[iModel:derive_file_path] {0}".format(self._FILE_PATH))
        
    
    def load(self):
        print("[iModel:load] Not Implemented")
        pass;
    
    def load_best_version(self):
        print("[iModel:load_best_version] Not Implemented")
        pass;
    
    def load_latest_version(self):
        print("[iModel:load_latest_version] Not Implemented")
        pass;
           
             
    def get_model_summary(self):
        return self._M.summary();
    
    def train(self, x, y, BatchSize=32, Epochs=1000):
        self._M.fit(x, y, batch_size=BatchSize, epochs=Epochs);        
        pass;
    
    def predict(self, x_test):
        return y_hat;
    
    def save(self):
        self._M.save(self._FULL_PATH)
        
    def print_summary(self):
        print(self._M.summary())
    
        
        
