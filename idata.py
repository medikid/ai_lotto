
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sb
import keras as kr
from numpy import array
!python3 '../ifile.py' #from ipynb.fs.full.ifile import iFile

class Data(iFile):
    _FILE_FORMAT = "pkl"
    _DATA_FOLDER = None;
    _DF_MASTER = None
    _DF_IDs = None;
    _DF_R = None;
    _DF_DR = None;
    _DF_DP = None;
    _NPZ_MASTER = None
    _MIN_ID=0;
    _MAX_ID=0;
    _COUNT_ID=0;
    
    def __init__(self, MasterFileName="hotspot_master", LoadMaster=False, DataFolder="data/"):
        self._FILE_NAME  = MasterFileName;
        self._DATA_FOLDER = DataFolder;
        super().__init__(MasterFileName, DataFolder, self._FILE_FORMAT);
        
        self.derive_folder_path();
        self.derive_file_path();
        self.derive_full_path();
        
        if(LoadMaster == True):
            self.load_master_df();
            self.deconstruct_master_df()
            print("Loaded data file %s" %(self._FULL_PATH))
            
    #v
    def decipher_file_name(self, Delimiter="_"):
        #if .format is included, remove this first
        
        FN_PARTS = self._FILE_NAME.split(Delimiter);
        try:
            self._INFO['GAME'], self._GAME = FN_PARTS[0], FN_PARTS[0]
            self._INFO['FOLDER_TYPE'], self._FOLDER_TYPE = FN_PARTS[1], FN_PARTS[1]
            self._INFO['DATA_TYPE'], self._DATA_TYPE = FN_PARTS[2], FN_PARTS[2]
            #check if file format
            self.set_file_format(self._FILE_NAME)
        except IndexError:
            pass;      
        
    def derive_folder_path(self):
        self._FOLDER_PATH = 'data/' + self._GAME + "/";
        
    def derive_file_path(self):
        self._FILE_PATH = "master" + "/" + self._FILE_NAME +"." + self._FILE_FORMAT;
    
    #v    
    def load_file(self):
        self.set_master_df(pd.read_pickle(self._FULL_PATH));   

            
    def get_data_folder(self):
        return self._DATA_FOLDER;
    
    def load_master_df(self):
        if (self._FILE_NAME != ""):
            self.set_master_df(pd.read_pickle(self._FULL_PATH));
    
    def set_master_df(self, master_df):
        self._DF_MASTER = master_df;
        self._MIN_ID = self._DF_MASTER.min();
        self._MAX_ID = self._DF_MASTER.index.max();
        self._COUNT_ID = self._DF_MASTER.shape[0];
        
    def get_master_df(self):
        return self._DF_MASTER;
    
    def set_master_npz(self, master_npz):
        self._NPZ_MASTER = master_df;
        
    def get_master_npz(self):
        return self._NPZ_MASTER
    
    def get_result_columns(self):
        i=1; r=[];
        while(i<=20):
            r.append('r'+str(i));
            i+=1;        
        return r;
    
    def get_draw_columns(self):
        i=1; drs=[];
        while(i<=80):
            drs.append('dr'+str(i));
            i+=1;        
        return drs;
    
    def get_depth_columns(self):
        i=1; dps=[];
        while(i<=80):
            dps.append('dp'+str(i));
            i+=1;        
        return dps;
    
    def get_count_id(self, df=None):
        if(df is None):
            count_id = self._COUNT_ID;
        else:
            count_id = df.shape[0];
        return count_id;
    
    def get_first_id(self, df=None):
        if(df is None):
            min_id = self._MIN_ID;
        else:
            min_id = df[:1].index.item();
        return min_id;
    
    def get_last_id(self, df=None):
        if(df is None):
            max_id = self._MAX_ID;
        else:
            max_id = df[-1:].index.item();
        return max_id;
    
    def deconstruct_master_df(self):
        if(self._DF_IDs is None):
            self._DF_IDs, self._DF_R, self._DF_DR, self._DF_DP = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame();

            if (self._DF_MASTER is not None ): #or self._DF_MASTER.empty): #empty is an attribute, not a function
                print("ERROR: Missing Master Dataframe");
            else:
                self._DF_IDs = pd.DataFrame(self._DF_MASTER.index);
                self._DF_R = self._DF_MASTER.loc[:,self.get_result_columns()]
                self._DF_DR = self._DF_MASTER.loc[:,self.get_draw_columns()]
                self._DF_DP = self._DF_MASTER.loc[:,self.get_depth_columns()]

            return self._DF_IDs, self._DF_R, self._DF_DR, self._DF_DP;
    
    def df2np(self, df):
        if(isinstance(df, pd.DataFrame)):
            df = df.to_numpy();
        return df;
    
    def np2df(self, nparray):
        if(isinstance(nparray, np.ndarray)):
            nparray = pd.DataFrame(nparray);
        return nparray;
            
    def split_xy(self, df):
        x, y = pd.DataFrame(), pd.DataFrame();
        if not df.empty:
           x, y = df[:-1],df[1:];
        else:
            print("Empty dataframe")
        return x,y;
        
    def print_xy(self, x, y):
        x = self.df2np(x);
        y= self.df2np(y)
        for i in range(len(x)):
            print(x[i],y[i])
            
    def print_xy_prediction(self, y_hat, y_test):
        print("[Predicted vs Actuual]")
        
        #n_input - input rows- number of previous input timesteps t,t-1,t-2,t....n
        #t_output - Y timesteps to predict. 0 by default - predicts t+1        
    def to_supervised(self, data_df, n_input, t_output=0):
        data = self.df2np(data_df);
        X, y = list(), list()
        ix_start = 0
        # step over the entire history one time step at a time
        for i in range(len(data)):
            # define the end of the input sequence
            ix_end = ix_start + n_input
            ix_output = ix_end + t_output
            # ensure we have enough data for this instance
            if ix_output < len(data):
                X.append(data[ix_start:ix_end])
                y.append(data[ix_output])
                # move along one time step
                ix_start += 1
        return array(X), array(y);
    
    def create_supervised_package(self, masterDF, startID, nDraws, nInputs, nTests):
        #ids, X, Y, X_test, Y_test = np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]);
        cutOff = nDraws + nInputs + nTests;
        nDFs = masterDF[startID:cutOff]
        nIDs = self.df2np(pd.DataFrame(nDFs.index))
        X_all, Y_all = self.to_supervised(nDFs,nInputs)
        
        CutSplit = -nTests; #split by 5, uses [:-5] and [-5:]
        X, Y, X_test, Y_test = self.xy_split(X_all, Y_all, CutSplit);
        IDs, IDs_test =  nIDs[nInputs:CutSplit], nIDs[CutSplit:];
        return IDs, X, Y, IDs_test, X_test, Y_test
        
        
    def xy_split(self, X_all, Y_all, CutSplit):
        x,y,x_test,y_test = X_all[:CutSplit], Y_all[:CutSplit], X_all[CutSplit:], Y_all[CutSplit:];
        return x, y, x_test, y_test;
    
    def create_new(self, newMasterFile, newMasterDF):
        self._FILE_NAME = newMasterFile;
        self.derive_folder_path();
        self.derive_file_path();
        self.derive_full_path();
        
        newMasterDF.to_pickle(self._FULL_PATH)
        print("Saved new file {0}".format(self._FULL_PATH))
        
