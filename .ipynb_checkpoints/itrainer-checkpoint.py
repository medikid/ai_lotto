
from idataset import Dataset
from imodel import Model
from ker_model import KER_Model
from tfl_model import TFL_Model

from keras.callbacks import ModelCheckpoint, History, CSVLogger
from custom_callbacks import cf_callbacks
import matplotlib.pyplot as plt

class Trainer:
    _MODEL = None
    _DATASET = None
    _EPOCHS = 10;
    _BATCH_SIZE=100;
    _CALLBACKS=[]
    _INFO={}
    
    def __init__(self, ModelID, DatasetID, LoadLatestCheckpoint=False):
        self._DATASET = Dataset(DatasetID);
        self._DATASET.load()
        
        self._MODEL = Model(ModelID, self._DATASET);
        self._MODEL.load();
        if (LoadLatestCheckpoint==True):
            self._MODEL.load_latest_checkpoint();
       
    
    def set_callbacks(self, Callbacks={}):
         for key, val in Callbacks.items():
            if(key == 'per_epoch'):
                self.set_checkpoint_per_epoch(val);
            if(key=='csv_logger'):
                self.get_csv_history(val);
            if(key=='upload_history'):
                self.upload_history_per_epoch(val);
            if(key=='custom_callbacks'):
                self.set_custom_callbacks(val);
                
    def set_custom_callbacks(self, CallBackName):
        custom_callbacks = cf_callbacks();
        self._CALLBACKS.append(custom_callbacks)
        print("[itrainer:set_custom_callbacks] added custom_callback {0}".format(CallBackName))
                
                
    def get_csv_history(self, PerEpoch=1):
        current_folder = pwd #gives a array with current working folder i.e notebook folder
        db_folder = current_folder[0] + '/db/'
        csv_file= db_folder + 'csv_logs.csv'
        csv_logger = CSVLogger(csv_file, separator=",", append=True);
        self._CALLBACKS.append(csv_logger)
        print("[itrainer:get_csv_histry] added csv_logger callback")
        
    def upload_history_per_epoch(self, PerEpoch=1):
        history = History();
        #self._CALLBACKS.append(chkpnt)
        print("[itrainer:uplload_histry_per_epoch] added db_upload_history callback")
            
    
    def set_checkpoint_per_epoch(self, PerEpoch=5):
        file_name = self._MODEL._INFO['GAME'] + '.';
        file_name += self._MODEL._INFO['API']  + '.';
        file_name += self._MODEL._INFO['BUILD']  + '.';
        file_name += self._MODEL._INFO['MAKE']  + '.';
        #file_name += 'e'+str(self._MODEL._CHECKPOINT_EPOCH)+'+'+int('{epoch:04d}') ;
        file_name += 'e{epoch:04d}' ;
        
        self._MODEL.derive_checkpoints_folder();    
        chkpnt_file = self._MODEL._CHECKPOINTS_FOLDER + file_name + '.h5';
        self._MODEL.ensure_dirs(chkpnt_file); #ensure checkpnt folder exists, folder doesn't work, so use filepath
        chkpnt = ModelCheckpoint(            
            chkpnt_file,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=PerEpoch);
        self._CALLBACKS.append(chkpnt)
        print("[itrainer:set_checkpoint_per_epoch] added model_checkpoint callback")
    
    def train(self, Epochs=10, BatchSize=100, Callbacks={}, Verbose=0):
        self._EPOCHS = Epochs;
        self._BATCH_SIZE = BatchSize;
        self.set_callbacks(Callbacks);       
                
        self._INFO['HISTORY'] = self._MODEL._M.fit( x= self._DATASET._D['X'] \
                            , y= self._DATASET._D['Y'] \
                            , batch_size= self._BATCH_SIZE \
                            , epochs= self._EPOCHS \
                            , verbose= Verbose \
                            , callbacks=self._CALLBACKS \
                            #, validation_split=0.0 \
                            #, validation_data=None \
                            , shuffle = False \
                            #, class_weight=None \
                            #, sample_weight=None \
                            , initial_epoch = self._MODEL._CHECKPOINT_EPOCH \
                            #, steps_per_epoch=None \
                            #, validation_steps=None \
                            #, validation_freq=1 \
                            #, max_queue_size=10 \
                            #, workers=1 \
                            #, use_multiprocessing=False\
                            );
    
    def plt_history(self):
        legends=[]
        print(self._INFO['HISTORY'].history)
        for key in self._INFO['HISTORY'].history:
            plt.plot(self._INFO['HISTORY'].history[key])
            legends.append(key)
            plt.ylabel(key)
        
        plt.title('model {0}'.format("lOSS VS ACC"))
        plt.legend(legends, loc='upper left')
        plt.xlabel('epoch')
        plt.show()
#         # summarize history for loss
#         plt.plot(self._HISTORY.history['loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.show()
        
      
