
from idataset import Dataset
from imodel import Model
from ker_model import KER_Model
from tfl_model import TFL_Model

from keras.callbacks import ModelCheckpoint, History, CSVLogger
from custom_callbacks import cf_callbacks
import matplotlib.pyplot as plt

from db.models import training_session, training_log

import os;

class Trainer:
    _MODEL = None
    _DATASET = None
    _MODSET_ID = None
    _EPOCHS = 10;
    _BATCH_SIZE=100;
    _CALLBACKS=[]
    _INFO={}
    _SESSION = None;
    
    def __init__(self, ModelID, DatasetID, LoadLatestCheckpoint=False):
        if (self.is_modset_id(ModelID)):
            self._MODSET_ID = ModelID;
            modelID, datasetID = self.parse_modset_id(self._MODSET_ID);
            
            self._DATASET = Dataset(datasetID);
            self._DATASET.load()

            self._MODEL = Model(modelID, self._DATASET);
            self._MODEL.load();
        else:
            modelID = ModelID;
            datasetID = DatasetID;
            
            self._DATASET = Dataset(datasetID);
            self._DATASET.load()

            self._MODEL = Model(modelID, self._DATASET);
            self._MODEL.load();

            self._MODSET_ID = self.derive_modset_id(self._MODEL, self._DATASET)        

        if (LoadLatestCheckpoint==True):
            self._MODEL.load_latest_checkpoint();
        
        self._SESSION = training_session.TrainingSession(self._MODSET_ID);
            
    def derive_modset_id(self, Model, Dataset):
        MODSET_ID = Model._INFO['GAME'] \
                            +'.'+  Model._INFO['API'] \
                            +'.'+  Model._INFO['BUILD'] \
                            +'.'+  Model._INFO['MAKE'] \
                            + '[' + Dataset._INFO['xnINPUTS'] \
                            +'_'+  Dataset._INFO['xnDRAWS'] \
                            +'_'+  Dataset._INFO['DATA_TYPE'] \
                            + ']';
        print("Modset ID: {0}".format(MODSET_ID))
        return MODSET_ID;
    
    def parse_modset_id(self, modsetID):
        ids = modsetID.split('[')
        mods = ids[0].split('.')
        dats = ids[1][:-1].split('_')
        model_id = '{0}.{1}.{2}.{3}'.format(mods[0],mods[1], mods[2], mods[3])
        dataset_id = '{0}_{1}_{2}_{3}'.format(mods[0],dats[0], dats[1], mods[2])
        return model_id, dataset_id;
        
    def is_modset_id(self, id):
        isModsetID = False;
        ids = id.split('[');
        if len(ids) > 1:
            isModsetID = True;
        return isModsetID;
        
    
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
        custom_callbacks.set_session(self._SESSION)
        self._CALLBACKS.append(custom_callbacks)
        print("[itrainer:set_custom_callbacks] added custom_callback {0}".format(CallBackName))
                
                
    def get_csv_history(self, PerEpoch=1):
        current_folder = os.getcwd() #gives a array with current working folder i.e notebook folder
        db_folder = current_folder + '/db/'
        csv_file= db_folder + 'csv_logs.csv'
        csv_logger = CSVLogger(csv_file, separator=",", append=True);
        self._CALLBACKS.append(csv_logger)
        print("[itrainer:get_csv_histry] added csv_logger callback {0}".format(csv_file))
        
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
        chkpnt_file = self._MODEL._CHECKPOINTS_FOLDER + file_name;
        
        self._MODEL.derive_checkpoints_folder();        
        self._MODEL.ensure_dirs(chkpnt_file); #ensure checkpnt folder exists, folder doesn't work, so use filepath
        
        
        chkpnt = ModelCheckpoint(            
            chkpnt_file,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=PerEpoch);
        self._CALLBACKS.append(chkpnt)
        
        
        chkpnt_file_ckpt = self._MODEL._CHECKPOINTS_FOLDER + file_name + '.ckpt';
        chkpnt_ckpt = ModelCheckpoint(            
            chkpnt_file_ckpt,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=PerEpoch);
        self._CALLBACKS.append(chkpnt_ckpt)
        
        
        chkpnt_file_h5 = self._MODEL._CHECKPOINTS_FOLDER + file_name + '.h5';
        chkpnt_h5 = ModelCheckpoint(            
            chkpnt_file_h5,
            monitor='loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=PerEpoch);
        self._CALLBACKS.append(chkpnt_h5)
        print("[itrainer:set_checkpoint_per_epoch] added model_checkpoint callback")
    
    def train(self, Epochs=10, BatchSize=100, Callbacks={}, Verbose=0):
        self._EPOCHS = Epochs;
        self._BATCH_SIZE = BatchSize;
        self.set_callbacks(Callbacks);
        
        self._SESSION.start_session(self._MODEL._CHECKPOINT_EPOCH, self._EPOCHS, self._BATCH_SIZE)
                
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
        
        self._SESSION.end_session();
        
    
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

    def plt_db_logs(self, ModsetID=None):
        if(ModsetID == None):
            ModsetID = self._MODSET_ID;
        
        trainingLog = training_log.TrainingLog(ModsetID);
        df_logs = trainingLog.get_dataframe();
        df_logs.set_index('train_log_id', inplace=True)
        logs = df_logs.loc[df_logs.modset_id == ModsetID ]

        logs.plot(kind='line',x='epoch',y='loss',ax=plt.gca())
        logs.plot(kind='line',x='epoch',y='metric_value', legend='precision_top10', color='red', ax=plt.gca())

        plt.title('{0} - Loss vs {1}'.format(ModsetID, logs['metric_name'].unique()[0]))
        plt.legend(['loss',logs['metric_name'].unique()[0]])
        plt.show()
        
      
