
from idataset import Dataset
from imodel import Model
from ker_model import KER_Model
from tfl_model import TFL_Model

from keras.callbacks import ModelCheckpoint

class Trainer:
    _MODEL = None
    _DATASET = None
    _EPOCHS = 10;
    _BATCH_SIZE=100;
    _CALLBACKS=[]
    
    def __init__(self, ModelID, DatasetID):
        self._DATASET = Dataset(DatasetID);
        self._DATASET.load()
        
        self._MODEL = Model(ModelID, self._DATASET);
        self._MODEL.load();
       
    
    def set_callbacks(self, Callbacks={}):
         for key, val in Callbacks.items():
            if(key == 'per_epoch'):
                self.set_checkpoint_per_epoch(val);
            
    
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
    
    def train(self, Epochs=10, BatchSize=100, Callbacks={}, Verbose=0):
        self._EPOCHS = Epochs;
        self._BATCH_SIZE = BatchSize;
        self.set_callbacks(Callbacks);       
                
        self._MODEL._M.fit( x= self._DATASET._D['X'] \
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
      
