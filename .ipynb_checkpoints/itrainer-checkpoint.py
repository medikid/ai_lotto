
from idataset import Dataset
from imodel import Model

from keras.callbacks import ModelCheckpoint

class Trainer:
    _MODEL = None
    _DATASET = None
    _EPOCHS = 10;
    _BATCH_SIZE=100;
    _CALLBACKS=[]
    
    def __init__(self, Model, Dataset):
        self._MODEL = Model;
        self._DATASET = Dataset;
        pass;
    
    def set_callbacks(self, Callbacks={}):
         for key, val in Callbacks:
            if(key == 'per_epoch'):
                self.set_checkpoint_per_epoch(val);
            
    
    def set_checkpoint_per_epoch(self, PerEpoch=5):
        filename = self._MODEL._INFO['GAME'] + '.';
        filename += self._MODEL._INFO['API']  + '.';
        filename += self._MODEL._INFO['BUILD']  + '.';
        filename += self._MODEL._INFO['MAKE']  + '.';
        filename += 'e{epoch:04d}';
        
        self._MODEL.derive_checkpoints_folder();    
        file_path = self._CHECKPOINTS_FOLDER + filename + '.h5'
        chkpnt = ModelCheckpoint(            
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=PerEpoch);
        self._CALLBACKS.append(ckhpnt)
    
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
                            , initial_epoch = self._MODEL._M._INITIAL_EPOCH \
                            #, steps_per_epoch=None \
                            #, validation_steps=None \
                            #, validation_freq=1 \
                            #, max_queue_size=10 \
                            #, workers=1 \
                            #, use_multiprocessing=False\
                            );
      
            
        
        
