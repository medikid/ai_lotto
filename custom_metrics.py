# custom precision
from keras.metrics import Precision
#from keras.backend import tf
import tensorflow as tf

class cf_precision(Precision):
    def __init__(self, name=None, top_k=10, dtype=None, thresholds=None, class_id=None):
        super(cf_precision, self).__init__(name=name, top_k=top_k)
        self.name = name;
        #self.dtype = dtype
        #self.thresholds = thresholds;
        self.top_k = top_k;
        #self.class_id = class_id;

    def get_config(self):
        config = super(cf_precision, self).get_config()
        config.update({
            'name': self.name,
            #'dtype': tf.convert_to_tensor(self.dtype),
            #'thresholds': tf.convert_to_tensor(self.thresholds),
            'top_k': self.top_k,
            #'class_id': tf.convert_to_tensor(self.class_id)
        })
        return config
