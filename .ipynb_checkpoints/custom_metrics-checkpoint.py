# custom precision
from keras.metrics import Precision

class cf_precision(Precision):
    def __init__(self, name=None, top_k=10):
        super(cf_precision, self).__init__(name=name, top_k=top_k)
        self.name = name;
        #self.dtype = dtype
        #self.thresholds = thresholds;
        self.top_k = top_k;
        #self.class_id = class_id;

    def get_config(self):
        config = super(cf_precision, self).get_config()
        config.update({
            #"units": self.units,
            'name':self.name,
            #'dtype': self.dtype,
            #'thresholds': self.thresholds,
            'top_k': self.top_k,
            #'class_id': self.class_id
        })
        return config
