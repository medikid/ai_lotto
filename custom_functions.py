
from keras import backend as K
from sklearn.metrics import confusion_matrix
import numpy as np

class cf_metrics:

    def recall_a(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_a(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_a(y_true, y_pred):
        precision = precision_a(y_true, y_pred)
        recall = recall_a(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    def recall_b(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        recall = np.diag(cm) / np.sum(cm, axis = 1)
        return recall;
        
    def precision_b(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        precision = np.diag(cm) / np.sum(cm, axis = 0)
        return precision;
    
    
    
class cf_losses:
    
    def loss_a(y_tru, y_pred):
        pass;


