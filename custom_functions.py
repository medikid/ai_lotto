
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


class cf_metrics:
    
##############################################################    
# Based on custom score defn using keras backend
##############################################################
    @staticmethod
    def recall_a(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    @staticmethod
    def precision_a(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    @staticmethod
    def f1_score_a(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        
        f1Score = 2*((precision*recall)/(precision+recall+K.epsilon()));
        return f1Score

    
    
##############################################################    
# Based on custom score defn using SK_LEARN confusion matrix

# DOES NOt WORK WItH KERAS check
# https://datascience.stackexchange.com/questions/74419/evaluate-keras-model-with-scikit-learn-metrics
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
##############################################################
    @staticmethod
    def recall_b(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        recall = np.diag(cm) / np.sum(cm, axis = 1)
        return recall;
    
    @staticmethod    
    def precision_b(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        precision = np.diag(cm) / np.sum(cm, axis = 0)
        return precision;
    
    @staticmethod
    def f1_score_b(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        precision = np.diag(cm) / np.sum(cm, axis = 0)        
        recall = np.diag(cm) / np.sum(cm, axis = 1)
        f1Score = 2 ((precision * recall) / (precision + recall));
        return f1Score;
    
##############################################################    
# Based on SK_LEARN F1_SCORE, RECALL_SCORE and PRECISION_SCORE

# DOES NOt WORK WItH KERAS check
# https://datascience.stackexchange.com/questions/74419/evaluate-keras-model-with-scikit-learn-metrics
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
##############################################################
    @staticmethod
    def recall_c(y_true, y_pred):
        recall = recall_score(y_true, y_pred)
        return recall
    
    @staticmethod
    def precision_c(y_true, y_pred):
        precision = precision_score(y_true, y_pred)
        return precision

    @staticmethod
    def f1_score_c(y_true, y_pred):
        f1Score = f1_score(y_true, y_pred)
        return f1Score
    
###############################################################    
# Based on custom score defn using keras backend
# modified to add threshold and top-k args
##############################################################
    @staticmethod
    def recall_aa(y_true, y_pred, top_k = None, threshold = None):
        pred = K.cast(K.greater(pred, threshold), K.floatx())
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    @staticmethod
    def precision_aa(y_true, y_pred, top _k = None, threshold = None):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    @staticmethod
    def f1_score_aa(y_true, y_pred, top_k = None, threshold = None):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        
        f1Score = 2*((precision*recall)/(precision+recall+K.epsilon()));
        return f1Score      
    
    

class cf_losses:
    
    def loss_a(y_true, y_pred):
        pass;
    

