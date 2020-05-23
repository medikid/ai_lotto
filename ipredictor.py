
import numpy as np
from idataset import Dataset
from imodel import Model
from ker_model import KER_Model

class iPredictor:
    _MODEL = None;
    _MODELS = {}
    _DATASET = None;
    _IDS_TEST = None;
    _X_TEST = None;
    _Y_TEST = None;
    _Y_HAT = None;
    
    def __init__(self, ModelID, Dataset, loadArchived=False):
        self._DATASET = Dataset;
        #self._DATASET.load();
        self._IDS_TEST = self._DATASET._D['IDs_test']
        self._X_TEST = self._DATASET._D['X_test']
        self._Y_TEST = self._DATASET._D['Y_test']
        
        self._MODEL = Model(ModelID, self._DATASET);
        self._MODEL.load();
        
    def test(self):
        self._Y_HAT = self._MODEL.predict(self._X_TEST)
        
    def predict(self, X_predict):
        Y_hat = self._M.predict(X_predict)
        return Y_hat
    
    def predict_batch(self):
        self._Y_HAT = self._MODEL._M.predict(self._X_TEST);
        for draw_id, y_hat, y_test in zip(self._IDS_TEST, self._Y_HAT, self._Y_TEST):
#             self.print_prediction(y_hat, y_test, draw_id);
            self.print_prediction_by_rank(y_hat, y_test, draw_id);
#             self.print_prediction_by_number(y_hat, y_test, draw_id);                
                                       
        
    def print_prediction(self, Predicted, Actual, DrawID=0):
        predicted = Predicted.flatten();
        #sorted_indexes = np.argsort(predicted)
        actual = Actual.flatten();
        
        print("\n [{0}] n.Predicted[Actual]".format(DrawID))
        for n in range(len(predicted)):
            print("{0}. {1} [{2}]".format(n+1,predicted[n],actual[n]) )
    
    #fix this
    def print_prediction_by_rank(self, Predicted, Actual, DrawID=0):
        predicted = Predicted.flatten();
        ranked_indexes =  np.argsort(predicted); # ranked_index[rank] = index
        ranks_0 = np.argsort(ranked_indexes); #0-indexed rank, so add+1
        ranks = [80] - ranks_0 #will covert it to desc ordered 1-indexed
        actual = Actual.flatten();
        
                
        print("\n [{0}] Rank.[n] Predicted [Actual] ".format(DrawID))
        for r in range(len(predicted)):
            i = ranked_indexes[r]
            print("R{0}.[#{1}] {2} [{3}]".format(r+1,i+1,predicted[i],actual[i]) )
            
    def print_prediction_by_number(self, Predicted, Actual, DrawID=0):
        predicted = Predicted.flatten();
        ranked_indexes = np.argsort(predicted); # ranked_index[rank] = index
        ranks_0 = np.argsort(ranked_indexes); #0-indexed ascending rank, so add+1
        ranks = [80] - ranks_0 #will covert it to desc ordered 1-indexed 80-rank0=rank80
        actual = Actual.flatten();
        
        print("\n [{0}] n.Predicted[Actual]-Rank#".format(DrawID))
        for n in range(len(predicted)):
            print("{0}. {1} [{2}] - Rank#{3}".format(n+1,predicted[n],actual[n],ranks[n]+1) )
            
