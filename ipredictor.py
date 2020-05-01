
class Predictor:
    _M = None
    _D = None
    _PREDS = {}
    def __init__(self, Model):
        self._M = Model;
        
    def predict(self, X_test)
        y_hat = self._M.predict(X_test);
        return y_hat;
    
    def predict_batch(self, Dataset):
        IDs_test = Dataset['IDs_test'];
        X_test = Dataset['X_test'];
        Y_test = Dataset['Y_test'];
        
        
