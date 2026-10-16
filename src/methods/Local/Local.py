import numpy as np
import copy
from HMTRF import HMTRF
from joblib import Parallel, delayed
class Local:
    """
    Local approach: a separate HMTRF per output.
    Supports mixed outputs (regression + classification).
    """
    def __init__(self, base_regressor=None, base_classifier=None, output_types = None):
        self.base_regressor = base_regressor
        self.base_classifier = base_classifier
        self.models_ = []
        self.output_types_ = output_types

    def fit(self, X, Y):
        """
        Train one HMTRF per output.
        X: shape (n_samples, n_features)
        Y: shape (n_samples, n_outputs)
        """
        n_samples, n_outputs = Y.shape
        self.models_ = []

        for i in range(n_outputs):
            if self.output_types_[i] == "regression":
                model = copy.deepcopy(self.base_regressor)
            else:
                model = copy.deepcopy(self.base_classifier)

            model.output_types = [self.output_types_[i]]
            model.fit(X, Y[:, i:i+1].ravel())
            self.models_.append(model)

        return self


    def predict(self, X):
        """
        Predict all outputs using the local HMTRFs.
        Returns shape (n_samples, n_outputs)
        """
        all_preds = np.zeros(n_samples, n_outputs)

        for i, t in enumerate(self.output_types_):
            model = chain_models[idx]
            pred = np.ravel(model.predict(X_ext))  # ensure 1D array
            final_preds[:, i] = pred
            
        return final_preds
    
    def predict_proba(self, X):
        """
        Predict all outputs using the local HMTRFs.
        Returns shape (n_samples, n_outputs)
        """
        y_pred = []
        for i, model in enumerate(self.models_):
            if self.output_types_[i] == 'classification':
                y_pred.append(model.predict_proba(X)[:,1].reshape(-1,1))   
            else:
                y_pred.append(model.predict(X).ravel().reshape(-1,1))
        y_pred = np.hstack(y_pred)
        return y_pred
                
