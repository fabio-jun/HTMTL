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
    # def fit(self, X, Y):
    #     """
    #     Train one HMTRF per output.
    #     X: shape (n_samples, n_features)
    #     Y: shape (n_samples, n_outputs)
    #     output_types: list of 'regression' or 'classification'
    #     """
    #     n_samples, n_outputs = Y.shape
    #     self.models_ = []

    #     for i in range(n_outputs):
    #         model = HMTRF(
    #             n_estimators=self.base_estimator.n_estimators,
    #             max_depth=self.base_estimator.max_depth,
    #             min_samples_split=self.base_estimator.min_samples_split,
    #             max_features=self.base_estimator.max_features,
    #             weights=self.base_estimator.weights,
    #             weights_scheme=self.base_estimator.weights_scheme,
    #             heuristic_normalization=self.base_estimator.heuristic_normalization,
    #             n_jobs=self.base_estimator.n_jobs,
    #             output_types=[self.output_types_[i]]
    #         ) 
    #         model.output_types = [self.output_types_[i]]
    #         model.fit(X, Y[:, i:i+1])
    #         self.models_.append(model)

    #     return self

    def predict(self, X):
        """
        Predict all outputs using the local HMTRFs.
        Returns shape (n_samples, n_outputs)
        """
        # n_samples = X.shape[0]
        # n_outputs = len(self.models_)
        # Y_pred = np.zeros((n_samples, n_outputs))

        # for i, model in enumerate(self.models_):
        #     pred = model.predict(X)
        #     # if self.output_types_[i] == 'classification':
        #     #     # ensure integer class labels
        #     #     pred = pred.astype(int)
        #     Y_pred[:, i] = pred
        all_preds = np.zeros(n_samples, n_outputs)

        for i, t in enumerate(self.output_types_):
            model = chain_models[idx]
            pred = np.ravel(model.predict(X_ext))  # ensure 1D array
#            print(pred)
            final_preds[:, i] = pred

        return final_preds
        # return Y_pred
    def predict_proba(self, X):
        """
        Predict all outputs using the local HMTRFs.
        Returns shape (n_samples, n_outputs)
        """
        # n_samples = X.shape[0]
        # n_outputs = self.models_[0].nb_outputs_transformed
        # Y_pred = np.zeros((n_samples, n_outputs))
        y_pred = []
        for i, model in enumerate(self.models_):
#            print(model)
            if self.output_types_[i] == 'classification':
                y_pred.append(model.predict_proba(X)[:,1].reshape(-1,1))   
            else:
                y_pred.append(model.predict(X).ravel().reshape(-1,1))
        y_pred = np.hstack(y_pred)
#        print(y_pred.shape)
        return y_pred
                
# Generate toy mixed-output data
# np.random.seed(42)
# X = np.random.rand(100, 5)
# y_reg = np.random.rand(100, 1)
# y_clf = np.random.randint(0, 3, size=(100, 2))
# Y = np.hstack([y_reg, y_clf])

# output_types = ['regression', 'classification', "classification"]

# hmtrf = HMTRF(n_estimators=5, max_depth=4, max_features=3)
# local_model = Local(base_estimator=hmtrf, output_types=output_types)
# local_model.fit(X, Y)
# Y_pred = local_model.predict(X)

# print(Y)
# print("Predictions shape:", Y_pred.shape)
# print("First 5 predictions:\n", Y_pred)
