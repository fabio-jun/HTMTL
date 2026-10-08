from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class RFStandard(BaseEstimator):
    def __init__(self, n_estimators=100, random_state=None, output_types=None, n_jobs=None, **kwargs):
        """
        Random Forest wrapper for multiple target types:
        - Binary targets: leave as-is
        - Regression targets: standardize
        - Classification targets: one-hot encode
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.rf_kwargs = kwargs
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.rf_kwargs
        )
        self.output_types_ = output_types
        self.scalers = {}
        self.encoders = {}
        self.output_slices = {}
        
    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray) and y.ndim == 1:
            y = pd.DataFrame(y, columns=["target"])
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        
        self.y_columns = y.columns
        y_transformed = []
        current_idx = 0
        
        for col, typ in zip(self.y_columns, self.output_types_):
            if typ == "binary":
                arr = y[[col]].values
                y_transformed.append(arr)
                self.output_slices[col] = (current_idx, current_idx + 1)
                current_idx += 1
            elif typ == "regression":
                scaler = StandardScaler()
                arr = scaler.fit_transform(y[[col]])
                self.scalers[col] = scaler
                y_transformed.append(arr)
                self.output_slices[col] = (current_idx, current_idx + 1)
                current_idx += 1
            elif typ == "classification":
                encoder = OneHotEncoder(sparse_output=False)
                arr = encoder.fit_transform(y[[col]])
                self.encoders[col] = encoder
                y_transformed.append(arr)
                self.output_slices[col] = (current_idx, current_idx + arr.shape[1])
                current_idx += arr.shape[1]
            else:
                raise ValueError(f"Unknown output type '{typ}' for column '{col}'")
        
        y_transformed = np.hstack(y_transformed)
        self.model.fit(X, y_transformed)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        
        result = pd.DataFrame(index=np.arange(preds.shape[0]))
        
        for col, typ in zip(self.y_columns, self.output_types_):
            start, end = self.output_slices[col]
            pred_slice = preds[:, start:end]
            
            if typ == "binary" or typ == "regression":
                if typ == "regression" and col in self.scalers:
                    pred_slice = self.scalers[col].inverse_transform(pred_slice)
                result[col] = pred_slice.ravel()
            elif typ == "classification":
                pred_slice = self.encoders[col].inverse_transform(pred_slice)
                result[col] = pred_slice.ravel()
        
        if result.shape[1] == 1:
            return result.iloc[:, 0]
        return result
