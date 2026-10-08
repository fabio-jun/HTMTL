import numpy as np
from sklearn.preprocessing import OneHotEncoder
from .heuristics import gini

class CategoricalOutput:
    def __init__(self, index, y, weight=1.0, heuristic_normalization=True):
        enc = OneHotEncoder(handle_unknown='error', sparse_output=False, dtype=int)
        enc.fit(y.reshape(-1, 1))
        self.label_encoder = enc
        self.index = [index, index + len(enc.categories_[0])]
        self.weight = weight
        # Gini
        self.normalization_factor = 1 - 1 / len(np.unique(y)) if heuristic_normalization else None

    def find_split(self, y_transformed, unique_indices, mask_valid, left_count, right_count):
        norm_factor = self.normalization_factor
        if norm_factor is not None and norm_factor == 0:
            norm_factor = None
        return gini(y_transformed[:, self.index[0]:self.index[1]], self.weight, unique_indices, mask_valid, left_count, right_count, norm_factor)

    def prototype(self, y_transformed):
        return np.mean(y_transformed[:, self.index[0]:self.index[1]], axis=0)
    
    # Returns the number of categories learned
    def get_nb_categories(self):
        return len(self.label_encoder.categories_[0])
