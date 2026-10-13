import numpy as np
from .heuristics import variance

class ContinuousOutput:
    def __init__(self, index, y=None, weight=1.0, heuristic_normalization=True):
        self.index = index
        self.weight = weight
        self.normalization_factor = np.var(y) if (heuristic_normalization and y is not None) else None
        self.output_name = "target_" + str(index)

    def find_split(self, y_transformed, unique_indices, mask_valid, left_count, right_count):
        y_col = y_transformed[:, self.index].ravel()
        norm_factor = self.normalization_factor
        if norm_factor is not None and norm_factor == 0:
            norm_factor = None
        return variance(y_col, self.weight, unique_indices, mask_valid, left_count, right_count, norm_factor)

    def prototype(self, y_transformed):
        return y_transformed[:, self.index].mean(axis=0)
