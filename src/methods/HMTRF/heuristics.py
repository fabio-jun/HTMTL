import numpy as np
from numba import njit

@njit
# y_col: output values
# unique_indices: sample indices after sorting by the feature;
# mask_valid: indicates which splits are valid;

def variance(y_col, weight, unique_indices, mask_valid, left_count, right_count, normalization_factor):
    #n_splits = len(mask_valid)
    n_splits = sum(mask_valid)
    # Creates an array that stores the score for every split (lower is better)
    score_vector = np.zeros(n_splits)
    impurity_decrease_vector = np.zeros(n_splits)
    # Calculates the variance of the output before splitting
    total_var = np.var(y_col) if len(y_col) > 0 else 0.0

    # Evaluating every split
    for i in range(n_splits):
        l_len = left_count[i]
        r_len = right_count[i]
        if l_len <= 0 or r_len <= 0:
            score_vector[i] = np.inf
            impurity_decrease_vector[i] = 0.0
            continue
        # From the start to l_len
        l_idx = unique_indices[:l_len]
        # From l_len to the end 
        r_idx = unique_indices[l_len:l_len+r_len]

        if len(l_idx) == 0 or len(r_idx) == 0:
            score_vector[i] = np.inf
            impurity_decrease_vector[i] = 0.0
            continue

        y_left = y_col[l_idx]
        y_right = y_col[r_idx]

        split_var = 0.0
        # Initializes the combined variance of both groups
        # Multiplying by group size gives larger groups proportionally more influence
        if len(y_left) > 0:
            split_var += np.var(y_left) * len(y_left)
        if len(y_right) > 0:
            split_var += np.var(y_right) * len(y_right)
        total_len = len(y_left) + len(y_right)
        if total_len == 0:
            score_vector[i] = np.inf
            impurity_decrease_vector[i] = 0.0
            continue

        # Converts the previous sum into a weighted average
        split_var /= total_len

        if normalization_factor is not None and normalization_factor > 0:
            split_var /= normalization_factor

        score_vector[i] = split_var * weight
        impurity_decrease_vector[i] = (total_var - split_var) * weight
#    print(score_vector.shape)
    return score_vector, impurity_decrease_vector


@njit
def gini(y_split, weight, unique_indices, mask_valid, left_count, right_count, normalization_factor):
    n_splits = sum(mask_valid)
    n_samples, n_classes = y_split.shape
    score_vector = np.zeros(n_splits)
    impurity_decrease_vector = np.zeros(n_splits)

    for i in range(n_splits):
        l_len = left_count[i]
        r_len = right_count[i]
        if l_len <= 0 or r_len <= 0:
            score_vector[i] = np.inf
            impurity_decrease_vector[i] = 0.0
            continue

        l_idx = unique_indices[:l_len]
        r_idx = unique_indices[l_len:l_len+r_len]

        if len(l_idx) == 0 or len(r_idx) == 0:
            score_vector[i] = np.inf
            impurity_decrease_vector[i] = 0.0
            continue

        y_left = y_split[l_idx, :]
        y_right = y_split[r_idx, :]

        sum_left = y_left.sum()
        sum_right = y_right.sum()
        if sum_left == 0: sum_left = 1.0
        if sum_right == 0: sum_right = 1.0

        # Sums each class column separately and devides by the total
        p_left = y_left.sum(axis=0) / sum_left
        p_right = y_right.sum(axis=0) / sum_right

        gini_left = 1.0 - np.sum(p_left**2)
        gini_right = 1.0 - np.sum(p_right**2)

        total_len = len(l_idx) + len(r_idx)
        if total_len == 0: total_len = 1.0

        # Calculates the weighted average impurity
        total_gini = (len(l_idx)*gini_left + len(r_idx)*gini_right)/total_len

        if normalization_factor is not None and normalization_factor > 0:
            total_gini /= normalization_factor

        score_vector[i] = total_gini * weight
        impurity_decrease_vector[i] = (1.0 - total_gini) * weight

    return score_vector, impurity_decrease_vector
