import numpy as np

class DecisionTreeNode:
    EPSILON = 1e-12

    def __init__(self, output_handlers, weights=None, depth=0, max_depth=None,
                 min_samples_split=2, max_features=None, nb_outputs=0,
                 heuristic_normalization=False):
        self.output_handlers = output_handlers
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.weights = np.ones(len(output_handlers)) if weights is None else np.array(weights)
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.value = None
        self.feature_importances_ = np.zeros(0)
        self.nb_outputs = nb_outputs
        self.heuristic_normalization = heuristic_normalization

    def fit(self, X, y_transformed):
        
        n_samples, n_features = X.shape
        if self.feature_importances_.size == 0:
            self.feature_importances_ = np.zeros(n_features)

        if n_samples < self.min_samples_split or (self.max_depth is not None and self.depth >= self.max_depth):
            self.value = self._leaf_value(y_transformed)
            return

        features = np.arange(n_features) if self.max_features is None else np.random.choice(n_features, self.max_features, replace=False)

        best_score = np.inf
        best_impurity = 0.0

        for f in features:
            X_f = X[:, f]
            sort_idx = np.argsort(X_f)
    
            X_sorted = X_f[sort_idx]
            y_sorted = y_transformed[sort_idx]
            unique_vals = np.unique(X_sorted)
            if len(unique_vals) == 1:
                continue  

            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            if len(thresholds) == 0:
                continue

            left_count = np.searchsorted(X_sorted, thresholds, side='right')
            right_count = n_samples - left_count
            mask_valid = left_count > 0
            mask_valid &= right_count > 0
            thresholds = thresholds[mask_valid]
            left_count = left_count[mask_valid]
            right_count = right_count[mask_valid]

            if len(thresholds) == 0:
                continue

            score_vec = np.zeros(len(thresholds))
            imp_vec = np.zeros(len(thresholds))
            unique_indexes = np.arange(n_samples)
            for t in self.output_handlers:
                s, i_d = t.find_split(y_sorted, unique_indexes, mask_valid, left_count, right_count)

                if (score_vec.shape[0] != s.shape[0]):
                    print(self.min_samples_split)
                    print(unique_vals)
                    print(unique_vals.shape)
                    
                    print(mask_valid)
                    print(mask_valid.shape)
                    
                    print(y_transformed)
                    print(y_transformed.shape)
                    
                    print(s)
                    print(s.shape)
                    print(score_vec)
                    print(score_vec.shape)
                    print(mask_valid)
                    print(sum(mask_valid))
                    print(mask_valid.shape)

                    print(left_count)
                    print(left_count.shape)
                    print(right_count)
                    print(right_count.shape)
                score_vec += s
                imp_vec += i_d

            best_idx = np.argmin(score_vec)
            if score_vec[best_idx] < best_score:
                best_score = score_vec[best_idx]
                self.feature_index = f
                self.threshold = thresholds[best_idx]
                best_impurity = imp_vec[best_idx]

        if self.feature_index is None:
            self.value = self._leaf_value(y_transformed)
            return

        self.feature_importances_[self.feature_index] += best_impurity

        left_mask = X[:, self.feature_index] <= self.threshold
        right_mask = X[:, self.feature_index] > self.threshold

        self.left = DecisionTreeNode(
            self.output_handlers,
            weights=self.weights,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            nb_outputs=self.nb_outputs,
            heuristic_normalization=self.heuristic_normalization
        )
        self.left.fit(X[left_mask], y_transformed[left_mask])
        self.feature_importances_ += self.left.feature_importances_

        self.right = DecisionTreeNode(
            self.output_handlers,
            weights=self.weights,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            nb_outputs=self.nb_outputs,
            heuristic_normalization=self.heuristic_normalization
        )
        self.right.fit(X[right_mask], y_transformed[right_mask])
        self.feature_importances_ += self.right.feature_importances_

    def _leaf_value(self, y_transformed):
        return np.hstack([t.prototype(y_transformed) for t in self.output_handlers])

    def traverse(self, X):
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, self.nb_outputs))
        stack = [(np.arange(n_samples), self)]
        while stack:
            mask, node = stack.pop()
            if node.value is not None:
                preds[mask] = node.value
            else:
                left_mask = mask[X[mask, node.feature_index] <= node.threshold]
                right_mask = mask[X[mask, node.feature_index] > node.threshold]
                if len(left_mask) > 0:
                    stack.append((left_mask, node.left))
                if len(right_mask) > 0:
                    stack.append((right_mask, node.right))
        return preds

    def predict(self, X):
        return self.traverse(X)

    def predict_proba(self, X):
        return self.traverse(X)
