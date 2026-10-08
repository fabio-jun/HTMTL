import numpy as np
from joblib import Parallel, delayed
from .DecisionTreeNode import DecisionTreeNode
from .ContinuousOutput import ContinuousOutput
from .CategoricalOutput import CategoricalOutput

# Helper used by joblib, it trains one tree. Each parallel worker gets everything
# it needs to build one tree independently (top-level function (outside HMTRF))
def fit_single_tree_worker(X, y_transformed, seed, output_types, y_original, 
                           weights_scheme, weights, max_depth, min_samples_split, 
                           max_features, nb_outputs, heuristic_normalization):
    output_handlers = []
    # i = output, t = type of that output column
    for i, t in enumerate(output_types):
        if t == "classification":
            # [:, i] all rows, and only column index 1
            handler = CategoricalOutput(i, y_original[:, i], weights[i], heuristic_normalization)
        else:
            handler = ContinuousOutput(i, y_original[:, i], weights[i], heuristic_normalization)
        output_handlers.append(handler)

    rng = np.random.default_rng(seed)
    # Gets the number of rows in X
    n_samples = X.shape[0]
    # Randomly chooses row indices
    # rng.choice(possible_values, number_to_choose, replace=True) replace=True allow repeats
    indices = rng.choice(n_samples, n_samples, replace=True)
    # Selects rows from X using those random indices
    X_sample = X[indices]
    # Selects the matching target rows from y_transformed
    y_sample = y_transformed[indices]

    tree_weights = np.random.rand(len(output_types)) if weights_scheme == "random" else weights

    tree = DecisionTreeNode(
        output_handlers,
        weights=tree_weights,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        nb_outputs=nb_outputs,
        heuristic_normalization=heuristic_normalization
    )
    tree.fit(X_sample, y_sample)
    return seed, tree, tree_weights

class HMTRF:
    EPSILON = 1e-12
    possible_output_types = {"classification", "regression", "binary"}
 
    def __init__(self, output_types=None, weights=None, weights_scheme="random", n_estimators=200,
                 max_depth=None, min_samples_split=2, max_features=None,
                 heuristic_normalization=False, n_jobs=-1, progress_label=None):
        # Number of trees
        self.n_estimators = n_estimators
        # Tree max depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        # Number of feature columns to try at each split
        self.max_features = max_features
        self.output_types = output_types
        # Manual wight selection
        self.weights = weights
        # "uniform" or "random"
        self.weights_scheme = weights_scheme
        # Number outputs
        self.nb_outputs = 0
        # Number outputs transformed
        self.nb_outputs_transformed = 0
        self.heuristic_normalization = heuristic_normalization
        self.output_handlers = []
        self.tree_weights = None
        # Parallel job count (cpu)
        self.n_jobs = n_jobs
        self.progress_label = progress_label

    def _log_progress(self, message):
        if self.progress_label:
            print(f"[hmtrf progress] {self.progress_label}: {message}", flush=True)

    def _infer_output_types(self, y):
        types = []
        # For each target column in y (output)
        for i in range(y.shape[1]):
            ## All rows, column i
            # If a target column is float, treat it as regression
            if np.issubdtype(y[:, i].dtype, np.floating):
                types.append("regression")
            else:
                types.append("classification")
        return types

    def _validate_and_count_output_types(self):
        # Initializes a dictionary with all possible output types with key 0
        # zip to crate pairs for the dict()
        #{
        #   "classification": 0,
        #   "regression": 0,
        #   "binary": 0,
        #}
        self.count_output_types = dict(zip(self.possible_output_types, np.zeros(len(self.possible_output_types), dtype=int)))
        for output_type in self.output_types:
            if output_type not in self.possible_output_types:
                raise ValueError(f"Output type {output_type} not recognized")
            self.count_output_types[output_type] += 1


    def _initialize_weights(self):
        # Total number of outputs
        n_outputs = len(self.output_types)
        if self.weights_scheme == "uniform":
            #np.full(shape, fill_value)
            self.weights = np.full(n_outputs, 1.0 / n_outputs)

        elif self.weights_scheme == "random":
            # Array with n_outputs with random numbers between 0 and 1
            w = np.random.rand(n_outputs)
            self.weights = w / w.sum()        
        elif self.weights is None:
            self.weights = np.ones(n_outputs)

    # Called in fit_single_tree_worker 
    def _get_random_weights(self):
        n_outputs = len(self.output_types)
        w = np.random.rand(n_outputs)
        return w / w.sum()
    

    def _instantiate_encoders_classification(self, y):
        # i = output column index, t = type of that column
        # enumerate returns a tuple index and value
        for i, t in enumerate(self.output_types):
            if t == "classification":
                handler = CategoricalOutput(i, y[:, i], self.weights[i], self.heuristic_normalization)
                self.nb_outputs += handler.get_nb_categories()
            elif t == "regression" or t == "binary":
                handler = ContinuousOutput(i, y[:, i], self.weights[i], self.heuristic_normalization)
                self.nb_outputs += 1
            self.output_handlers.append(handler)

    # Will collect transformed target blocks
    def transform_output(self, y):
        y_transformed = []
        for i, t in enumerate(self.output_types):
            if t == "classification":
                # Turns classification labels in one-hot rows
                y_transformed.append(self.output_handlers[i].label_encoder.transform(y[:, i].reshape(-1, 1)))
                # Counts how many columns were added
                self.nb_outputs_transformed += self.output_handlers[i].get_nb_categories()
            # Regression/binary targets stay as one numeric column
            else:
                y_transformed.append(y[:, i].reshape(-1, 1))
                self.nb_outputs_transformed += 1
                # Concatenates all transformed target blocks into one matrix
        return np.hstack(y_transformed).astype(float)

    # Converts expanded predictions back to original output columns.
    def _reverse_transform_encoded_classification(self, preds):
        start_index = 0
        untransformed_preds = []
        for i, t in enumerate(self.output_types):
            if t == "regression":
                untransformed_preds.append(preds[:, i].reshape(-1, 1))
                start_index += 1
            else:
                nb_categories = self.output_handlers[i].get_nb_categories()
                max_per_row = preds[:, start_index:start_index + nb_categories].argmax(axis=1, keepdims=True).astype(int)
                untransformed_preds.append(max_per_row.reshape(-1, 1))
                start_index += nb_categories
        return np.hstack(untransformed_preds)

    # Train one tree
    def _fit_single_tree(self, X, y_transformed, seed):
        rng = np.random.default_rng(seed)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y_transformed[indices]

        tree_weights = self._get_random_weights() if self.weights_scheme == "random" else self.weights

        tree = DecisionTreeNode(
            self.output_handlers,
            weights=tree_weights,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            nb_outputs=self.nb_outputs,
            heuristic_normalization=self.heuristic_normalization
        )
        tree.fit(X_sample, y_sample)
        return tree, tree_weights

    # Trains the forest
    def fit(self, X, y):
        if self.output_types is None:
            self.output_types = self._infer_output_types(y)
        else:
            self._validate_and_count_output_types()

        self._initialize_weights()
        # Creates one handler per target column 
        self._instantiate_encoders_classification(y)
        y_transformed = self.transform_output(y)
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)

        self._log_progress(f"starting {self.n_estimators} trees")

        # Train many trees in parallel
        result_stream = Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            # Prepares a function call, but does not execute it imediately 
            delayed(fit_single_tree_worker)(
                X,
                y_transformed,
                seed,
                self.output_types,
                y,  # original y for reconstructing encoders
                self.weights_scheme,
                self.weights,
                self.max_depth,
                self.min_samples_split,
                self.max_features,
                self.nb_outputs,
                self.heuristic_normalization
            )
            for seed in range(self.n_estimators)
        )

        self.trees = [None] * self.n_estimators
        # Creates weight matrix (trees x tasks)
        self.tree_weights = np.zeros((self.n_estimators, len(self.output_types)))
        for completed, (seed, tree, tw) in enumerate(result_stream, start=1):
            self.trees[seed] = tree
            self.feature_importances_ += tree.feature_importances_
            self.tree_weights[seed] = tw
            self._log_progress(f"trained tree {completed}/{self.n_estimators}")

        # Normalize feature importances so they sum to 1
        denom = self.feature_importances_.sum()
        if denom > 0:
            self.feature_importances_ /= denom

    
    def _combine_predictions_trees(self, preds):
        results = []
        for i, t in enumerate(self.output_types):
            # Selects all predictions for all trees and samples for task i
            column_data = preds[:, :, i]
             # Calculates mean of trees for each sample
            if t == "regression":
                # axis=0 combines rows, that is trees
                res = column_data.mean(axis=0)
            # Classification or binary
            else:
                # Applies a function in each column of column_data
                res = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=column_data)
            # Gathers task results as columns
            results.append(res)
            # Final form (n of samples, n of tasks)
        return np.stack(results, axis=-1)

    # Calculates mean of predictions from the trees
    def predict_proba(self, X):
        all_preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_preds, axis=0)

    # Combine trees
    def predict(self, X):
        all_preds = np.array([self._reverse_transform_encoded_classification(tree.predict(X)) for tree in self.trees])
        return self._combine_predictions_trees(all_preds)

    def print_tree(self):
        for t in self.trees:
            t.print_tree()
            print("--------")
