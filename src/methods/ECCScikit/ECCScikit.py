import numpy as np
import copy
from itertools import permutations

class ECCScikit:
    """
    Ensemble of Classifier Chains using separate base estimators for
    regression and classification, supporting mixed outputs.
    """
    def __init__(self, base_regressor=None, base_classifier=None, n_chains=10, random_state=None, output_types=None):
        """
        base_regressor: estimator for regression outputs
        base_classifier: estimator for classification outputs
        output_types: list like ['regression', 'classification', ...]
        """
        self.base_regressor = base_regressor
        self.base_classifier = base_classifier
        self.n_chains = n_chains
        self.random_state = random_state
        self.output_types_ = output_types
        self.chains_ = []
        self.orders_ = []

    def fit(self, X, Y):
        n_samples, n_outputs = Y.shape
        rng = np.random.default_rng(self.random_state)

        # Generate all possible unique orders
        all_orders = np.array(list(permutations(range(n_outputs))))

        # If n_chains=="all", use all permutations; else sample without repetition
#        self.n_chains = "all"
        
        if self.n_chains == "all":
            self.orders_ = all_orders
        else:
            rng.shuffle(all_orders)
            self.orders_ = all_orders[:self.n_chains]
#        print(self.orders_)
        self.chains_ = []

        for order in self.orders_:
            X_ext = X.copy()
            chain_models = []

            for idx in order:
                if self.output_types_[idx] == "regression" or self.output_types_[idx] == "binary":
                    model = copy.deepcopy(self.base_regressor)
                else:
                    model = copy.deepcopy(self.base_classifier)

                model.fit(X_ext, Y[:, idx:idx+1].ravel())
                chain_models.append(model)

#                print(model.predict(X_ext).shape)
                # Append true output for training
                if self.output_types_[idx] == "regression" or self.output_types_[idx] == "binary":
                    X_ext = np.hstack([X_ext, model.predict(X_ext).reshape(-1,1)])
                else:
                    #print(model.predict_proba(X_ext))
                    X_ext = np.hstack([X_ext, model.predict_proba(X_ext)]) 


            self.chains_.append(chain_models)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_outputs = len(self.output_types_)
        all_preds = np.zeros((len(self.orders_), n_samples, n_outputs))

        for chain_idx, (order, chain_models) in enumerate(zip(self.orders_, self.chains_)):
            X_ext = X.copy()
            chain_preds = np.zeros((n_samples, n_outputs))

            for idx, out_idx in enumerate(order):
                model = chain_models[idx]
                pred = np.ravel(model.predict(X_ext))  # ensure 1D array
                chain_preds[:, out_idx] = pred
                X_ext = np.hstack([X_ext, pred[:, np.newaxis]])

            all_preds[chain_idx] = chain_preds

        # Aggregate predictions
        final_preds = np.zeros((n_samples, n_outputs))
        for i, t in enumerate(self.output_types_):
            if t == "regression":
                final_preds[:, i] = all_preds[:, :, i].mean(axis=0)
            else:
                for j in range(n_samples):
                    votes = all_preds[:, j, i].astype(int)
                    final_preds[j, i] = np.bincount(votes).argmax()

        return final_preds
    def predict_proba(self, X):
        """
        Predict regression outputs (averaged) and classification outputs (averaged probabilities)
        Returns a single 2D array: rows=samples, columns=regression + class probabilities
        """
        n_samples = X.shape[0]
        n_outputs = len(self.output_types_)
        all_preds = [[] for _ in range(n_outputs)]  # per output, store predictions from all chains
        class_labels = [None] * n_outputs  # store class labels for each classification output

        # Step 1: Collect predictions from all chains
        for order, chain_models in zip(self.orders_, self.chains_):
            X_ext = X.copy()
            chain_preds = [None] * n_outputs

            for idx, out_idx in enumerate(order):
                model = chain_models[idx]
                if self.output_types_[out_idx] == "classification":
                    pred_proba = model.predict_proba(X_ext)  # shape: (n_samples, n_classes)
                    chain_preds[out_idx] = pred_proba
                    # store class labels for later
                    if class_labels[out_idx] is None:
                        class_labels[out_idx] = np.arange(pred_proba.shape[1])
                    # feed argmax as feature for next in chain
#                    X_ext = np.hstack([X_ext, np.argmax(pred_proba, axis=1)[:, np.newaxis]])
                    X_ext = np.hstack([X_ext, pred_proba])

                else:
                    pred = model.predict(X_ext).ravel()  # regression
                    chain_preds[out_idx] = pred[:, np.newaxis]
                    X_ext = np.hstack([X_ext, pred[:, np.newaxis]])

            # append per output
            for i in range(n_outputs):
                all_preds[i].append(chain_preds[i])

        # Step 2: Aggregate across chains
        output_list = []
        for i, t in enumerate(self.output_types_):
            preds_stack = np.stack(all_preds[i], axis=0)  # shape: (n_chains, n_samples, ...)
            if t == "regression":
                # average regression predictions
                avg_preds = preds_stack.mean(axis=0)  # shape: (n_samples, 1)
                output_list.append(avg_preds)
            else:
                # average probabilities across chains
                avg_proba = preds_stack.mean(axis=0)  # shape: (n_samples, n_classes)
                output_list.append(avg_proba)

        # Step 3: Concatenate all outputs into a single 2D array
        final_array = np.hstack(output_list)  # shape: (n_samples, sum of regression + class columns)
        return final_array