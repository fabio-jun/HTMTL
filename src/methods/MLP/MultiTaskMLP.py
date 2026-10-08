import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class MultiTaskMLP:
    def __init__(self,
                    output_types, 
                    hidden_units=[128, 64],
                    
                    # ---- added parameters ----
                    activation="relu",
                    optimizer="adam",
                    learning_rate=0.001,
                    batch_size=16,
                    num_epochs=100,
                    dropout_prob=0.0,
                    use_batchnorm=False,
                    require_gpu=False,
                    gpu_memory_limit_mb=14000,
                    fit_verbose=0):

        self.hidden_units = hidden_units
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob
        self.use_batchnorm = use_batchnorm
        self.require_gpu = require_gpu
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.fit_verbose = fit_verbose

        self.available_gpus = tf.config.list_physical_devices("GPU")
        if self.available_gpus:
            try:
                tf.config.set_logical_device_configuration(
                    self.available_gpus[0],
                    [
                        tf.config.LogicalDeviceConfiguration(
                            memory_limit=self.gpu_memory_limit_mb
                        )
                    ],
                )
            except (RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "Failed to limit GPU memory to "
                    f"{self.gpu_memory_limit_mb} MiB before TensorFlow initialization"
                ) from exc
        self.selected_device = "/GPU:0" if self.available_gpus else "/CPU:0"
        if not self.available_gpus:
            print("MultiTaskMLP CPU fallback enabled: TensorFlow detected no GPU")
        print(f"MultiTaskMLP device: {self.selected_device}")

        self.model = None

        self.output_types_ = output_types
        self.scalers = {}
        self.encoders = {}

    # ---------------- TYPE INFERENCE ----------------
    def _infer_type(self, col, y):
        if "binary" in col.lower():
            return "binary"
        elif "regression" in col.lower():
            return "regression"
        elif y[col].dtype.kind in "ifu":  # numeric
            return "regression"
        else:
            return "classification"

    # ---------------- MODEL ----------------
    def _build_model(self, input_dim, output_types, y):
        inputs = layers.Input(shape=(input_dim,))
        
        x = inputs
        for u in self.hidden_units:
            x = layers.Dense(u, use_bias=not self.use_batchnorm)(x)

            # ---- added batchnorm ----
            if self.use_batchnorm:
                x = layers.BatchNormalization()(x)

            # ---- activation replaced ----
            x = layers.Activation(self.activation)(x)

            # ---- added dropout ----
            if self.dropout_prob > 0:
                x = layers.Dropout(self.dropout_prob)(x)

        outputs = []
        losses = {}
        metrics = {}

        for col, typ in zip(y.columns.astype(str), output_types):

            if typ == "regression":
                out = layers.Dense(1, name=col)(x)
                losses[col] = "mse"
                metrics[col] = ["mae"]

            elif typ == "binary":
                out = layers.Dense(1, activation="sigmoid", name=col)(x)
                losses[col] = "binary_crossentropy"
                metrics[col] = ["accuracy"]

            elif typ == "classification":
                n_classes = y[col].nunique()
                out = layers.Dense(n_classes, activation="softmax", name=col)(x)
                losses[col] = "sparse_categorical_crossentropy"
                metrics[col] = ["accuracy"]

            outputs.append(out)

        model = Model(inputs, outputs)

        # ---- added optimizer with learning rate ----
        if self.optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
        )

        return model

    # ---------------- FIT ----------------
    def fit(self, X, y):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y)

        self.y_columns = y.columns

        # -------- transform targets --------
        y_dict = {}

        for col, typ in zip(y.columns, self.output_types_):

            if typ == "binary":
                y_dict[str(col)] = y[col].values.astype(float)

            elif typ == "regression":
                scaler = StandardScaler()
                y_dict[str(col)] = scaler.fit_transform(y[[col]])
                self.scalers[str(col)] = scaler

            elif typ == "classification":
                y_dict[str(col)] = y[col].values

        # build model once
        if self.model is None:
            with tf.device(self.selected_device):
                self.model = self._build_model(X.shape[1], self.output_types_, y)

        with tf.device(self.selected_device):
            self.model.fit(
                X,
                y_dict,
                epochs=self.num_epochs,
                batch_size=self.batch_size,
                verbose=self.fit_verbose
            )

        return self

    # ---------------- PREDICT PROBA ----------------
    def predict_proba(self, X):
        with tf.device(self.selected_device):
            preds = self.model.predict(X, verbose=0)

        if not isinstance(preds, list):
            preds = [preds]

        result = {}

        for col, typ, pred in zip(self.y_columns.astype(str), self.output_types_, preds):

            if typ == "regression":
                scaler = self.scalers[col]
                result[col] = scaler.inverse_transform(pred).squeeze()

            elif typ == "binary":
                result[col] = pred.squeeze()

            elif typ == "classification":
                result[col] = pred  # full probability distribution

        return result
