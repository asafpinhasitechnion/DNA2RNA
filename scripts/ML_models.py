from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class SklearnModelWrapper(BaseEstimator):
    def __init__(self, model, task='binary'):
        self.model = model
        self.task = task
        self.feature_importances_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        if hasattr(self.model, "coef_"):
            self.feature_importances_ = self.model.coef_.flatten()
        elif hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_

    def predict(self, X):
        if self.task == 'binary':
            return self.model.predict(X)
        elif self.task == 'multiclass':
            return self.model.predict(X)
        elif self.task == 'regression':
            return self.model.predict(X)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X)
            return probas if self.task == 'multiclass' else probas[:, 1]
        elif self.task == 'regression':
            raise NotImplementedError("predict_proba is not applicable to regression.")
        else:
            raise NotImplementedError("This model does not support probability predictions.")


class PytorchModelWrapper(BaseEstimator):
    def __init__(self, input_dim, layers, task='binary', output_dim=None,
                 learning_rate=0.001, batch_size=32, epochs=50, device=None, verbose=0):
        self.input_dim = input_dim
        self.layers = layers
        self.task = task
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()
        self.criterion = self._get_criterion()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.verbose = verbose

    def _build_model(self):
        model = nn.Sequential()
        prev_dim = self.input_dim

        for i, hidden_dim in enumerate(self.layers):
            model.add_module(f"Linear_{i+1}", nn.Linear(prev_dim, hidden_dim))
            model.add_module(f"ReLU_{i+1}", nn.ReLU())
            model.add_module(f"Dropout_{i+1}", nn.Dropout(0.2))
            prev_dim = hidden_dim

        if self.task == 'binary':
            model.add_module("Output", nn.Linear(prev_dim, 1))
            model.add_module("Sigmoid", nn.Sigmoid())
        elif self.task == 'multiclass':
            if self.output_dim is None:
                raise ValueError("output_dim must be specified for multiclass task.")
            model.add_module("Output", nn.Linear(prev_dim, self.output_dim))
            model.add_module("Softmax", nn.Softmax(dim=1))
        elif self.task == 'regression':
            model.add_module("Output", nn.Linear(prev_dim, 1))
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return model.to(self.device)

    def _get_criterion(self):
        if self.task == 'binary':
            return nn.BCELoss()
        elif self.task == 'multiclass':
            return nn.CrossEntropyLoss()
        elif self.task == 'regression':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.task == 'multiclass':
            y = torch.tensor(y, dtype=torch.long).to(self.device)
        else:
            y = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs}")
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch).squeeze()
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy()

        if self.task == 'binary':
            return (y_pred.flatten() > 0.5).astype(int)
        elif self.task == 'multiclass':
            return y_pred.argmax(axis=1)
        elif self.task == 'regression':
            return y_pred.flatten()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def predict_proba(self, X):
        if self.task == 'regression':
            raise NotImplementedError("predict_proba not available for regression.")

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy()

        return y_pred
