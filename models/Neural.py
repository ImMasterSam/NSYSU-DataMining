import numpy as np
import pandas as pd

class NeuralNetClassifier:
    def __init__(self, n_hidden=10, learning_rate=0.01, n_iters=1000):
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.n_iters = n_iters
        self.params = {}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        s = self._sigmoid(z)
        return s * (1 - s)

    def _initialize_weights(self, n_features):
        self.params['W1'] = np.random.randn(n_features, self.n_hidden) * 0.01
        self.params['b1'] = np.zeros((1, self.n_hidden))
        self.params['W2'] = np.random.randn(self.n_hidden, 1) * 0.01
        self.params['b2'] = np.zeros((1, 1))

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.astype(float).to_numpy()
        y_train = y_train.astype(int).to_numpy().reshape(-1, 1)

        n_samples, n_features = x_train.shape
        self._initialize_weights(n_features)

        for _ in range(self.n_iters):
            # Forward pass
            Z1 = np.dot(x_train, self.params['W1']) + self.params['b1']
            A1 = self._sigmoid(Z1)
            Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
            A2 = self._sigmoid(Z2)

            # Compute loss (binary cross-entropy)
            loss = -np.mean(y_train * np.log(A2 + 1e-8) + (1 - y_train) * np.log(1 - A2 + 1e-8))

            # Backward pass
            dZ2 = A2 - y_train
            dW2 = np.dot(A1.T, dZ2) / n_samples
            db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

            dA1 = np.dot(dZ2, self.params['W2'].T)
            dZ1 = dA1 * self._sigmoid_derivative(Z1)
            dW1 = np.dot(x_train.T, dZ1) / n_samples
            db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

            # Gradient descent update
            self.params['W1'] -= self.lr * dW1
            self.params['b1'] -= self.lr * db1
            self.params['W2'] -= self.lr * dW2
            self.params['b2'] -= self.lr * db2

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy().astype(float)
        Z1 = np.dot(x_test, self.params['W1']) + self.params['b1']
        A1 = self._sigmoid(Z1)
        Z2 = np.dot(A1, self.params['W2']) + self.params['b2']
        A2 = self._sigmoid(Z2)
        predictions = (A2 > 0.5).astype(int).flatten()
        return predictions

    def score(self, x_test: pd.DataFrame, y_test: pd.Series):
        y_pred = self.predict(x_test)
        accuracy = np.mean(y_pred == y_test.to_numpy().astype(int))
        print(f"Score: {accuracy * 100:.2f} %")
        return accuracy
