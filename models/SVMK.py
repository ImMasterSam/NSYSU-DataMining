import numpy as np
import pandas as pd

class SVMClassifierWithKernel:
    def __init__(self, learning_rate=0.001, n_iters=1000, kernel="rbf", degree=3, gamma=0.1):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.kernel = kernel
        self.degree = degree  # For polynomial kernel
        self.gamma = gamma    # For RBF kernel
        self.b = 0
        self.alpha = None     # 拉格朗日乘數
        self.x_train = None
        self.y_train = None

    def _kernel_function(self, x1, x2):
        if self.kernel == "linear":
            return np.dot(x1, x2.T)
        elif self.kernel == "polynomial":
            return (np.dot(x1, x2.T) + 1) ** self.degree
        elif self.kernel == "rbf":
            # Gaussian RBF kernel
            if x1.ndim == 1:
                x1 = x1[np.newaxis, :]
            if x2.ndim == 1:
                x2 = x2[np.newaxis, :]
            sq_dists = np.sum((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2, axis=2)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel")

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.astype(float).to_numpy()
        y_train = np.where(y_train <= 0, -1, 1)

        n_samples = x_train.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.x_train = x_train
        self.y_train = y_train

        K = self._kernel_function(x_train, x_train)

        for _ in range(self.n_iters):
            for i in range(n_samples):
                margin = np.sum(self.alpha * y_train * K[:, i]) + self.b
                if y_train[i] * margin < 1:
                    self.alpha[i] += self.lr
                    self.b += self.lr * y_train[i]

    def project(self, X):
        K = self._kernel_function(X, self.x_train)
        result = np.dot(K, self.alpha * self.y_train) + self.b
        return result

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.astype(float).to_numpy()
        return np.sign(self.project(x_test))

    def score(self, x_test: pd.DataFrame, y_test: pd.Series):
        y_true = np.where(y_test <= 0, -1, 1)
        predictions = self.predict(x_test)
        accuracy = np.mean(predictions == y_true)
        print(f"Score: {accuracy * 100:.2f} %")
        return accuracy
