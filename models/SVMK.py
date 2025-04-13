import numpy as np
import pandas as pd

class SVMClassifierWithKernel:
    def __init__(self, kernel="rbf", C=3, gamma=0.2, learning_rate=0.005, n_iters=1000, normalize=True):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.lr = learning_rate
        self.n_iters = n_iters
        self.normalize = normalize
        self.alpha = None
        self.b = 0
        self.x_train = None
        self.y_train = None
        self.mean = None
        self.std = None

    def _kernel_function(self, x1, x2):
        if self.kernel == "linear":
            return np.dot(x1, x2.T)
        elif self.kernel == "rbf":
            sq_dist = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
            return np.exp(-self.gamma * sq_dist)
        elif self.kernel == "polynomial":
            if self.gamma <= 0:
                raise ValueError("Gamma must be positive for polynomial kernel")
            base = np.dot(x1, x2.T) + 2  # 調整偏移量為 2
            base = np.maximum(base, 0)  # 確保基數為非負數
            return base ** self.gamma
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.astype(float).to_numpy()
        y_train = np.where(y_train <= 0, -1, 1)

        if self.normalize:
            self.mean = x_train.mean(axis=0)
            self.std = x_train.std(axis=0)
            x_train = (x_train - self.mean) / self.std

        n_samples = x_train.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.x_train = x_train
        self.y_train = y_train

        K = self._kernel_function(x_train, x_train)

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                margin = y_train[idx] * (np.dot(self.alpha * y_train, K[:, idx]) + self.b)
                if margin < 1:
                    self.alpha[idx] += self.lr * (1 - margin)
                    self.b += self.lr * y_train[idx]
                else:
                    self.alpha[idx] -= self.lr * self.alpha[idx] * self.C

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.astype(float).to_numpy()

        if self.normalize:
            x_test = (x_test - self.mean) / self.std

        K = self._kernel_function(x_test, self.x_train)
        decision = np.dot(K, self.alpha * self.y_train) + self.b
        return np.where(decision >= 0, 1, 0)

    def score(self, x_test: pd.DataFrame, y_test: pd.Series) -> float:
        predictions = self.predict(x_test)
        print("Predictions:", predictions)  # 印出預測結果
        accuracy = np.mean(predictions == y_test)
        print(f"SVM with Kernel Classifier Score: {accuracy * 100:.2f}%")
        return accuracy