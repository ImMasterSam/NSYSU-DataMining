import numpy as np
import pandas as pd
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.tree = self._build_tree(data)

    def _build_tree(self, data, depth=0):
        X, y = data[:, :-1], data[:, -1]
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or num_labels == 1:
            leaf_value = self._majority_vote(y)
            return leaf_value

        best_feat, best_thresh = self._best_split(data)
        if best_feat is None:
            return self._majority_vote(y)

        left_data = data[data[:, best_feat] < best_thresh]
        right_data = data[data[:, best_feat] >= best_thresh]

        left_branch = self._build_tree(left_data, depth + 1)
        right_branch = self._build_tree(right_data, depth + 1)

        return (best_feat, best_thresh, left_branch, right_branch)

    def _best_split(self, data):
        X, y = data[:, :-1], data[:, -1]
        best_gini = 1
        best_feat, best_thresh = None, None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for thresh in thresholds:
                left = y[X[:, feature_idx] < thresh]
                right = y[X[:, feature_idx] >= thresh]
                gini = self._gini_index(left, right)
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feature_idx
                    best_thresh = thresh
        return best_feat, best_thresh

    def _gini_index(self, left, right):
        def gini(y):
            if len(y) == 0:
                return 0
            counts = np.bincount(y.astype(int))
            prob_sq = (counts / len(y)) ** 2
            return 1 - np.sum(prob_sq)
        total = len(left) + len(right)
        return (len(left)/total)*gini(left) + (len(right)/total)*gini(right)

    def _majority_vote(self, y):
        counts = Counter(y.astype(int))
        return counts.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature_idx, threshold, left_branch, right_branch = node
        if x[feature_idx] < threshold:
            return self._traverse_tree(x, left_branch)
        else:
            return self._traverse_tree(x, right_branch)

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        for _ in range(self.n_estimators):
            indices = np.random.choice(len(x_train), len(x_train), replace=True)
            x_sample = x_train[indices]
            y_sample = y_train[indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x_test: pd.DataFrame):
        x_test = x_test.to_numpy()
        tree_preds = np.array([tree.predict(x_test) for tree in self.trees])
        tree_preds = tree_preds.T
        final_preds = [Counter(row).most_common(1)[0][0] for row in tree_preds]
        return np.array(final_preds)

    def score(self, x_test: pd.DataFrame, y_test: pd.Series):
        predictions = self.predict(x_test)
        accuracy = np.mean(predictions == y_test.to_numpy())
        print(f"Random Forest Classifier Score: {accuracy * 100:.2f} %")
        return accuracy
