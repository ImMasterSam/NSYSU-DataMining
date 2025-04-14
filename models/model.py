import numpy as np
import pandas as pd
import prettytable as pt

class Classifier:

    def __init__(self, name: str, normalize: bool = True):
        self.name = name
        self.normalize = normalize

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    def predict(self, x_test: pd.DataFrame):
        pass

    def score(self, x_test: pd.DataFrame, y_test : pd.Series) -> float:
        ''' 輸出準確度 Accuracy '''
        y_predict = self.predict(x_test)
        accuracy = np.mean(y_predict == y_test.to_numpy().astype(int))
        print(f"{self.name} Score:  {accuracy * 100:.2f} %")
        return accuracy

    def analysis(self) -> dict:
        ''' 輸出各項測量值 '''
        pass