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

    def analysis(self, x_test: pd.DataFrame, y_test : pd.Series) -> dict:
        ''' 輸出各項測量值 '''

        measures = {}

        table = pt.PrettyTable()
        field_names = ['Classifier', 'Sensitivity', 'Specificity', 'Precission', 'Recall', 'F-Score', 'Accuracy']
        table.field_names = field_names

        y_predict = self.predict(x_test)
        p = np.sum(y_test == 1)
        n = np.sum(y_test == 0)
        all = p + n
        tp = np.sum((y_test == 1) & (y_predict == 1))
        fp = np.sum((y_test == 0) & (y_predict == 1))
        tn = np.sum((y_test == 0) & (y_predict == 0))
        fn = np.sum((y_test == 1) & (y_predict == 0))

        measures['Sensitivity'] = tp / p
        measures['Specificity'] = tn / n
        measures['Precission'] = tp / (tp + fp)
        measures['Recall'] = tn / (tn + fn)
        measures['F-Score'] = (2 * measures['Precission'] * measures['Recall']) / (measures['Precission'] + measures['Recall'])
        measures['Accuracy'] = (tp + tn) / all
        
        table.add_row([self.name] + [*(str(round(measures[s] * 100, 2)) + " %" for s in field_names[1:])])
        print(table)

        return measures
        

       