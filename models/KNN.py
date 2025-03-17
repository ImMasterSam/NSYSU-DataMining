import numpy as np
import pandas as pd

class KNNClassifier:

    def __init__(self, k: int, normalize: bool = True):
        self.k = k
        self.normal = normalize
        pass

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.trainDatas = x_train
        self.trainLabels = y_train

        if self.normal:
            self.mean = self.trainDatas.mean()
            self.std = self.trainDatas.std()
            self.trainDatas = (self.trainDatas - self.mean) / self.std

    def predict(self, x_test: pd.DataFrame):

        res = []

        if self.normal:
            x_test = (x_test - self.mean) / self.std
        
        for i in range(x_test.shape[0]):                      # 測試資料行數
            te_num = x_test.loc[i].to_numpy()                 # 轉成numpy
            distance_between = []
            for j in range(self.trainDatas.shape[0]):               # 訓練資料行數
                df_num = self.trainDatas.loc[j].to_numpy()          # 轉乘numpy
 
                distance = 0                               	# 初始化距離
 
                for k in range(x_test.shape[1]):            # 到outcome前
                    distance += (te_num[k] - df_num[k])**2  # 算距離

                distance_between.append([distance**(1/2), self.trainLabels.values[j]])  # append開方距離以及outcome
            distance_between.sort()

            # Moore Majority Voting Algorithm
            cnt = 0
            major = distance_between[0][1]
            for i in range(1, k):
                if distance_between[i][1] == major:
                    cnt += 1
                else:
                    cnt -= 1
                
                if cnt <= 0:
                    major = distance_between[i][1]
            
            res.append(major)

        return res


    def score(self, y_test: pd.DataFrame, y_predict : list) -> float:
        
        total = 0
        right = 0

        for i in range(len(y_predict)):
            if y_test[i] == y_predict[i]:
                total += 1
                right += 1
            else:
                total += 1

        print((right/total)*100,"%")