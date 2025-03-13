import numpy as np
import pandas as pd

class KNNClassifier:

    def __init__(self, k: int):
        self.k = k
        pass

    def fit(self,df: pd.DataFrame):
        self.df = df
        pass

    def predict(self,test: pd.DataFrame):

        res = []
        
        for i in range(test.shape[0]):                      # 測試資料行數
            te_num = test.loc[i].to_numpy()                 # 轉成numpy
            distance_between = []
            for j in range(self.df.shape[0]):               # 訓練資料行數
                df_num = self.df.loc[j].to_numpy()          # 轉乘numpy
 
                distance = 0                               	# 初始化距離
 
                for k in range(test.shape[1]-1):            # 到outcome前
                    distance += (te_num[k] - df_num[k])**2  # 算距離

                distance_between.append([distance**(1/2),self.df['Outcome'].values[j]])  # append開方距離以及outcome
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


    def score(self,test: pd.DataFrame,outcome : list) -> float:
        
        total = 0
        right = 0

        for i in range(len(outcome)):
            if test["Outcome"].values[i] == outcome[i]:
                total += 1
                right += 1
            else:
                total += 1

        print((right/total)*100,"%")