import numpy as np
import pandas as pd

class SVMClassifier:

    def __init__(self,learning_rate=0.001, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.w = None #公式的權重
        self.b = None #偏差
        pass

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        x_train = x_train.astype(float)
        y_train = y_train.astype(int)

        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        n_samples , n_features = x_train.shape #sameple 為行 feature為列
        self.w = np.zeros(n_features) #因為svm最後要表達為f(x) = w_1*x1 + w_2*x2 + ...... + w_8*x8 + b 所以初始化向量的特徵
        self.b = 0

        y_t = np.where(y_train <= 0 , -1 ,1) #這個公式在將outcome傳換成-1 1因為svm是用-1 1做分類的 y_train <= 0 就是-1 else 0
        
        for _ in range(self.n_iters):#跑迴圈 不斷分類
            for idx , x_i in enumerate(x_train):
                condition = y_t[idx] * (np.dot(x_i,self.w) + self.b) >= 1 #梯度下降 分類對邊 >= 1 分類對邊但離中間太近0~1 分錯邊
                if condition:
                    continue
                else:
                    self.w -= self.lr * (-np.dot(x_i,y_t[idx]))
                    self.b -= self.lr * y_t[idx]
        pass

    def predict(self , x_test: pd.DataFrame):
        x_test = np.array(x_test)

        #分類 (x.w + b) 內積 + bias
        classification = np.sign(np.dot(x_test,self.w)+self.b)
        return classification

    def score(self,x_test:pd.DataFrame, y_test: pd.Series) -> float:
        predictions = self.predict(x_test)
        y_true = np.where(y_test <= 0 , -1 , 1)

        accuracy = np.mean(predictions == y_true)
        print(f"Score: {accuracy * 100: .2f} %")
        return accuracy
        