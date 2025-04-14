import numpy as np
import pandas as pd
from models.KNN import *
from models.SVM import *
from models.RandomF import *
from models.Neural import *
from models.SVMK import *

dataset = 'dtA'

train_path = f"./dataset/{dataset}/train_data.csv"
train_data = pd.read_csv(train_path)

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

test_path = f"./dataset/{dataset}/test_data.csv"
test_data = pd.read_csv(test_path)

x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# 建立模型
KNN_model    = KNNClassifier(k = 21, normalize = True, normDistance = 2)
SVM_model    = SVMClassifier(learning_rate=0.01, n_iters=1000)
RADF_model   = RandomForestClassifier()
Neural_model = NeuralNetClassifier()
SVMK_model   = SVMClassifierWithKernel()

# 訓練模型
KNN_model.fit(x_train,  y_train)
SVM_model.fit(x_train, y_train)
RADF_model.fit(x_train, y_train)
Neural_model.fit(x_train, y_train)
SVMK_model.fit(x_train, y_train)

# 測試模型
KNN_model.analysis(x_test, y_test)
SVM_model.analysis(x_test , y_test)
RADF_model.analysis(x_test , y_test)
Neural_model.analysis(x_test , y_test)
SVMK_model.analysis(x_test, y_test)