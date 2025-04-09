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

model = KNNClassifier(k = 21, normalize = True, normDistance = 2)
model1 = SVMClassifier(learning_rate=0.01, n_iters=1000)
model2 = RandomForestClassifier()
model3 = NeuralNetClassifier()
model4 = SVMClassifierWithKernel()

model.fit(x_train,  y_train)
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)
y_predict = model.predict(x_test)

model.score(y_test, y_predict)
model1.score(x_test , y_test)
model2.score(x_test , y_test)
model3.score(x_test , y_test)
model4.score(x_test, y_test)