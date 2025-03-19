import numpy as np
import pandas as pd
from models.KNN import *

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
model.fit(x_train,  y_train)
y_predict = model.predict(x_test)

model.score(y_test, y_predict)
