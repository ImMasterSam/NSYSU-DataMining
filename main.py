import numpy as np
import pandas as pd
from models.KNN import *

train_path = "./dataset/dtA/test_data.csv"
train_data = pd.read_csv(train_path)
test_path = "./dataset/dtB/test_data.csv"
test_data = pd.read_csv(test_path)

model = KNNClassifier(k = 5)
model.fit(train_data)
o = model.predict(test_data)

model.score(test_data,o)
