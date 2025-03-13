import numpy as np
import pandas as pd
from models.KNN import *

path = "./dataset/dtA/test_data.csv"
df = pd.read_csv(path)

print(df)