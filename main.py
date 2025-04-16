import numpy as np
import pandas as pd
import time
from models.KNN import *
from models.SVM import *
from models.RandomF import *
from models.Neural import *
from models.SVMK import *

datasets = ['dtA', 'dtB']
models = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Kernel SVM')
model_options = {'K Nearest Neighbors' : KNNClassifier(k = 21, normDistance = 2, normalize = True),
                 'Linear SVM' : SVMClassifier(learning_rate = 0.001, n_iters = 1000),
                 'Neural Network' : NeuralNetClassifier(n_hidden = 10, learning_rate = 0.001, n_iters = 1000, normalize = True),
                 'Random Forest' : RandomForestClassifier(n_estimators = 10, max_depth = 10, min_samples_split = 2, normalize = True),
                 'Kernel SVM' : SVMClassifierWithKernel(kernel = "rbf", C = 3, gamma = 0.2, n_iters = 1000, normalize = True),}

test_iters = 10

for dataset in datasets:

    print(f'\n ----------    {dataset}     --------------\n')

    train_path = f"./dataset/{dataset}/train_data.csv"
    train_data = pd.read_csv(train_path)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    test_path = f"./dataset/{dataset}/test_data.csv"
    test_data = pd.read_csv(test_path)

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    for model_name in models:

        total_measures = []

        for _ in range(test_iters):

            model = model_options[model_name]           # 建立模型
            start_time = time.time()
            model.fit(x_train,  y_train)                # 訓練模型
            end_time = time.time()
            fit_time = end_time - start_time

            measures = model.analysis(x_test, y_test)   # 測試模型
            measures['Fit time'] = fit_time

            total_measures.append(measures)

        measures_df = pd.DataFrame.from_records(total_measures)
        print(model_name, ': ')
        print(measures_df.mean(), '\n')