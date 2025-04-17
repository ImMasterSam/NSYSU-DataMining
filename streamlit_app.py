import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import time

from models.KNN import *
from models.SVM import *
from models.Neural import *
from models.RandomF import *
from models.SVMK import *

kValue = 15
NormValue = 2
Normalize = True
learning_rate = 0.001
n_iters = 1000
n_estimators = 10
max_depth = 10
min_samples_split = 2
kernal = 'rbf'
c = 1.0
gamma = 0.2

test_iters = 1
test_flag = False

A_Score = 0
A_Analysis = {}
A_test_measures = []

B_Score = 0
B_Analysis = {}
B_test_measures = []

dtA_train_path = f'./dataset/dtA/train_data.csv'
dtA_test_path = f'./dataset/dtA/test_data.csv'
dtB_train_path = f'./dataset/dtB/train_data.csv'
dtB_test_path = f'./dataset/dtB/test_data.csv'

def run_model(model_option):

    global kValue, NormValue, Normalize, learning_rate, n_iters, n_estimators, max_depth, min_samples_split, kernal, c, gamma
    global A_Score, B_Score, A_Analysis, B_Analysis


    model_options = {'K Nearest Neighbors' : KNNClassifier(k = kValue, normalize = Normalize, normDistance = NormValue),
                     'Linear SVM' : SVMClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Neural Network' : NeuralNetClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Random Forest' : RandomForestClassifier(n_estimators= n_estimators , max_depth= max_depth , min_samples_split= min_samples_split , normalize= Normalize),
                     'Kernel SVM' : SVMClassifierWithKernel(kernel= kernal , C=c , gamma= gamma , n_iters= n_iters , normalize= Normalize),}

    # Dataset A
    dataset = 'dtA'

    train_path = f"./dataset/{dataset}/train_data.csv"
    train_data = pd.read_csv(train_path)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    test_path = f"./dataset/{dataset}/test_data.csv"
    test_data = pd.read_csv(test_path)

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model = model_options[model_option]
    start_time = time.time()
    model.fit(x_train,  y_train)
    end_time = time.time()
    fit_time = end_time - start_time
    A_Analysis = model.analysis(x_test, y_test)
    A_Analysis['Fit time'] = fit_time
    A_Score = A_Analysis['Accuracy'] * 100

    # Dataset B
    dataset = 'dtB'

    train_path = f"./dataset/{dataset}/train_data.csv"
    train_data = pd.read_csv(train_path)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    test_path = f"./dataset/{dataset}/test_data.csv"
    test_data = pd.read_csv(test_path)

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model = model_options[model_option]
    start_time = time.time()
    model.fit(x_train,  y_train)
    end_time = time.time()
    fit_time = end_time - start_time
    B_Analysis = model.analysis(x_test, y_test)
    B_Analysis['Fit time'] = fit_time
    B_Score = B_Analysis['Accuracy'] * 100

def test_models():

    global kValue, NormValue, Normalize, learning_rate, n_iters, n_estimators, max_depth, min_samples_split, kernal, c, gamma
    global A_Score, B_Score, A_Analysis, B_Analysis, A_test_measures, B_test_measures, test_iters

    models = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Kernel SVM')
    model_options = {'K Nearest Neighbors' : KNNClassifier(k = kValue, normalize = Normalize, normDistance = NormValue),
                     'Linear SVM' : SVMClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Neural Network' : NeuralNetClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Random Forest' : RandomForestClassifier(n_estimators= n_estimators , max_depth= max_depth , min_samples_split= min_samples_split , normalize= Normalize),
                     'Kernel SVM' : SVMClassifierWithKernel(kernel= kernal , C=c , gamma= gamma , n_iters= n_iters , normalize= Normalize),}
    
    # Dataset A
    dataset = 'dtA'
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
        A_test_measures.append({model_name : measures_df.mean().to_dict()})
    
    # Dataset B
    dataset = 'dtB'
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
        B_test_measures.append({model_name : measures_df.mean().to_dict()})



def sideBar_config(model: str):

    global kValue, NormValue, Normalize, learning_rate, n_iters, n_estimators, max_depth, min_samples_split, kernal, c, gamma, test_iters, test_flag
    st.sidebar.write('## 參數設定')

    match model:
        case 'K Nearest Neighbors':
            kValue = st.sidebar.slider(label = 'K 鄰近值',
                              min_value = 3,
                              max_value = 99,
                              step = 2,
                              value = 15,
                              help = '選定 k 個最鄰近的鄰居來決定類別 (固定為奇數)')
            NormValue = st.sidebar.slider(label = 'Norm 值',
                              min_value = 1,
                              max_value = 10,
                              value = 2,
                              help = r'''
                                      ### 範數距離 p - Norm Distance
                                      $$ d = ( \sum_{i=1}^n {|x_i - y_i|}^p )^{\frac{1}{p}} $$
                                      - $p=1$ : 曼哈頓距離
                                      - $p=2$ : 歐幾里德距離
                                      ''')
            Normalize = st.sidebar.checkbox(label = '標準化資料',
                                            value = True,
                                            help = '將資料標準化後再進行分類')
        case 'Linear SVM':
            learning_rate = st.sidebar.number_input(label = '學習率',
                              min_value = 0.,
                              max_value = 1.,
                              step = 0.001,
                              value = 0.001,
                              format = '%.3f',
                              help = '控制訓練速度，太大容易訓練緩慢，太小容易無法收斂')
            n_iters = st.sidebar.slider(label = 'N 次迭代值',
                              min_value = 100,
                              max_value = 2000,
                              value = 1000,
                              step = 100,
                              help = '模型學習的次數')
            Normalize = st.sidebar.checkbox(label = '標準化資料',
                                            value = True,
                                            help = '將資料標準化後再進行分類')
        case 'Neural Network':
            learning_rate = st.sidebar.number_input(label = '學習率',
                              min_value = 0.,
                              max_value = 1.,
                              step = 0.001,
                              value = 0.001,
                              format = '%.3f',
                              help = '控制訓練速度，太大容易訓練緩慢，太小容易無法收斂')
            n_iters = st.sidebar.slider(label = 'N 次迭代值',
                              min_value = 100,
                              max_value = 10000,
                              value = 1000,
                              step = 100,
                              help = '模型學習的次數')
            Normalize = st.sidebar.checkbox(label = '標準化資料',
                                            value = True,
                                            help = '將資料標準化後再進行分類')
        case 'Random Forest':
            n_estimators = st.sidebar.slider(label = '決策樹的數量',
                                min_value = 1,
                                max_value = 100,
                                value = 10,
                                step = 1,
                                help = '表隨機森林中樹的數量')
            max_depth = st.sidebar.slider(label = '樹的最大深度',
                                min_value = 1,
                                max_value = 100,
                                value = 10,
                                step = 1,
                                help = '表決策樹的最大深度')
            min_samples_split = st.sidebar.slider(label = '最小分割樣本數',
                                min_value = 1,
                                max_value = 50,
                                value = 2,
                                step = 1,
                                help = '表什麼時候該停止繼續分裂節點')
            Normalize = st.sidebar.checkbox(label = '標準化資料',
                                            value = True,
                                            help = '將資料標準化後再進行分類')
        case 'Kernel SVM':
            kernal = st.sidebar.selectbox(label = '核函數',
                                options = ('linear', 'poly', 'rbf', 'sigmoid'),
                                index = 2,
                                help = '表決定資料的邊界')
            c = st.sidebar.number_input(label = '懲罰參數 C',
                                min_value = 0.01,
                                max_value = 100.0,
                                value = 1.0,
                                step = 0.1,
                                format = '%.2f',
                                help = '表對錯誤分類的懲罰程度')
            gamma = st.sidebar.number_input(label = 'Gamma 值',
                                min_value = 0.01,
                                max_value = 100.0,
                                value = 0.2,
                                step = 0.01,
                                format = '%.2f',
                                help = '表決定資料的邊界')
            n_iters = st.sidebar.slider(label = 'N 次迭代值',
                              min_value = 100,
                              max_value = 10000,
                              value = 1000,
                              step = 100,
                              help = '模型學習的次數')
            Normalize = st.sidebar.checkbox(label = '標準化資料',
                                            value = True,
                                            help = '將資料標準化後再進行分類')
        case '綜合測試':
            test_iters = st.sidebar.slider(label = '重複試驗次數',
                                           min_value = 1,
                                           max_value = 15,
                                           step = 1,
                                           value = 1,
                                           help = '每個模型在每個資料集的重複試驗次數')
            st.sidebar.warning('花費較多時間，點擊按鈕後請稍後')

    if st.sidebar.button(label = '訓練'):
        if model == '綜合測試':
            test_flag = True
            test_models()
        else:
            run_model(model_options)

# 頁面設定
st.set_page_config(page_title = 'NSYSU - 資料探勘')
st.title('資料探勘 Data Mining')

# 側邊欄設定
model_options = st.sidebar.selectbox(label = '請選擇分類模型: ',
                                     options = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Kernel SVM', '綜合測試'))
sideBar_config(model_options)

# 分頁設定
models_tab, data_tab = st.tabs(['分析📈', '原始資料📃'])

# 資料讀取
dtA_train_data = pd.read_csv(dtA_train_path)
dtA_test_data = pd.read_csv(dtA_test_path)
dtB_train_data = pd.read_csv(dtB_train_path)
dtB_test_data = pd.read_csv(dtB_test_path)

# 資料分析分頁
with models_tab:

    st.subheader('分布圖')

    dataset = st.selectbox(label = '資料集: ',
                           options = ('資料集 A', '資料集 B'))
    
    if dataset == '資料集 A':
        train_data = dtA_train_data
    else:
        train_data = dtB_train_data

    # 圖表調整欄
    col1, col2 = st.columns(2)
    with col1:
        x_select = st.selectbox(label = 'X 軸',
                                options = train_data.columns.values[:-1],
                                index = 2)
    with col2:
        y_select = st.selectbox(label = 'Y 軸',
                                options = train_data.columns.values[:-1],
                                index = 3)

    # 統整圖表 
    st.scatter_chart(train_data, x = x_select, y = y_select, color = 'Outcome')

    # fig, ax = plt.subplots()
    # ax.scatter(data[x_select], data[y_select], c = data['Outcome'])
    # st.pyplot(fig)

    # 顯示訓練結果
    st.subheader('模型訓練結果')
    if test_flag:
   
        def flatten_results(results_list):
            flattened = {}
            for entry in results_list:
                for model_name, metrics in entry.items():
                    flattened[model_name] = metrics
            return flattened

        A_test_result = flatten_results(A_test_measures)
        A_test_df = pd.DataFrame(A_test_result).T  # 模型為 index

        st.write('### 資料集 A 測試結果')

        st.write('##### 準確度 Accuracy')
        st.bar_chart(A_test_df[['Accuracy']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### F - Score')
        st.bar_chart(A_test_df[['F-Score']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 召回率 Recall')
        st.bar_chart(A_test_df[['Recall']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 精準率 Precision')
        st.bar_chart(A_test_df[['Precision']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 特異度 Specificity')
        st.bar_chart(A_test_df[['Specificity']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 靈敏度 Sensitivity')
        st.bar_chart(A_test_df[['Sensitivity']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 訓練時間 Fit time')
        st.bar_chart(A_test_df[['Fit time']], x_label = '執行時間 (秒)', horizontal = True)
        st.write('##### 預測時間 Predict time')
        st.bar_chart(A_test_df[['Predict time']], x_label = '結果 (%)', horizontal = True)

        st.write('##### 綜合比較')
        st.bar_chart(A_test_df.drop(axis = 1, labels = ['Predict time', 'Fit time']), stack = False, horizontal = False)

        st.write('---')

        st.write('### 資料集 B 測試結果')

        B_test_result = flatten_results(B_test_measures)
        B_test_df = pd.DataFrame(B_test_result).T  # 模型為 index

        st.write('##### 準確度 Accuracy')
        st.bar_chart(B_test_df[['Accuracy']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### F - Score')
        st.bar_chart(B_test_df[['F-Score']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 召回率 Recall')
        st.bar_chart(B_test_df[['Recall']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 精準率 Precision')
        st.bar_chart(B_test_df[['Precision']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 特異度 Specificity')
        st.bar_chart(B_test_df[['Specificity']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 靈敏度 Sensitivity')
        st.bar_chart(B_test_df[['Sensitivity']] * 100, x_label = '結果 (%)', horizontal = True)
        st.write('##### 訓練時間 Fit time')
        st.bar_chart(B_test_df[['Fit time']], x_label = '執行時間 (秒)', horizontal = True)
        st.write('##### 預測時間 Predict time')
        st.bar_chart(B_test_df[['Predict time']], x_label = '結果 (%)', horizontal = True)

        st.write('##### 綜合比較')
        st.bar_chart(B_test_df.drop(axis = 1, labels = ['Predict time', 'Fit time']), stack = False, horizontal = False)


    elif A_Score:

        measures = pd.DataFrame.from_records({'dtA': A_Analysis, 'dtB': B_Analysis})
        st.write(f'### 模型 : {model_options} Classifier')

        # Dataset A
        st.write(f'##### 資料集 A')
        st.success(f'正確率 : {A_Score:.2f} %')
        st.table(A_Analysis)

        A_col1, A_col2 = st.columns(2)
        with A_col1:
            st.info(f'訓練時間 : {measures['dtA']['Fit time']: .3f} 秒')
        with A_col2:
            st.info(f'預測時間 : {measures['dtA']['Predict time']: .3f} 秒')

        st.write('---')

        # Dataset B
        st.write(f'##### 資料集 B')
        st.success(f'正確率 : {B_Score:.2f} %')
        st.table(B_Analysis)

        B_col1, B_col2 = st.columns(2)
        with B_col1:
            st.info(f'訓練時間 : {measures['dtB']['Fit time']: .3f} 秒')
        with B_col2:
            st.info(f'預測時間 : {measures['dtB']['Predict time']: .3f} 秒')

        st.write('---')

        st.write(f'##### 比較結果')

        analysis = measures.drop(['Predict time', 'Fit time']) * 100
        st.bar_chart(analysis, x_label = '結果 (%)', stack = False, horizontal = True)

    else:
        st.info(f'請先在左側側邊欄訓練後觀看結果')

# 原始數據分頁
with data_tab:

    # 顯示資料
    st.write('### 資料集 A')
    st.write('##### 訓練集')
    st.write(dtA_train_data)
    st.write('##### 測試集')
    st.write(dtA_test_data)
    st.write('---')
    st.write('### 資料集 B')
    st.write('##### 訓練集')
    st.write(dtB_train_data)
    st.write('##### 測試集')
    st.write(dtB_test_data)