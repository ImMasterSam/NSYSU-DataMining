import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

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
Score = 0
Analysis = {}

dtA_train_path = f'./dataset/dtA/train_data.csv'
dtA_test_path = f'./dataset/dtA/test_data.csv'
dtB_train_path = f'./dataset/dtB/train_data.csv'
dtB_test_path = f'./dataset/dtB/test_data.csv'

def run_model(model_option):

    global kValue, NormValue, Normalize, learning_rate, n_iters, Score, Analysis

    dataset = 'dtA'

    train_path = f"./dataset/{dataset}/train_data.csv"
    train_data = pd.read_csv(train_path)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    test_path = f"./dataset/{dataset}/test_data.csv"
    test_data = pd.read_csv(test_path)

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model_options = {'K Nearest Neighbors' : KNNClassifier(k = kValue, normalize = Normalize, normDistance = NormValue),
                     'Linear SVM' : SVMClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Neural Network' : NeuralNetClassifier(learning_rate = learning_rate, n_iters = n_iters),
                     'Random Forest' : RandomForestClassifier(),
                     'Kernel SVM' : SVMClassifierWithKernel(),}

    model = model_options[model_option]
    model.fit(x_train,  y_train)
    Analysis = model.analysis(x_test, y_test)
    Score = Analysis['Accuracy'] * 100


def sideBar_config(model: str):

    global kValue, NormValue, Normalize, learning_rate, n_iters

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
                              max_value = 10000,
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
            st.sidebar.write('此模型無自訂參數')
        case 'Kernel SVM':
            st.sidebar.write('此模型無自訂參數')
        case '綜合測試':
            st.sidebar.write('花費較多時間，點擊訓練按鈕後請稍後')
        case 'test':
            st.sidebar.write('測試用 123123')

    if st.sidebar.button(label = '訓練'):
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

    # 圖表調整欄
    col1, col2 = st.columns(2)
    with col1:
        x_select = st.selectbox(label = 'X 軸',
                                options = dtA_train_data.columns.values[:-1],
                                index = 2)
    with col2:
        y_select = st.selectbox(label = 'Y 軸',
                                options = dtA_train_data.columns.values[:-1],
                                index = 3)

    # 統整圖表 
    st.scatter_chart(dtA_train_data, x = x_select, y = y_select, color = 'Outcome')

    # fig, ax = plt.subplots()
    # ax.scatter(data[x_select], data[y_select], c = data['Outcome'])
    # st.pyplot(fig)

    # 顯示訓練結果
    st.subheader('模型訓練結果')
    if Score:
        st.write(f'#### 模型 : {model_options} Classifier')
        st.success(f'正確率 : {Score:.2f} %')
        st.table(Analysis)
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