import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from models.KNN import *

kValue = 15
NormValue = 2
Normalize = True
Score = 0

def run_model():

    global kValue, NormValue, Normalize, Score

    dataset = 'dtA'

    train_path = f"./dataset/{dataset}/train_data.csv"
    train_data = pd.read_csv(train_path)

    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    test_path = f"./dataset/{dataset}/test_data.csv"
    test_data = pd.read_csv(test_path)

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    model = KNNClassifier(k = kValue, normalize = Normalize, normDistance = NormValue)
    model.fit(x_train,  y_train)
    y_predict = model.predict(x_test)

    Score = model.score(y_test, y_predict)


def sideBar_config(model: str):

    global kValue, NormValue, Normalize

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
        case 'test':
            st.sidebar.write('測試用 123123')

    if st.sidebar.button(label = '訓練'):
        run_model()

# 頁面設定
st.set_page_config(page_title = 'NSYSU - 資料探勘')
st.title('資料探勘 Data Mining')
st.write('---')

# 側邊欄設定
model_options = st.sidebar.selectbox(label = '請選擇分類模型: ',
                                     options = ("K Nearest Neighbors", "test"))
sideBar_config(model_options)


# 顯示資料
data_path = f'./dataset/dtA/train_data.csv'
data = pd.read_csv(data_path)
st.write('### 原始資料 : ')
st.write(data)

# 顯示訓練結果
if Score:
    st.success(f'正確率 : {Score:.2f} %')
else:
    st.warning(f'請先在左側側邊欄訓練後觀看結果')