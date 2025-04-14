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
    Score = model.score(x_test, y_test) * 100


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

# 側邊欄設定
model_options = st.sidebar.selectbox(label = '請選擇分類模型: ',
                                     options = ('K Nearest Neighbors', 'Linear SVM', 'Neural Network', 'Random Forest', 'Kernel SVM', 'test'))
sideBar_config(model_options)

# 分頁設定
models_tab, data_tab = st.tabs(['分析📈', '原始資料📃'])

# 資料讀取
data_path = f'./dataset/dtA/train_data.csv'
data = pd.read_csv(data_path)


# 資料分析分頁
with models_tab:

    st.subheader('分布圖')

    # 圖表調整欄
    col1, col2 = st.columns(2)
    with col1:
        x_select = st.selectbox(label = 'X 軸',
                                options = data.columns.values[:-1],
                                index = 2)
    with col2:
        y_select = st.selectbox(label = 'Y 軸',
                                options = data.columns.values[:-1],
                                index = 3)

    # 統整圖表 
    st.scatter_chart(data, x = x_select, y = y_select, color = 'Outcome')

    # fig, ax = plt.subplots()
    # ax.scatter(data[x_select], data[y_select], c = data['Outcome'])
    # st.pyplot(fig)

    # 顯示訓練結果
    st.subheader('模型訓練')
    if Score:
        st.success(f'正確率 : {Score:.2f} %')
    else:
        st.info(f'請先在左側側邊欄訓練後觀看結果')

# 原始數據分頁
with data_tab:

    # 顯示資料
    st.write('### 原始資料 : ')
    st.write(data)