import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

data_path = f"./dataset/dtA/train_data.csv"
data = pd.read_csv(data_path)

st.title('資料探勘 Data Mining')
st.write('---')
st.write(data)