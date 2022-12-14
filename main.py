import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import locale 
import babel
from streamlit.web import cli as stcli
from babel.numbers import format_currency
import time
import seaborn as sns

@st.cache
def load_dataset():
    raw_dataset = pd.read_excel('DATA RUMAH.xlsx', usecols=['HARGA','LB','LT','KT','KM'])
    dataset = raw_dataset.copy()

    dataset = dataset.rename(columns={'HARGA':'Harga','LB':'Luas Bangunan','LT':'Luas Tanah','KT':'Jumlah Kamar Tidur','KM':'Jumlah Kamar Mandi'})

    return dataset

def load_header():
    st.title("Linear Regression")
    st.subheader("Linear Regression - Prediksi Harga Rumah")
    st.caption("menggunakan scikit-learn, streamlit, pandas, numpy, matplotlib, seaborn")
    st.text("Rumus linear regression : ") 
    st.latex("y= ax + b")   

@st.cache
def train_model(dataset):
    reg_model = linear_model.LinearRegression()
    reg_model.fit(dataset[['Luas Bangunan','Luas Tanah','Jumlah Kamar Tidur','Jumlah Kamar Mandi']].values, dataset.Harga.values)

    return reg_model

def load_figure(dataset, option):
    fig = plt.figure()
    
    if option=="Luas Bangunan":
        sns.scatterplot(x= dataset['Luas Bangunan'], y=dataset['Harga'])
    elif option=='Luas Tanah':
        sns.scatterplot(x= dataset['Luas Tanah'], y= dataset['Harga'])
    elif option=='Jumlah Kamar Tidur':
        sns.scatterplot(x= dataset['Jumlah Kamar Tidur'], y= dataset['Harga'])
    else:
        sns.scatterplot(x= dataset['Jumlah Kamar Mandi'], y= dataset['Harga'])
    
    return fig


# @st.cache
def predict(model, lb, lt, km, kt):
    predict_value = model.predict([[lb, lt, km, kt]])
    predict_value = predict_value.reshape(1,-1)
    predict_price = round(predict_value[0][0],0)

    return predict_price

def main():
    load_header()
    dataset = load_dataset()
    st.dataframe(dataset.head(50), use_container_width=True)
    st.info("Dataset yang ditampilkan hanya sebagian!")
    reg_model = train_model(dataset)

    st.header('')
    st.header('Grafik Hubungan Antara Variable')
    option = st.selectbox(
    'Tampilkan Grafik hubungan antara Harga dengan ?',
    ('Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi'))
    st.write('Menampilkan Grafik hubungan antara Harga dengan', option)
    
    fig = load_figure(dataset,option)
    st.pyplot(fig)

    st.header('')
    st.subheader("Lakukan prediksi harga rumah")
    lb_input = st.number_input("Masukkan Luas Bangunan (m2):", min_value=50, max_value=2000, step=10)
    lt_input = st.number_input("Masukkan Luas Tanah (m2):", min_value=50, max_value=2000, step=10)
    km_input = st.number_input("Masukkan Jumlah Kamar Tidur :", min_value=1, max_value=20, step=1)
    kt_input = st.number_input("Masukkan Jumlah Kamar Mandi :", min_value=1, max_value=20, step=1)

    if st.button('Hitung Harga Prediksi'):
        predict_price = predict(reg_model, lb_input, lt_input, km_input, kt_input)

        with st.spinner('Wait for it...'):
            time.sleep(.3)
        if (int(predict_price) <= 0):
            st.warning("Harga rumah terlalu rendah Masukkan kategori yang benar", icon="⚠️")
        else:
            predict_price = format_currency(predict_price,'Rp. ', locale='en_US')
            st.success("Prediksi harga rumah : "+str(predict_price))
    else:
        st.caption("Tekan Tombol untuk memprediksi")
        
main()
