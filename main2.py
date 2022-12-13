import streamlit as st
import pandas as pd
import numpy as numpy
import matplotlib.pyplot as plt
from sklearn import linear_model
import locale 
import time
import sys
import babel
from streamlit.web import cli as stcli
from babel.numbers import format_currency



def main():
    
    data_frame = pd.read_excel("DATA RUMAH.xlsx", usecols=['HARGA','LB','LT','KM','KT'])
    
    st.title("Linear Regression")
    st.subheader("Contoh Linear Regression Prediksi Harga Rumah")
    st.caption("Dengan menggunakan **machine learning** ")
    st.caption("Dengan rumus") 
    st.latex("y= ax + b")
    
    @st.cache
    st.caption("Kita memiliki data sebagai berikut : ")
    st.dataframe(data_frame.head(10))
    st.text("Data Rumah hanya ditampilkan sebagian untuk melihat keseluruhan data kamu bisa mencari datasetnya di kaggle")
    
    st.caption("Diperoleh line chart")
    fig1 = plt.figure() 
    plt.scatter(data_frame['LB'], data_frame['HARGA'])
    st.write(fig1)

    reg_model = linear_model.LinearRegression()
    reg_model.fit(data_frame[['LB','LT','KM','KT']].values,data_frame.HARGA.values)

    lb_user = st.number_input("Masukkan Luas Bangunan :", min_value=10, max_value=1000, step=10)
    lt_user = st.number_input("Masukkan Luas Tanah :", min_value=10, max_value=1000, step=10)
    km_user = st.number_input("Masukkan Jumlah Kamar Tidur :", min_value=1, max_value=10, step=1)
    kt_user = st.number_input("Masukkan Jumlah Kamar Mandi :", min_value=1, max_value=10, step=1)

    predict_value = reg_model.predict([[lb_user,lt_user,km_user,kt_user]])
    predict_value = predict_value.reshape(1,-1)
    predict_value = round(predict_value[0][0],0)
    # predict_value = format_currency(100, 'Rp', locale='en_US'))
    
    new = format_currency(predict_value,'Rp. ', locale='en_US')

    if st.button('Hitung Harga Prediksi'):
        with st.spinner('Wait for it...'):
            time.sleep(.3)

        if (int(predict_value) <= 0):
            st.warning("Harga rumah terlalu rendah Masukkan kategori yang benar", icon="⚠️")
        else:
            st.success("Prediksi harga rumah : "+str(new))
    else:
        st.caption("Tekan Tombol untuk memprediksi")
        
main()
