import pytest
import os
import pandas as pd
import pickle
from train import load_and_preprocess_data
from main import predict_price, load_model

def test_data_loading():
    """Menguji apakah file dataset ada dan dapat dimuat dengan kolom yang benar."""
    filepath = 'DATA RUMAH.xlsx'
    assert os.path.exists(filepath), f"{filepath} tidak ditemukan"
    
    df = load_and_preprocess_data(filepath)
    expected_columns = ['Harga', 'Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi']
    for col in expected_columns:
        assert col in df.columns

def test_model_training_output():
    """Menguji apakah file model dihasilkan setelah proses training."""
    # Pastikan folder model ada
    if not os.path.exists('model'):
        os.makedirs('model')
        
    # File yang diharapkan ada (mengacu pada model utama)
    model_path = os.path.join('model', 'linear_regression.pkl')
    
    # Kita asumsikan train.py sudah dijalankan di CI sebelum test ini
    assert os.path.exists(model_path), "Model linear_regression.pkl tidak ditemukan di folder model/"

def test_prediction_logic():
    """Menguji fungsi prediksi dengan input dummy."""
    model_path = os.path.join('model', 'linear_regression.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Test prediksi (LB: 100, LT: 120, KT: 3, KM: 2)
    price = predict_price(model, 100, 120, 3, 2)
    
    assert isinstance(price, (int, float))
    assert price >= 0, "Harga prediksi tidak boleh negatif"

def test_load_model_helper():
    """Menguji helper load_model dari main.py."""
    model = load_model("Linear Regression")
    assert model is not None
