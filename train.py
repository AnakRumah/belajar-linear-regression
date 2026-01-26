import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(filepath):
    """
    Loads the dataset and performs preprocessing (renaming columns).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Loading data from {filepath}...")
    raw_dataset = pd.read_excel(filepath, usecols=['HARGA', 'LB', 'LT', 'KT', 'KM'])
    
    dataset = raw_dataset.rename(columns={
        'HARGA': 'Harga',
        'LB': 'Luas Bangunan',
        'LT': 'Luas Tanah',
        'KT': 'Jumlah Kamar Tidur',
        'KM': 'Jumlah Kamar Mandi'
    })
    
    print("Data loaded successfully.")
    print(dataset.head())
    return dataset

def train():
    data_path = 'DATA RUMAH.xlsx'
    model_dir = 'model'
    model_path = os.path.join(model_dir, 'linear_regression_model.pkl')
    
    try:
        df = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    feature_cols = ['Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi']
    target_col = 'Harga'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R^2 Score: {r2:.4f}")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
