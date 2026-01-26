import pandas as pd
import pickle
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
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
    metrics_path = 'model/metadata.json'
    
    try:
        df = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare features and target
    X = df.drop(columns=['Harga'])
    y = df['Harga'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Preprocessing and Pipeline
    numeric_features = ["Luas Bangunan", "Luas Tanah", "Jumlah Kamar Tidur", "Jumlah Kamar Mandi"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="passthrough"
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model = TransformedTargetRegressor(
        regressor=model_pipeline, 
        transformer=StandardScaler()
    )
    
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Evaluation:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  R^2 Score: {r2:.4f}")
    
    # Save Metrics
    metrics = {
        "mse": mse,
        "r2score": r2,
        "model": "Linear-Regression"
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save Model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()