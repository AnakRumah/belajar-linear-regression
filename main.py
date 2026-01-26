import time
import pickle
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from babel.numbers import format_currency

# Configuration
st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered"
)

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Loads and preprocesses the housing dataset.
    """
    try:
        raw_dataset = pd.read_excel('DATA RUMAH.xlsx', usecols=['HARGA', 'LB', 'LT', 'KT', 'KM'])
        dataset = raw_dataset.rename(columns={
            'HARGA': 'Harga',
            'LB': 'Luas Bangunan',
            'LT': 'Luas Tanah',
            'KT': 'Jumlah Kamar Tidur',
            'KM': 'Jumlah Kamar Mandi'
        })
        return dataset
    except FileNotFoundError:
        st.error("File 'DATA RUMAH.xlsx' not found. Please ensure the dataset is in the correct directory.")
        return pd.DataFrame()

@st.cache_resource
def train_model(dataset: pd.DataFrame) -> LinearRegression:
    """
    Trains the Linear Regression model.
    """
    if dataset.empty:
        return None
        
    X = dataset[['Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi']].values
    y = dataset['Harga'].values
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def plot_relationship(dataset: pd.DataFrame, feature: str):
    """
    Creates a scatter plot for the selected feature against Price.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dataset, x=feature, y='Harga', ax=ax, color='teal')
    ax.set_title(f'Hubungan {feature} vs Harga', fontsize=14)
    ax.set_ylabel('Harga (Rp)', fontsize=12)
    ax.set_xlabel(feature, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def predict_price(model: LinearRegression, lb: float, lt: float, kt: int, km: int) -> float:
    """
    Predicts the house price based on features.
    """
    # Feature order: LB, LT, KT, KM
    features = [[lb, lt, kt, km]]
    prediction = model.predict(features)[0]
    return max(0, round(prediction, 0))

def main():
    # Header
    st.title("🏡 Linear Regression")
    st.subheader("Prediksi Harga Rumah")
    st.caption("Tech Stack: Scikit-learn, Streamlit, Pandas, Seaborn")
    
    with st.expander("Lihat Rumus"):
        st.latex(r"y = ax + b")

    # Load Data
    dataset = load_dataset()
    if dataset.empty:
        return

    # Show Data
    st.write("### Data Preview")
    st.dataframe(dataset.head(10), use_container_width=True)
    st.caption(f"Total Data: {len(dataset)} baris")

    # Train Model
    model = train_model(dataset)

    # Visualization Section
    st.divider()
    st.header("📊 Analisis Data")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        feature_option = st.selectbox(
            'Pilih Variabel:',
            ('Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi')
        )
    
    with col2:
        fig = plot_relationship(dataset, feature_option)
        st.pyplot(fig)

    # Prediction Section
    st.divider()
    st.header("🔮 Prediksi Harga")
    st.info("Masukkan parameter rumah di bawah ini:")

    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        lb_input = st.number_input("Luas Bangunan (m²)", min_value=20, max_value=5000, value=100, step=10)
        lt_input = st.number_input("Luas Tanah (m²)", min_value=20, max_value=5000, value=120, step=10)

    with col_input2:
        kt_input = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=20, value=3, step=1)
        km_input = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=20, value=2, step=1)

    if st.button('Hitung Estimasi Harga', type="primary", use_container_width=True):
        with st.spinner('Sedang menghitung...'):
            time.sleep(0.5) # User experience delay
            
            estimated_price = predict_price(model, lb_input, lt_input, kt_input, km_input)
            
            if estimated_price <= 0:
                st.error("Kombinasi input menghasilkan prediksi negatif atau nol. Coba sesuaikan input.")
            else:
                formatted_price = format_currency(estimated_price, 'Rp', locale='id_ID')
                st.success(f"Estimasi Harga Rumah: **{formatted_price}**")

if __name__ == "__main__":
    main()