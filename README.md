# 🏠 House Price Prediction (Linear Regression)

A professional, interactive web application built with **Streamlit** that predicts house prices using a **Linear Regression** machine learning model. This tool analyzes the relationship between various housing features (Land Area, Building Area, Bedrooms, Bathrooms) and predicts the estimated market price.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)

## ✨ Features

*   **Interactive Dashboard**: User-friendly interface powered by Streamlit.
*   **Data Visualization**: Dynamic scatter plots to analyze relationships between features (e.g., Building Area vs. Price) using Seaborn and Matplotlib.
*   **Real-time Prediction**: Instant price estimation based on user inputs.
*   **Currency Formatting**: Automatic formatting of predictions into Indonesian Rupiah (IDR) using `babel`.
*   **Optimized Performance**: Utilizes Streamlit's caching mechanisms (`@st.cache_data`, `@st.cache_resource`) for fast reloading.

## 🛠️ Tech Stack

*   **Python**: Core programming language.
*   **Streamlit**: Web framework for data apps.
*   **Scikit-learn**: Machine learning library for the Linear Regression model.
*   **Pandas**: Data manipulation and analysis.
*   **Seaborn & Matplotlib**: Data visualization.
*   **OpenPyXL**: Reading Excel datasets.

## 📂 Project Structure

```
├── DATA RUMAH.xlsx      # Dataset source (Excel format)
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have [Python](https://www.python.org/) installed on your system.

### Installation

1.  **Clone or Download the repository** (if applicable) or navigate to your project folder.

2.  **Install Dependencies**:
    Run the following command to install all required libraries:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  Start the Streamlit server:
    ```bash
    streamlit run main.py
    ```

2.  The application will automatically open in your default web browser at `http://localhost:8501`.

## 📊 Dataset Format

The application expects an Excel file named `DATA RUMAH.xlsx` in the root directory with the following columns:

| Column Name | Description               |
| :---        | :---                      |
| **HARGA**   | House Price (Target)      |
| **LB**      | Building Area (m²)        |
| **LT**      | Land Area (m²)            |
| **KT**      | Number of Bedrooms        |
| **KM**      | Number of Bathrooms       |

## 📝 License

This project is open-source and available for educational purposes.