
import numpy as np
import joblib
import streamlit as st

st.set_page_config(
    page_title="California House Prediction",
    page_icon="🏡",
    layout="wide"
)

# Load model
obj = joblib.load("california.joblib")
model = obj["model"]
cols = obj["columns"]

# CSS styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f0fdfa);
}

.header {
    background: linear-gradient(90deg, #1e3a8a, #0f766e);
    padding: 28px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.header h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.header p {
    font-size: 16px;
    margin: 0;
}

.form-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    background: #0f766e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    font-size: 17px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #115e59;
    color: white;
    border: none;
}

.result-card {
    background: #d1fae5;
    border-left: 6px solid #059669;
    padding: 20px;
    border-radius: 10px;
    color: #065f46;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>🏡 California House Price Prediction</h1>
    <p>Machine Learning Based House Value Estimator</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<div class="form-card">', unsafe_allow_html=True)
st.subheader("🏠 Enter House Details")
st.write("Enter the values below to estimate the house value.")

In = []
left, right = st.columns(2)

for index, col in enumerate(cols):
    with (left if index % 2 == 0 else right):
        value = st.number_input(
            f"Enter {col} value",
            value=0.0,
            key=col
        )
        In.append(value)

st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if st.button("🔍 Predict House Value"):
    out = model.predict(np.array([In]))[0]

    st.markdown(
        f'<div class="result-card">'
        f'🏡 Predicted House Value: {out:.2f}'
        f'</div>',
        unsafe_allow_html=True
    )
