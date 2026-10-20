import numpy as np
import joblib
import streamlit as st

obj = joblib.load("california.joblib")
model = obj["model"]
cols = obj["columns"]

st.title("California app")

In = []
for i in cols:
    v = st.number_input(f"enter {i} value")
    In.append(v)


if st.button("Click"):
    out = model.predict(np.array([In]))[0]
    st.success(f"California house value is {out}")