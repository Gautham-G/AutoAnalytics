import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Simple (Linear regression) auto analytics!")

uploaded_file = st.file_uploader("Upload a CSV format dataframe")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df.head())
  
  y = st.selectbox('Choose your response variable (y)', df.columns)
  st.write('you selected', y)
  data = st.multiselect("Predictor variables (x)", df.columns)
  
  plt.plot(x, y)
  
