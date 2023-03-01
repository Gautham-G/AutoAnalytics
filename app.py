import streamlit as st
import pandas as pd
import numpy as np


st.title("Custom Auto Analytics!")

uploaded_file = st.file_uploader("Upload a CSV format dataframe")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df.head())
  
  