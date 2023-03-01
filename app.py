import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Simple (Linear regression) auto analytics!")

uploaded_file = st.file_uploader("Upload a CSV format dataframe")
if uploaded_file is not None:
  
# Find a way to dynamically change type of column
# Find a way to automatically impute null values 

  df = pd.read_csv(uploaded_file)
  st.write(df.head())
  
  response_col = st.selectbox('Choose your response variable (y)', df.columns)
  st.write('Your response variable is : ', response_col)
  
  data_columns = st.multiselect("Predictor variables (x)", df.columns)
  if data_columns is not None:
    x = df[data_columns]
    y = df[response_col]

    train = df.sample(frac=0.8,random_state=200)
    test = df.drop(train.index)

    x_train = np.array(train[data_columns])
    x_test = np.array(test[response_col])

    y_train = np.array(train[data_columns])
    y_test = np.array(test[response_col])

    bias_col = np.ones((len(x_train), 1))
  #   st.write(bias_col.shape)
  #   st.write(x_train.shape)
    x_train = np.hstack((bias_col, x_train))
  #   st.write(x_train.shape)

    b_hat = np.linalg.inv(x_train.T@x_train)@x_train.T@y_train
  #   st.write(b_hat[0], b_hat[1])
    y_hat = b_hat[0] + b_hat[1]*x_test


    fig, ax = plt.subplots()

    ax.plot(x_test, y_test, label = 'test truth')
    ax.plot(x_test, y_hat, label = 'test pred')
    ax.legend()

    st.pyplot(fig)

    rmse = np.linalg.norm(y_hat-y_test)
    st.write('RMSE is ', rmse)
    if(rmse<1e-5):
      st.write('Woah! Thats a great model!')
    else:
      st.write('Bad model :(')
  else:
    st.write('Choose predictors!')

  
#   choose alpha with st.slider
#   display type of plot with st.multiselect
#   display qq, residual etc
#   Find a way to check for multi
  
