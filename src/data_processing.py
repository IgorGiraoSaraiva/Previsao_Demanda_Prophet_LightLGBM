import pandas as pd
import joblib
import streamlit as st
from config import path_train_data, path_test_data,path_ML_train_data, path_ML_test_data, path_lgbm_otim, path_features_ML, dados_metricas


df_train = pd.read_csv(path_train_data)
df_test = pd.read_csv(path_train_data)
df_ML = pd.read_csv(path_ML_train_data)
df_ML_test = pd.read_csv(path_ML_test_data)


def converte_df(df):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

df_train = converte_df(df_train)
df_test = converte_df(df_test)
df_ML = converte_df(df_ML)
df_ML_test = converte_df(df_ML_test)  
df_resultados = pd.DataFrame(dados_metricas)