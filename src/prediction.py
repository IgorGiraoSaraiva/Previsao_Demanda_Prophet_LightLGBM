import pandas as pd
from prophet import Prophet
import holidays
from config import path_lgbm_otim, path_prophet, path_features_ML
import streamlit as st
import joblib


lgbm_otimo = joblib.load(path_lgbm_otim)
modelo_prophet = joblib.load(path_prophet)
features_do_modelo = joblib.load(path_features_ML)


@st.cache_data
def previsao_prophet(df,periodo,granularidade):
    '''
    Função que lê um dataframe, quantidade de dias para prever e prevê valores para diferente granularidades de data.
    '''
    # Agrupa de acordo com a granularidade desejada
    vendas_agg = df['sales'].resample(granularidade).sum()
    
    # Adpata dataframe para o prophet
    df_prophet = vendas_agg.reset_index()
    df_prophet.columns = ['ds', 'y']
    #Inclusão dos feriados
    df_feriados = pd.DataFrame([{'holiday': data[1], 
                                 'ds': data[0]} for data in holidays.Brazil(years=range(2013, 2018)).items()])
    df_feriados['ds'] = pd.to_datetime(df_feriados['ds'])

    # Treinamento do modelo
    modelo_prophet = Prophet(holidays= df_feriados)

    modelo_prophet.fit(df_prophet)


    df_futuro = modelo_prophet.make_future_dataframe(periods = periodo, 
                                                     freq = granularidade)
    forecast = modelo_prophet.predict(df_futuro)
    return df_prophet, forecast


def previsao_lgbm(df,loja,item,agregacao,df_past):
    '''
    Função que recebe um data frame, número da loja e número do item e faz a previsão de demanda.
    '''
    # Condicionais para filtrar dataframe
    if loja == 0 and item == 0:
        df = df
        df_past = df_past
    elif loja == 0 and item != 0:
        df = df[df['item'] == item]
        df_past = df_past[df_past['item'] == item]
    elif loja !=0 and item == 0:
        df = df[df['store'] == loja]
        df_past = df_past[df_past['store'] == loja]        
    else:
        df = df[(df['store'] == loja) & (df['item'] == item)]
        df_past = df_past[(df_past['store'] == loja) & (df_past['item'] == item)]

    # Realizando revisão com dateframe filtrado
    previsao = pd.Series(lgbm_otimo.predict(df),
                         index = df.index)
    
    #Montando dataframe com previsão
    df_futuro = df.copy()
    df_futuro['sales'] = previsao
    
    # Agregando rpevisão para plotagem do gráfico
    previsao_agg = previsao.resample(agregacao).sum().head(30)
    # Histórico dos últimos 30 dias 
    vendas_past = df_past['sales'].resample('D').sum()
    previsao_past =  vendas_past.last('30D')
    return previsao, previsao_agg, previsao_past, df_futuro