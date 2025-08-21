# config.py
# Caminhos dos Arquivos 
path_train_data = 'data/raw/train.csv'
path_test_data = 'data/raw/test.csv'
path_ML_train_data = 'data/processed/df_ML.csv'
path_ML_test_data = 'data/processed/df_ML_test.csv'
path_lgbm_otim = 'models/lgbm_otimizado.joblib'
path_prophet = 'models/prophet.joblib'
path_features_ML = 'models/lista_de_features.joblib'

# Configurações do Dashboard PREVISÃO
opcoes_granularidade = {
'Diário': 'D',
'Semanal': 'W',
'Mensal': 'ME',
'Trimestral': 'QE',
'Anual': 'YE'
}
periodos_granularidade = {
    'D': 1095,
    'W': 156,    
    'ME': 36,    
    'QE': 12,    
    'YE': 3     
}

dados_metricas = {
    'Métrica': ['R²', 'MAE', 'RMSE'],
    'Treino': [0.936, 5.435, 7.031],
    'Teste': [0.937, 6.093, 7.924]
}
