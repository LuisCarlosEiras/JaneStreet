# 5 - modelo 24/11/24

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import os

# Caminhos para os dados pré-processados
data_path = "/kaggle/working/preprocessed/20241121_102946/processed_data_adjusted.parquet"
scaler_path = "/kaggle/working/preprocessed/20241121_102946/models/scaler_adjusted.joblib"
features_path = "/kaggle/working/preprocessed/20241121_102946/selected_features.txt"

# Carregar dados e configurações
print("Carregando dados pré-processados...")
data = pd.read_parquet(data_path)

print("Carregando scaler...")
scaler = joblib.load(scaler_path)

print("Carregando lista de features selecionadas...")
with open(features_path, "r") as f:
    selected_features = f.read().splitlines()

# Separar features e variável-alvo
print("Separando variáveis de entrada e saída...")
X = data[selected_features]
y = data['responder_0']  # Alvo de previsão (contínuo)

# Dividir os dados em conjuntos de treino e teste
print("Dividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar o modelo XGBoost para Regressão
print("Treinando o modelo XGBoost para regressão...")
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    objective="reg:squarederror"
)
model.fit(X_train, y_train)

# Avaliação do modelo
print("Avaliando o modelo...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n=== Métricas de Avaliação ===")
print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {np.sqrt(mse):.4f}")

# Salvar o modelo treinado
output_model_path = "/kaggle/working/preprocessed/20241121_102946/models/xgboost_regressor.joblib"
os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
print(f"Salvando modelo treinado em: {output_model_path}")
joblib.dump(model, output_model_path)

print("Treinamento concluído!")

'''=== Métricas de Avaliação ===
Erro Quadrático Médio (MSE): 0.0060
Raiz do Erro Quadrático Médio (RMSE): 0.0774
Salvando modelo treinado em: /kaggle/working/preprocessed/20241121_102946/models/xgboost_regressor.joblib
Treinamento concluído!'''
