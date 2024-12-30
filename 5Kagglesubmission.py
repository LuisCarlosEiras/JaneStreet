import os
import pandas as pd
import polars as pl
import kaggle_evaluation.jane_street_inference_server

lags_: pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Realiza predições com base nos dados fornecidos."""
    print("Iniciando a função de predição.")
    
    global lags_
    if lags is not None:
        print("Atualizando lags com os valores fornecidos.")
        lags_ = lags
    else:
        print("Nenhum lag foi fornecido. Usando os lags anteriores.")
    
    print(f"Conjunto de teste recebido: {len(test)} linhas.")

    # Criando previsões
    try:
        predictions = test.select(
            'row_id',
            pl.lit(0.0).alias('responder_6'),
        )
        print("Previsões geradas com sucesso.")
    except Exception as e:
        print(f"Erro ao gerar previsões: {str(e)}")
        raise

    # Validação da saída
    if not isinstance(predictions, (pl.DataFrame, pd.DataFrame)):
        raise TypeError("As previsões devem ser um DataFrame do tipo Polars ou Pandas.")
    
    if list(predictions.columns) != ['row_id', 'responder_6']:
        raise ValueError("As colunas das previsões devem ser ['row_id', 'responder_6'].")
    
    if len(predictions) != len(test):
        raise ValueError("O número de linhas nas previsões deve ser igual ao número de linhas no conjunto de testes.")

    return predictions

print("Inicializando o servidor de inferência.")
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Modo competição detectado. Iniciando o servidor.")
    inference_server.serve()
else:
    print("Modo local detectado. Verificando arquivos locais.")
    
    test_path = '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet'
    lags_path = '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet'

    if os.path.exists(test_path) and os.path.exists(lags_path):
        print("Arquivos locais encontrados. Executando gateway local.")
        inference_server.run_local_gateway((test_path, lags_path))
    else:
        print("Arquivos locais não encontrados. Certifique-se de que os caminhos estão corretos.")
        raise FileNotFoundError("Os arquivos de teste ou lags não foram encontrados.")

print("Processamento concluído.")
