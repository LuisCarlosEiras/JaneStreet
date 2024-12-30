# 2 - Pré-processamento

import dask.dataframe as dd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import gc
from tqdm import tqdm
import os
import joblib
from datetime import datetime

def create_output_dirs(base_path):
    """
    Cria diretórios para salvar os resultados
    """
    # Cria timestamp para o diretório de processamento
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define caminhos
    output_dir = os.path.join(base_path, 'preprocessed', timestamp)
    chunks_dir = os.path.join(output_dir, 'chunks')
    models_dir = os.path.join(output_dir, 'models')
    
    # Cria diretórios
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    return output_dir, chunks_dir, models_dir

def save_chunk(chunk, chunk_number, chunks_dir):
    """
    Salva um chunk processado em formato parquet
    """
    chunk_path = os.path.join(chunks_dir, f'chunk_{chunk_number:04d}.parquet')
    chunk.to_parquet(chunk_path, index=False)
    return chunk_path

def load_and_combine_chunks(chunks_dir, final_path):
    """
    Carrega e combina todos os chunks salvos
    """
    print("Combinando chunks...")
    # Lista todos os arquivos de chunk
    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith('.parquet')])
    
    # Combina usando dask para eficiência
    df_dask = dd.read_parquet(os.path.join(chunks_dir, 'chunk_*.parquet'))
    
    # Ordena e salva
    print("Ordenando e salvando resultado final...")
    df_dask = df_dask.sort_values(['date_id', 'time_id'])
    df_dask.to_parquet(final_path, write_index=False)
    
    return df_dask

def preprocess_data(data_path, output_base_path, chunk_size=1000000):
    """
    Pipeline de pré-processamento com salvamento em disco
    """
    # Cria diretórios de saída
    output_dir, chunks_dir, models_dir = create_output_dirs(output_base_path)
    print(f"Resultados serão salvos em: {output_dir}")
    
    # Inicializa processamento
    print("Iniciando pré-processamento dos dados...")
    df_dask = dd.read_parquet(f"{data_path}train.parquet", engine='pyarrow')
    
    # Análise inicial e seleção de colunas
    print("Analisando valores nulos...")
    null_percentages = df_dask.isnull().mean().compute() * 100
    valid_columns = list(null_percentages[null_percentages <= 30].index)
    
    # Seleciona colunas por tipo
    essential_columns = ['date_id', 'time_id', 'symbol_id', 'weight']
    feature_columns = [col for col in valid_columns if col.startswith('feature_')]
    responder_columns = [col for col in valid_columns if col.startswith('responder_')]
    selected_columns = essential_columns + feature_columns + responder_columns
    
    df_dask = df_dask[selected_columns]
    
    # Análise de correlação em amostra
    print("Analisando correlações em amostra...")
    total_rows = df_dask.shape[0].compute()
    sample_size = min(1000000, total_rows)
    sample_frac = sample_size / total_rows
    
    # Usa frac em vez de n para o sample
    df_sample = df_dask.sample(frac=sample_frac, random_state=42).compute()
    
    # Seleciona features importantes
    correlation_features = feature_columns + responder_columns
    correlation_matrix = df_sample[correlation_features].corr()
    
    important_features = set()
    for responder in responder_columns:
        corr_with_responder = correlation_matrix[responder].abs()
        selected = list(corr_with_responder[
            (corr_with_responder > 0.1) &
            (corr_with_responder.index.str.startswith('feature_'))
        ].index)
        important_features.update(selected)
    
    # Remove features altamente correlacionadas
    features_to_keep = list(important_features)
    to_drop = set()
    for i in range(len(features_to_keep)):
        if features_to_keep[i] not in to_drop:
            for j in range(i + 1, len(features_to_keep)):
                if features_to_keep[j] not in to_drop:
                    if abs(correlation_matrix.loc[features_to_keep[i], features_to_keep[j]]) > 0.95:
                        to_drop.add(features_to_keep[j])
    
    final_features = [f for f in features_to_keep if f not in to_drop]
    del df_sample, correlation_matrix
    gc.collect()
    
    # Salva lista de features selecionadas
    features_path = os.path.join(output_dir, 'selected_features.txt')
    with open(features_path, 'w') as f:
        f.write('\n'.join(final_features))
    
    # Prepara colunas finais
    final_columns = essential_columns + final_features + responder_columns
    
    # Inicializa scaler
    scaler = StandardScaler()
    
    # Processa dados em chunks
    print("Processando dados em chunks...")
    processed_chunks = []
    total_chunks = int(np.ceil(len(df_dask) / chunk_size))
    
    for i in tqdm(range(total_chunks)):
        # Carrega e processa chunk
        start_idx = i * chunk_size
        chunk = df_dask.loc[start_idx:start_idx + chunk_size - 1].compute()
        
        # Processa nulos e outliers
        for col in final_features:
            chunk[col] = chunk[col].fillna(chunk[col].median())
            Q1 = chunk[col].quantile(0.01)
            Q3 = chunk[col].quantile(0.99)
            chunk[col] = chunk[col].clip(lower=Q1, upper=Q3)
        
        # Seleciona colunas finais
        chunk = chunk[final_columns]
        
        # Salva chunk
        chunk_path = save_chunk(chunk, i, chunks_dir)
        processed_chunks.append(chunk_path)
        
        # Limpa memória
        del chunk
        gc.collect()
    
    # Combina chunks e salva resultado final
    final_path = os.path.join(output_dir, 'processed_data.parquet')
    df_final = load_and_combine_chunks(chunks_dir, final_path)
    
    # Salva scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"""
    Pré-processamento concluído!
    
    Arquivos gerados:
    - Dados processados: {final_path}
    - Chunks intermediários: {chunks_dir}
    - Scaler: {scaler_path}
    - Features selecionadas: {features_path}
    """)
    
    return df_final, scaler, final_features

if __name__ == "__main__":
    try:
        # Configuração de caminhos
        data_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/"
        output_base_path = "/kaggle/working/"  # Diretório com suporte a até 20GB
        
        # Executa o pré-processamento
        preprocess_data(
            data_path=data_path,
            output_base_path=output_base_path,
            chunk_size=250000
        )
        
        # Libera memória
        gc.collect()
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")

'''Resultados serão salvos em: /kaggle/working/preprocessed/20241120_001338
Iniciando pré-processamento dos dados...
Analisando valores nulos...
Analisando correlações em amostra...
Processando dados em chunks...
100%|██████████| 189/189 [1:59:57<00:00, 38.08s/it]  
Combinando chunks...
Ordenando e salvando resultado final...

    Pré-processamento concluído!
    
    Arquivos gerados:
    - Dados processados: /kaggle/working/preprocessed/20241120_001338/processed_data.parquet
    - Chunks intermediários: /kaggle/working/preprocessed/20241120_001338/chunks
    - Scaler: /kaggle/working/preprocessed/20241120_001338/models/scaler.joblib
    - Features selecionadas: /kaggle/working/preprocessed/20241120_001338/selected_features.txt'''
