# 3 - Revisão e Ajuste do Pré-processamento 

import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

def review_and_adjust_preprocessing(data_path, features_path, scaler_path):
    """
    Revisa e ajusta o pré-processamento dos dados.
    """
    try:
        print("=== Revisão e Ajuste do Pré-processamento ===")
        
        # Carregar dados processados
        print("Carregando dados processados...")
        df = pd.read_parquet(data_path)

        # Carregar features selecionadas
        print("Carregando features selecionadas...")
        with open(features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]

        # Carregar scaler
        print("Carregando scaler...")
        scaler = joblib.load(scaler_path)

        # Estatísticas descritivas
        print("Calculando estatísticas descritivas...")
        stats = df[selected_features].describe()

        # Verificar problemas de escala
        print("Verificando escala...")
        scaled_features = [
            col for col in selected_features 
            if col not in ['date_id', 'time_id', 'symbol_id']
        ]
        mean_check = stats.loc['mean', scaled_features]
        std_check = stats.loc['std', scaled_features]

        scale_issues = mean_check[mean_check.abs() > 1e-2].index.union(
            std_check[std_check > 2].index
        )
        
        if not scale_issues.empty:
            print(f"ATENÇÃO: Problemas encontrados na escala das seguintes features: {scale_issues.tolist()}")

            # Corrigir escala
            for col in scale_issues:
                print(f"Corrigindo escala para a feature: {col}")
                df[col] = scaler.fit_transform(df[[col]])
                # Recalcular estatísticas após correção
                col_stats = df[col].describe()
                print(f"Após correção - mean: {col_stats['mean']:.4f}, std: {col_stats['std']:.4f}")
            
            # Atualizar scaler e salvar novamente
            updated_scaler_path = scaler_path.replace(".joblib", "_adjusted.joblib")
            print("Salvando scaler ajustado...")
            joblib.dump(scaler, updated_scaler_path)
            print(f"Scaler ajustado salvo em: {updated_scaler_path}")

        else:
            print("Escala aplicada corretamente. Nenhuma correção necessária.")

        # Atualizar o arquivo de features selecionadas
        print("Atualizando lista de features selecionadas...")
        with open(features_path, "w") as f:
            f.write("\n".join(selected_features))

        # Salvar os dados corrigidos
        corrected_data_path = data_path.replace("processed_data", "processed_data_adjusted")
        print("Salvando dados corrigidos...")
        df.to_parquet(corrected_data_path, index=False)
        print(f"Dados corrigidos salvos em: {corrected_data_path}")

        # Recalcular estatísticas finais após todas as correções
        print("Recalculando estatísticas descritivas após correções...")
        final_stats = df[selected_features].describe()

        # Verificar novamente problemas de escala
        final_mean_check = final_stats.loc['mean', scaled_features]
        final_std_check = final_stats.loc['std', scaled_features]

        final_scale_issues = final_mean_check[final_mean_check.abs() > 1e-2].index.union(
            final_std_check[final_std_check > 2].index
        )

        # Gerar relatório final
        print("Gerando relatório de revisão...")
        report = {
            "n_linhas": len(df),
            "n_colunas": len(df.columns),
            "features": list(df.columns),
            "estatisticas": final_stats.to_dict(),
            "problemas_escala": final_scale_issues.tolist(),
        }

        print("\n=== Relatório Final ===")
        print(f"Número de linhas: {report['n_linhas']}")
        print(f"Número de colunas: {report['n_colunas']}")
        print(f"Features: {report['features'][:10]}... (total {len(report['features'])})")
        print(f"Problemas de escala: {len(report['problemas_escala'])} features com problemas.")

        return report

    except Exception as e:
        print(f"Erro durante a revisão e ajuste: {str(e)}")
        return None


# Caminhos para os arquivos
data_path = "/kaggle/working/preprocessed/20241121_102946/processed_data.parquet"
features_path = "/kaggle/working/preprocessed/20241121_102946/selected_features.txt"
scaler_path = "/kaggle/working/preprocessed/20241121_102946/models/scaler.joblib"

# Executa a revisão e ajuste do pré-processamento
report = review_and_adjust_preprocessing(data_path, features_path, scaler_path)
