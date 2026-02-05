#!/usr/bin/env python3
"""
Script para testar apenas o carregamento do modelo
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Classe dummy para compatibilidade
class ModelBundle:
    """Classe dummy para compatibilidade com arquivos joblib que contêm ModelBundle"""
    def __init__(self, pipeline=None, feature_columns=None):
        self.pipeline = pipeline
        self.feature_columns = feature_columns

def test_model_loading():
    """Testa o carregamento do modelo"""
    from pathlib import Path
    import joblib

    model_path = Path('tch_rf_bundle.joblib')
    if not model_path.exists():
        print("ERRO: Arquivo tch_rf_bundle.joblib não encontrado")
        return False

    try:
        print("Tentando carregar o modelo...")

        # Tentar carregar o bundle normalmente
        bundle = joblib.load(model_path)
        print(f"Bundle carregado. Tipo: {type(bundle)}")

        # Verificar se tem os atributos necessários
        if hasattr(bundle, 'pipeline'):
            print("[OK] Pipeline encontrado")
            print(f"  Tipo do pipeline: {type(bundle.pipeline)}")

            # Verificar feature_columns
            if hasattr(bundle, 'feature_columns'):
                print(f"[OK] Feature columns encontrado: {len(bundle.feature_columns)} features")
                print(f"  Features: {bundle.feature_columns[:5]}...")
            elif hasattr(bundle.pipeline, 'feature_names_in_'):
                print(f"[OK] Feature names extraídos do pipeline: {len(bundle.pipeline.feature_names_in_)} features")
                bundle.feature_columns = list(bundle.pipeline.feature_names_in_)
            else:
                print("[WARN] Feature columns não encontrados - usando padrão")

            # Testar uma predição simples
            try:
                import pandas as pd
                import numpy as np

                # Criar dados de teste
                test_data = pd.DataFrame({
                    'EXPANSAO': [1.0], 'Devolucao': [0.0], 'Reforma': [0.0], 'Vinhaca_E': [0.0],
                    'TORTA': [0.0], 'SISTEMA_COL': [1.0], 'UNID_IND': [1.0], 'IRRIGADO': [0.0],
                    'QTDE_MESES': [12.0], 'MES_PLANTIO': [8.0], 'area_prod': [10.0],
                    'PLANTIO_MECANIZADO': [1.0], 'TRATOS_CANA_PLANTA': [2.0],
                    'PREPARO_SOLO': [1.0], 'TRATOS_CANA_PLANTA_PARC': [0.0],
                    'PREPARO_SOLO_PARCERIA': [0.0], 'PLANTIO_SEMI_MECANIZADO': [0.0],
                    'CONTROLE_AGRICOLA': [1.0], 'GESTAO_DA_QUALIDADE': [1.0],
                    'TOTAL_OPERS': [5.0], 'ambiente_correto_bula': [1.0],
                    'tempo_colheita_correto_bula': [1.0], 'incendios': [0.0],
                    'EVI': [0.3], 'GNDVI': [0.5], 'NDVI': [0.6], 'NDWI': [0.2], 'SAVI': [0.25],
                    'gdd': [2000.0], 'rainfall': [1000.0], 'prev_rainfall': [900.0]
                })

                prediction = bundle.pipeline.predict(test_data)
                print(f"[OK] Predição de teste bem-sucedida: {prediction[0]:.2f}")

                return True

            except Exception as e:
                print(f"[ERRO] Erro na predição de teste: {e}")
                return False

        # Se não tem pipeline, pode ser que o bundle seja o próprio pipeline
        elif hasattr(bundle, 'predict'):
            print("[OK] Bundle é o próprio modelo (RandomForest)")
            bundle_obj = ModelBundle(pipeline=bundle)

            if hasattr(bundle, 'feature_names_in_'):
                bundle_obj.feature_columns = list(bundle.feature_names_in_)
                print(f"[OK] Feature names extraídos: {len(bundle_obj.feature_columns)} features")

            return True

        else:
            print("[ERRO] Arquivo não contém pipeline válido")
            return False

    except Exception as e:
        print(f"[ERRO] Erro ao carregar modelo: {str(e)}")

        # Tentar abordagem alternativa
        try:
            print("Tentando abordagem alternativa...")
            import pickle

            with open(model_path, 'rb') as f:
                data = pickle.load(f, fix_imports=True, encoding='latin1')

            print(f"Dados carregados como: {type(data)}")

            if isinstance(data, dict):
                if 'pipeline' in data:
                    print("[OK] Pipeline encontrado no dicionário")
                    bundle = ModelBundle(pipeline=data['pipeline'])
                    if 'feature_columns' in data:
                        bundle.feature_columns = data['feature_columns']
                        print("[OK] Feature columns encontrado no dicionário")
                    return True

            print("[ERRO] Não foi possível extrair dados do arquivo")
            return False

        except Exception as e2:
            print(f"[ERRO] Falha na abordagem alternativa: {str(e2)}")
            return False

if __name__ == "__main__":
    print("=== Teste de Carregamento do Modelo ===")
    success = test_model_loading()
    print(f"\nResultado: {'SUCESSO' if success else 'FALHA'}")
    sys.exit(0 if success else 1)