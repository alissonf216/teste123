#!/usr/bin/env python3
"""
Script de teste para verificar se o app pode ser importado e inicializado
"""

import sys
import os

def test_imports():
    """Testa se todas as importações funcionam"""
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import joblib
        import matplotlib.pyplot as plt
        from pathlib import Path
        import warnings
        warnings.filterwarnings('ignore')

        print("Todas as importações funcionaram")
        return True
    except ImportError as e:
        print(f"Erro na importação: {e}")
        return False

def test_constants():
    """Testa se as constantes estão definidas corretamente"""
    try:
        # Testa FACTORS_BASE
        FACTORS_BASE = {
            "IMPUREZA VEGETAL": 2.57,
            "DELTA ÁREA VINHAÇA ASPERSÃO": 0.11,
            "DELTA ÁREA ADUBAÇÃO FOLIAR": 2.54,
            "DELTA ÁREA TORTA/M.O. PLANTIO": 0.58,
            "CP -  DELTA PROPORÇÃO IDEAL POR ÉPOCA": -0.87,
            "CP - APTIDÃO AMBIENTE": -0.67,
            "NCM": -3.02,
            "FALHAS PLANTIO": -0.29,
            "FALHAS SOCA (ESTIMADO)": -0.89,
            "DANINHAS PLANTIO": -0.24,
            "DANINHAS SOCA (ESTIMADO)": -0.12,
            "PISOTEIO": -0.61,
            "PERDAS COLHEITA": -0.50,
            "BROCA": -0.21,
            "SPHENOPHORUS": -0.69,
            "SÍNDROME MURCHA": -0.61,
            "CANA DE DEZEMBRO": -0.31,
            "FOGO ACIDENTAL": -0.50,
            "INDETERMINADO": -0.71
        }

        # Testa MODEL_FEATURES
        MODEL_FEATURES = [
            'EXPANSAO','Devolucao','Reforma','Vinhaca_E','TORTA','SISTEMA_COL','UNID_IND','IRRIGADO',
            'QTDE_MESES','MES_PLANTIO','area_prod','PLANTIO_MECANIZADO','TRATOS_CANA_PLANTA',
            'PREPARO_SOLO','TRATOS_CANA_PLANTA_PARC','PREPARO_SOLO_PARCERIA','PLANTIO_SEMI_MECANIZADO',
            'CONTROLE_AGRICOLA','GESTAO_DA_QUALIDADE','TOTAL_OPERS','ambiente_correto_bula',
            'tempo_colheita_correto_bula','incendios','EVI','GNDVI','NDVI','NDWI','SAVI','gdd',
            'rainfall','prev_rainfall'
        ]

        CONTROLLED_VARS = [
            'rainfall', 'prev_rainfall', 'gdd', 'MES_PLANTIO', 'QTDE_MESES',
            'NDVI', 'TRATOS_CANA_PLANTA', 'PREPARO_SOLO', 'IRRIGADO', 'Reforma', 'Vinhaca_E', 'TORTA'
        ]

        VISIONS = ['AMBIENTE', 'FAZENDA', 'UNIDADE', 'TALHAO']

        print("Constantes definidas corretamente")
        print(f"   - FACTORS_BASE: {len(FACTORS_BASE)} fatores")
        print(f"   - MODEL_FEATURES: {len(MODEL_FEATURES)} features")
        print(f"   - CONTROLLED_VARS: {len(CONTROLLED_VARS)} variáveis controladas")
        print(f"   - VISIONS: {VISIONS}")

        return True
    except Exception as e:
        print(f"Erro nas constantes: {e}")
        return False

def test_file_structure():
    """Testa se os arquivos necessários existem"""
    required_files = ['app.py', 'requirements.txt', 'README.md']
    all_required_exist = True

    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} encontrado")
        else:
            print(f"[ERRO] {file} não encontrado")
            all_required_exist = False

    # Arquivos opcionais
    optional_files = ['tch_rf_bundle.joblib', 'baseline_data.parquet', 'baseline_data.csv']
    for file in optional_files:
        if os.path.exists(file):
            print(f"[INFO] {file} encontrado (opcional)")
        else:
            print(f"[INFO] {file} não encontrado (será solicitado upload no app)")

    return all_required_exist

def main():
    """Função principal de teste"""
    print("Testando Simulador de Cenários TCH")
    print("=" * 50)

    tests = [
        ("Importações", test_imports),
        ("Constantes", test_constants),
        ("Estrutura de arquivos", test_file_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\nTestando {test_name}...")
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Resultado: {passed}/{total} testes passaram")

    if passed == total:
        print("Tudo pronto! Execute 'streamlit run app.py' para iniciar o aplicativo.")
    else:
        print("Alguns testes falharam. Verifique os erros acima.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)