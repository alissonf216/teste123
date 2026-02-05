#!/usr/bin/env python3
"""
Script para verificar se o aplicativo Streamlit está funcionando corretamente
"""

import requests
import time

def check_app():
    """Verifica se o app está funcionando"""
    try:
        # Aguardar um pouco para o app inicializar
        time.sleep(3)

        response = requests.get("http://localhost:8501", timeout=10)

        if response.status_code == 200:
            content = response.text

            # Verificar se há mensagens de erro
            if "Modelo não carregado" in content:
                print("ERRO: Modelo ainda não carregado")
                return False
            elif "Erro ao carregar o modelo" in content:
                print("ERRO: Ainda há erro de carregamento")
                return False
            elif "Simulador de Cenários" in content:
                print("SUCESSO: Aplicativo funcionando corretamente!")
                return True
            else:
                print("AVISO: Página carregada mas conteúdo não identificado")
                # Mostrar primeiras 500 caracteres para debug
                print("Conteúdo (primeiros 500 chars):")
                print(content[:500])
                print("...")
                return False
        else:
            print(f"ERRO: Status code {response.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"ERRO: Não foi possível conectar: {e}")
        return False

if __name__ == "__main__":
    success = check_app()
    print(f"Status final: {'OK' if success else 'ERRO'}")