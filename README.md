# 🌾 Simulador de Cenários - TCH

Aplicativo Streamlit para simulação de cenários de produção de cana-de-açúcar usando modelo preditivo RandomForest.

## 📋 Visão Geral

Este simulador permite:
- Simular cenários de produção de cana-de-açúcar (TCH - Toneladas de Cana por Hectare)
- Usar modelo RandomForest treinado para predições
- Explorar impactos de diferentes fatores operacionais
- Visualizar análise de cascata (waterfall) dos impactos
- Exportar resultados detalhados para análise de talhões

## 🚀 Instalação e Execução

### Pré-requisitos

- Python 3.8+
- Arquivos de dados: `tch_rf_bundle.joblib` e `baseline_data.parquet` (ou `baseline_data.csv`)

### Instalação

1. Clone ou baixe os arquivos do projeto
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

### Execução

```bash
streamlit run app.py
```

O aplicativo estará disponível em `http://localhost:8501`

## 📁 Estrutura de Arquivos

```
├── app.py                    # Aplicativo principal Streamlit
├── requirements.txt          # Dependências Python
├── README.md                 # Este arquivo
├── tch_rf_bundle.joblib      # Modelo RandomForest treinado (obrigatório)
└── baseline_data.parquet     # Dados baseline (obrigatório)
   └── baseline_data.csv      # Alternativa ao parquet
```

## 🎯 Como Usar

### 1. Configuração Inicial

#### Arquivos Necessários

**Modelo (`tch_rf_bundle.joblib`)**:
- Arquivo joblib contendo o bundle do modelo treinado
- Deve ter `bundle.pipeline` (modelo RandomForest) e `bundle.feature_columns` (lista de features)

**Dados Baseline**:
- `baseline_data.parquet` ou `baseline_data.csv`
- Deve conter todas as features do modelo + colunas de agrupamento
- Colunas de agrupamento esperadas: AMBIENTE, FAZENDA/CD_FAZENDA, UNIDADE/UNID_IND, TALHAO/COD

#### Upload de Dados (se arquivos não existirem)

Se os arquivos de dados não estiverem presentes, o app oferecerá upload:
1. Faça upload do arquivo baseline (parquet ou CSV)
2. O arquivo será salvo localmente para uso futuro

### 2. Configuração da Simulação

#### Sidebar - Configurações

**Visão de Simulação**:
- Escolha entre: AMBIENTE, FAZENDA, UNIDADE, TALHAO
- Define como os dados serão agrupados para cálculo do baseline

**Mapeamento de Colunas** (opcional):
- Permite ajustar quais colunas representam cada dimensão
- Útil se os nomes das colunas forem diferentes do padrão

**Seleção da Dimensão**:
- Escolha o valor específico da dimensão selecionada
- Ex: AMBIENTE=7, FAZENDA=X, etc.

### 3. Controles de Simulação

#### 🌤️ Clima / Energia (Coluna Esquerda)
- **Rainfall (mm)**: Precipitação atual
- **Previous Rainfall (mm)**: Precipitação anterior
- **Growing Degree Days (GDD)**: Dias grau de crescimento

#### 📅 Calendário / Ciclo (Coluna Central Superior)
- **Mês de Plantio**: Seleção 1-12
- **Quantidade de Meses**: Duração do ciclo

#### ⚠️ Ofensores Operacionais (Coluna Central Inferior)
- **NDVI**: Índice de vegetação (0-1)
- **Tratos Cana Planta**: Percentual vs baseline (-30% a +30%)
- **Preparo do Solo**: Toggle (0/1)
- **Irrigado**: Toggle (0/1)
- **Reforma**: Toggle (0/1)
- **Vinhaça E**: Toggle (0/1)
- **Torta**: Toggle (0/1)

### 4. Execução da Simulação

1. Ajuste todos os controles conforme desejado
2. Clique em **"🚀 Simular Cenário"**
3. Visualize os resultados no painel direito

## 📊 Resultados da Simulação

### Métricas Principais
- **TCH Base (predito)**: Predição usando apenas o baseline
- **TCH Final**: Predição com os overrides aplicados
- **Delta**: Diferença entre Final e Base (colorido: verde=positivo, vermelho=negativo)

### Intervalos de Confiança
Se o modelo suportar, exibe intervalo de 95% para as predições.

### Análise de Cascata (Waterfall)
- **Tabela de Impactos**: Lista todos os fatores e seus impactos em TCH
- **Fatores Fixos**: Valores pré-definidos (FACTORS_BASE)
- **Impactos do Modelo**: Calculados marginalmente para variáveis controladas

### Gráfico Waterfall
- Visualização gráfica dos impactos (exceto visão TALHAO)
- Barras coloridas: verde=positivo, vermelho=negativo
- Linha cumulativa mostrando efeito total

### Export para Talhão
Na visão **TALHAO**:
- Gera arquivo Excel com dados detalhados do talhão
- Inclui: dados originais, predições, baseline, overrides e impactos
- Botão de download automático após simulação

## 🔧 Como Funciona o Baseline + Overrides

### Estratégia de Baseline

1. **Seleção de Visão**: Usuário escolhe dimensão (AMBIENTE/FAZENDA/UNIDADE/TALHAO)
2. **Filtro**: Aplica filtro pelo valor selecionado
3. **Cálculo**: Usa **mediana** de todas as features do subconjunto filtrado
4. **Fallback**: Se poucos dados ou NaNs:
   - Primeiro: fallback para UNIDADE
   - Segundo: fallback global (todo dataset)

### Ranges dos Sliders

- Baseados em percentis do baseline (P05-P95)
- Adicionado buffer de 20% para flexibilidade
- Valores clampados para evitar extremos

### Variáveis Não Controladas

- **Satélite**: EVI, GNDVI, NDWI, SAVI permanecem na mediana do baseline
- **Outras**: Todas as features não expostas na UI ficam no baseline

### Aplicação de Overrides

- Apenas variáveis controladas são modificadas
- Demais features permanecem no valor baseline
- Predição final usa `bundle.feature_columns` para alinhamento

## 🧮 Lógica do Cascade/Waterfall

### Fatores Fixos (FACTORS_BASE)

```python
FACTORS_BASE = {
    "IMPUREZA VEGETAL": 2.57,
    "DELTA ÁREA VINHAÇA ASPERSÃO": 0.11,
    "DELTA ÁREA ADUBAÇÃO FOLIAR": 2.54,
    # ... outros fatores
}
```

### Impactos Marginais do Modelo

Para cada variável controlada:
```
impacto_var = predição(override_individual) - TCH_BASE
```

### Combinação Final

```
TCH_FINAL = TCH_BASE + soma(FACTORS_BASE) + soma(impactos_marginais)
```

## 🛠️ Desenvolvimento e Personalização

### Modificar Fatores Fixos

Edite o dicionário `FACTORS_BASE` no início do arquivo `app.py`

### Adicionar Novos Controles

1. Adicione à lista `CONTROLLED_VARS`
2. Inclua na lista `MODEL_FEATURES` se necessário
3. Adicione controle na UI apropriada
4. Atualize lógica de aplicação de overrides

### Modificar Visões

Edite a lista `VISIONS` e o dicionário `DEFAULT_COLUMN_MAPPING`

## 📈 Performance e Otimização

- **Caching**: Usa `st.cache_resource` para modelo e `st.cache_data` para dados
- **Pré-computação**: Baselines calculados uma vez por sessão
- **Fallback Robusto**: Lida com dados faltantes ou insuficientes
- **Validação**: Verifica presença de colunas e tipos de dados

## 🚨 Troubleshooting

### Erro: "Arquivo do modelo não encontrado"
- Verifique se `tch_rf_bundle.joblib` existe no diretório
- Arquivo deve conter `bundle.pipeline` e `bundle.feature_columns`

### Erro: "Dados baseline não encontrados"
- Faça upload via interface ou coloque arquivo `baseline_data.parquet`/`baseline_data.csv`
- Verifique se contém as colunas necessárias

### Sliders não aparecem ou ranges errados
- Verifique se baseline foi calculado corretamente
- Pode haver poucos dados na seleção atual (fallback automático)

### Predições retornam NaN
- Verifique alinhamento das features com `bundle.feature_columns`
- Dados de entrada podem ter valores extremos

## 📝 Logs e Debug

O aplicativo registra informações sobre:
- Carregamento de arquivos
- Cálculo de baselines
- Aplicação de fallbacks
- Execução de predições

Verifique o terminal/console para mensagens de debug.

## 🤝 Suporte

Para questões ou problemas:
1. Verifique os arquivos de dados estão no formato correto
2. Confirme que todas as dependências estão instaladas
3. Execute com `streamlit run app.py --logger.level=debug` para mais detalhes

---

**Versão**: 1.0.0
**Última atualização**: Janeiro 2026