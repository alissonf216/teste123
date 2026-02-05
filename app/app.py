"""
Simulador de Cenários - TCH Prediction App
Aplicativo Streamlit para simulação de cenários de produção de cana-de-açúcar
usando modelo RandomForest treinado.

Regras implementadas (conforme alinhado):
1) TODOS os fatores variam apenas no intervalo de multiplicador [0.5, 1.5] em relação ao FACTORS_BASE.
2) Fatores "apenas negativos" (penalizadores): valor final nunca pode ser > 0 (pode ser 0).
3) Fatores "apenas positivos" (ações): valor final nunca pode ser < 0 (pode ser 0).
4) Removido INDETERMINADO e removido bucket OUTROS do waterfall.
5) Labels do waterfall padronizados e simplificados.
6) Correção do KPI de gap: agora é (Cascade - Modelo).
"""

from pathlib import Path
from io import BytesIO
import warnings
import re

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================
st.set_page_config(
    page_title="Simulador de Cenários - TCH",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CLASSE DUMMY
# =========================
class ModelBundle:
    """Classe dummy para compatibilidade com arquivos joblib que contêm ModelBundle"""

    def __init__(self, pipeline=None, feature_columns=None):
        self.pipeline = pipeline
        self.feature_columns = feature_columns


# =========================
# CSS
# =========================
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    * { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
    .stApp { background-color: #FCFCFC; }
    .main .block-container {
        padding-top: 1.5rem; padding-bottom: 1.5rem;
        padding-left: 2rem; padding-right: 2rem;
        max-width: 1600px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stButton > button {
        background-color: #124141; color: #FFFFFF; border: none; border-radius: 14px;
        padding: 0.875rem 1.5rem; font-size: 1rem; font-weight: 600;
        width: 100%; transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(18, 65, 65, 0.2);
    }
    .stButton > button:hover {
        background-color: #0D2F2F; box-shadow: 0 4px 12px rgba(18, 65, 65, 0.3);
        transform: translateY(-1px);
    }

    div[data-baseweb="slider"] > div > div { background-color: #E5E7EB; }
    div[data-baseweb="slider"] > div > div > div { background-color: #124141; }
    div[data-baseweb="slider"] button {
        background-color: #124141 !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(18, 65, 65, 0.3) !important;
    }
    div[data-baseweb="slider"] button:hover {
        background-color: #0D2F2F !important;
        box-shadow: 0 4px 12px rgba(18, 65, 65, 0.4) !important;
    }

    .stSelectbox > div > div {
        background-color: #FFFFFF; border: 1px solid #DDE6EA; border-radius: 8px;
    }
    .stCheckbox > label { font-weight: 500; color: #374151; }

    label { font-weight: 500; color: #374151; font-size: 0.875rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# NORMALIZAÇÃO E REGRAS DE NEGÓCIO (FATORES)
# =========================
def _canon_key(s: str) -> str:
    """Normaliza chaves (evita mismatch por espaços duplicados, caixa, etc.)."""
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.upper()


# Clamp universal: todos os fatores variam no máximo 50% para cada lado
FACTOR_MULT_RANGE = (0.1, 1.1)

# Penalizadores (apenas negativos): valor final <= 0 (pode ser 0)
NEGATIVE_ONLY_FACTORS = {
    _canon_key("IMPUREZA VEGETAL"),
    _canon_key("NCM"),
    _canon_key("FALHAS PLANTIO"),
    _canon_key("FALHAS SOCA (ESTIMADO)"),
    _canon_key("DANINHAS PLANTIO"),
    _canon_key("DANINHAS SOCA (ESTIMADO)"),
    _canon_key("PISOTEIO"),
    _canon_key("PERDAS COLHEITA"),
    _canon_key("BROCA"),
    _canon_key("SPHENOPHORUS"),
    _canon_key("SÍNDROME MURCHA"),
    _canon_key("FOGO ACIDENTAL"),
    # Conservador: também apenas negativos (base negativo no seu dicionário)
    _canon_key("CP - DELTA PROPORÇÃO IDEAL POR ÉPOCA"),
    _canon_key("CP - APTIDÃO AMBIENTE"),
    _canon_key("CANA DE DEZEMBRO"),
}

# Ações (apenas positivas): valor final >= 0 (pode ser 0)
POSITIVE_ONLY_FACTORS = {
    _canon_key("DELTA ÁREA VINHAÇA ASPERSÃO"),
    _canon_key("DELTA ÁREA ADUBAÇÃO FOLIAR"),
    _canon_key("DELTA ÁREA TORTA/M.O. PLANTIO"),
}

# Labels simples (fatores)
FACTOR_LABELS = {
    _canon_key("IMPUREZA VEGETAL"): "Impureza vegetal",
    _canon_key("DELTA ÁREA VINHAÇA ASPERSÃO"): "Área com vinhaça",
    _canon_key("DELTA ÁREA ADUBAÇÃO FOLIAR"): "Área com adubação foliar",
    _canon_key("DELTA ÁREA TORTA/M.O. PLANTIO"): "Área com torta/MO (plantio)",
    _canon_key("CP - DELTA PROPORÇÃO IDEAL POR ÉPOCA"): "Aderência à época (CP)",
    _canon_key("CP - APTIDÃO AMBIENTE"): "Aptidão do ambiente (CP)",
    _canon_key("NCM"): "NCM",
    _canon_key("FALHAS PLANTIO"): "Falhas no plantio",
    _canon_key("FALHAS SOCA (ESTIMADO)"): "Falhas na soca",
    _canon_key("DANINHAS PLANTIO"): "Daninhas (plantio)",
    _canon_key("DANINHAS SOCA (ESTIMADO)"): "Daninhas (soca)",
    _canon_key("PISOTEIO"): "Pisoteio",
    _canon_key("PERDAS COLHEITA"): "Perdas na colheita",
    _canon_key("BROCA"): "Broca",
    _canon_key("SPHENOPHORUS"): "Sphenophorus",
    _canon_key("SÍNDROME MURCHA"): "Síndrome da murcha",
    _canon_key("CANA DE DEZEMBRO"): "Cana de dezembro",
    _canon_key("FOGO ACIDENTAL"): "Fogo acidental",
}

# Labels simples (features do modelo)
MODEL_FEATURE_LABELS = {
    "rainfall": "Chuva",
    "prev_rainfall": "Chuva anterior",
    "gdd": "GDD",
    "MES_PLANTIO": "Mês do plantio",
    "QTDE_MESES": "Idade (meses)",
    "NDVI": "NDVI",
    "EVI": "EVI",
    "GNDVI": "GNDVI",
    "NDWI": "NDWI",
    "SAVI": "SAVI",
    "TRATOS_CANA_PLANTA": "Tratos",
    "PREPARO_SOLO": "Preparo do solo",
    "IRRIGADO": "Irrigado",
    "Reforma": "Reforma",
    "Vinhaca_E": "Vinhaça (flag)",
    "TORTA": "Torta (flag)",
}


def pretty_factor_label(factor_key: str) -> str:
    k = _canon_key(factor_key)
    return f"Fator: {FACTOR_LABELS.get(k, factor_key.title())}"


def pretty_model_label(feature_name: str) -> str:
    return f"Modelo: {MODEL_FEATURE_LABELS.get(feature_name, feature_name)}"


# =========================
# CONSTANTES (BASE + PONDERAÇÃO)
# =========================
RAW_FACTORS_BASE = {
    # Penalizadores (negativos)
    "IMPUREZA VEGETAL": -2.57,
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

    # Ações (positivas)
    "DELTA ÁREA VINHAÇA ASPERSÃO": 0.11,
    "DELTA ÁREA ADUBAÇÃO FOLIAR": 2.54,
    "DELTA ÁREA TORTA/M.O. PLANTIO": 0.58,

    # Removido: INDETERMINADO
}

FACTORS_BASE = {_canon_key(k): float(v) for k, v in RAW_FACTORS_BASE.items()}

# Definição de pesos (heurística)
RAW_FACTOR_DEF = {
    # Penalizadores
    "IMPUREZA VEGETAL": [("NDVI", 0.60, +1), ("SAVI", 0.40, +1)],

    "CP -  DELTA PROPORÇÃO IDEAL POR ÉPOCA": [("NDWI", 0.70, -1), ("EVI", 0.30, -1)],
    "CP - APTIDÃO AMBIENTE": [("NDWI", 0.70, -1), ("EVI", 0.30, -1)],
    "NCM": [("NDWI", 0.70, -1), ("EVI", 0.30, -1)],

    "FALHAS PLANTIO": [("SAVI", 0.65, -1), ("NDVI", 0.35, -1)],
    "FALHAS SOCA (ESTIMADO)": [("SAVI", 0.60, -1), ("EVI", 0.40, -1)],

    "DANINHAS PLANTIO": [("EVI", 0.45, -1), ("SAVI", 0.35, -1), ("NDVI", 0.20, -1)],
    "DANINHAS SOCA (ESTIMADO)": [("EVI", 0.45, -1), ("SAVI", 0.35, -1), ("NDVI", 0.20, -1)],

    "PISOTEIO": [("EVI", 0.60, -1), ("NDVI", 0.40, -1)],
    "PERDAS COLHEITA": [("EVI", 0.60, -1), ("NDVI", 0.40, -1)],

    "BROCA": [("EVI", 0.70, -1), ("GNDVI", 0.30, -1)],
    "SPHENOPHORUS": [("NDWI", 0.70, -1), ("EVI", 0.30, -1)],
    "SÍNDROME MURCHA": [("NDWI", 0.80, -1), ("EVI", 0.20, -1)],

    "CANA DE DEZEMBRO": [("NDWI", 0.70, -1), ("EVI", 0.30, -1)],
    "FOGO ACIDENTAL": [("NDVI", 0.60, -1), ("EVI", 0.40, -1)],

    # Ações
    "DELTA ÁREA VINHAÇA ASPERSÃO": [("GNDVI", 0.70, +1), ("EVI", 0.30, +1)],
    "DELTA ÁREA ADUBAÇÃO FOLIAR": [("GNDVI", 0.80, +1), ("EVI", 0.20, +1)],
    "DELTA ÁREA TORTA/M.O. PLANTIO": [("EVI", 0.60, +1), ("NDVI", 0.20, +1), ("GNDVI", 0.20, +1)],

    # Removido: INDETERMINADO
}

FACTOR_DEF = {}
for k, defs in RAW_FACTOR_DEF.items():
    kk = _canon_key(k)
    FACTOR_DEF[kk] = [(str(m).upper(), float(w), float(s)) for (m, w, s) in defs]

METRICS = ["EVI", "GNDVI", "NDVI", "NDWI", "SAVI"]


MODEL_FEATURES = [
    "EXPANSAO",
    "Devolucao",
    "Reforma",
    "Vinhaca_E",
    "TORTA",
    "SISTEMA_COL",
    "UNID_IND",
    "IRRIGADO",
    "QTDE_MESES",
    "MES_PLANTIO",
    "area_prod",
    "PLANTIO_MECANIZADO",
    "TRATOS_CANA_PLANTA",
    "PREPARO_SOLO",
    "TRATOS_CANA_PLANTA_PARC",
    "PREPARO_SOLO_PARCERIA",
    "PLANTIO_SEMI_MECANIZADO",
    "CONTROLE_AGRICOLA",
    "GESTAO_DA_QUALIDADE",
    "TOTAL_OPERS",
    "ambiente_correto_bula",
    "tempo_colheita_correto_bula",
    "incendios",
    "EVI",
    "GNDVI",
    "NDVI",
    "NDWI",
    "SAVI",
    "gdd",
    "rainfall",
    "prev_rainfall",
]

CONTROLLED_VARS = [
    "rainfall",
    "prev_rainfall",
    "gdd",
    "MES_PLANTIO",
    "QTDE_MESES",
    "NDVI",
    "EVI",
    "GNDVI",
    "NDWI",
    "SAVI",
    "TRATOS_CANA_PLANTA",
    "PREPARO_SOLO",
    "IRRIGADO",
    "Reforma",
    "Vinhaca_E",
    "TORTA",
]

DEFAULT_COLUMN_MAPPING = {
    "AMBIENTE": ["AMBIENTE", "ambiente", "AMB"],
    "FAZENDA": ["FAZENDA", "CD_FAZENDA", "fazenda", "FAZ"],
    "UNIDADE": ["UNIDADE", "UNID_IND", "unidade", "UNID"],
    "TALHAO": ["TALHAO", "COD", "talhao", "TAL"],
}

VISIONS = ["AMBIENTE", "FAZENDA", "UNIDADE", "TALHAO"]


# =========================
# FUNÇÕES UTILITÁRIAS
# =========================
def _to_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_minmax(mn: float, mx: float, value: float, min_range: float) -> tuple[float, float, float]:
    mn = _to_float(mn, value - min_range / 2)
    mx = _to_float(mx, value + min_range / 2)
    value = _to_float(value, (mn + mx) / 2)

    if mx <= mn:
        mn = value - min_range / 2
        mx = value + min_range / 2

    value = min(max(value, mn), mx)
    return mn, mx, value


def _clamp(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


# =========================
# FUNÇÕES (MODELO / BASELINES)
# =========================
def load_model():
    model_path = Path("tch_rf_bundle.joblib")
    if not model_path.exists():
        st.error(f"Arquivo do modelo não encontrado: {model_path}")
        st.info("Coloque o arquivo 'tch_rf_bundle.joblib' no diretório do projeto.")
        return None

    try:
        bundle = joblib.load(model_path)

        if hasattr(bundle, "pipeline"):
            if hasattr(bundle, "feature_columns") and bundle.feature_columns is not None:
                return bundle
            if hasattr(bundle.pipeline, "feature_names_in_"):
                bundle.feature_columns = list(bundle.pipeline.feature_names_in_)
                return bundle
            st.warning("Não foi possível encontrar 'feature_columns'. Usando features padrão.")
            bundle.feature_columns = MODEL_FEATURES
            return bundle

        if hasattr(bundle, "predict"):
            bundle_obj = ModelBundle(pipeline=bundle)
            if hasattr(bundle, "feature_names_in_"):
                bundle_obj.feature_columns = list(bundle.feature_names_in_)
            else:
                bundle_obj.feature_columns = MODEL_FEATURES
            return bundle_obj

        st.error("O arquivo do modelo não contém um pipeline válido.")
        return None

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        return None


@st.cache_data
def load_baseline_data():
    parquet_path = Path("baseline_data.parquet")
    csv_path = Path("baseline_data.csv")

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return df, "baseline_data.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:")]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        df = df.dropna(axis=1, how="all")
        return df, "baseline_data.csv"

    return None, None


def handle_baseline_upload():
    st.warning("Arquivo de dados baseline não encontrado.")
    st.info("Faça upload do arquivo baseline (formato parquet ou csv) para usar o simulador.")

    uploaded_file = st.file_uploader(
        "Upload do arquivo baseline",
        type=["parquet", "csv"],
        help="Selecione um arquivo .parquet ou .csv com os dados baseline",
    )

    if uploaded_file is None:
        st.info("Aguardando upload do arquivo baseline...")
        return None

    try:
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
            file_path = "baseline_data.parquet"
            df.to_parquet(file_path)
        else:
            df = pd.read_csv(uploaded_file)
            unnamed_cols = [col for col in df.columns if col.startswith("Unnamed:")]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            df = df.dropna(axis=1, how="all")
            file_path = "baseline_data.csv"
            df.to_csv(file_path, index=False)

        st.success(f"Arquivo '{uploaded_file.name}' salvo como '{file_path}'.")
        st.info(f"Dataset carregado: {len(df)} linhas × {len(df.columns)} colunas.")
        st.cache_data.clear()
        return df

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
        return None


def get_column_mapping(df: pd.DataFrame):
    mapping = {}
    df_columns = df.columns.tolist()

    for group_name, possible_names in DEFAULT_COLUMN_MAPPING.items():
        found = None
        for name in possible_names:
            if name in df_columns:
                found = name
                break
        mapping[group_name] = found

    return mapping


def precompute_baselines(df: pd.DataFrame, column_mapping: dict):
    baselines = {}

    default_values = {
        "rainfall": 1000,
        "prev_rainfall": 900,
        "gdd": 2000,
        "MES_PLANTIO": 8,
        "QTDE_MESES": 12,
        "NDVI": 0.60,
        "EVI": 0.45,
        "GNDVI": 0.55,
        "NDWI": 0.10,
        "SAVI": 0.55,
        "TRATOS_CANA_PLANTA": 2.0,
        "PREPARO_SOLO": 1,
        "IRRIGADO": 0,
        "Reforma": 0,
        "Vinhaca_E": 0,
        "TORTA": 0,
    }
    default_series = pd.Series(default_values)

    if df.empty or len(df) < 5:
        for vision in VISIONS:
            baselines[f"{vision}_median"] = default_series
            baselines[f"{vision}_p05"] = default_series * 0.8
            baselines[f"{vision}_p95"] = default_series * 1.2
        baselines["global_median"] = default_series
        baselines["global_p05"] = default_series * 0.8
        baselines["global_p95"] = default_series * 1.2
        return baselines

    available_features = [f for f in MODEL_FEATURES if f in df.columns]
    if not available_features:
        for vision in VISIONS:
            baselines[f"{vision}_median"] = default_series
            baselines[f"{vision}_p05"] = default_series * 0.8
            baselines[f"{vision}_p95"] = default_series * 1.2
        baselines["global_median"] = default_series
        baselines["global_p05"] = default_series * 0.8
        baselines["global_p95"] = default_series * 1.2
        return baselines

    for vision in VISIONS:
        col_name = column_mapping.get(vision)
        if col_name and col_name in df.columns:
            try:
                grouped = df.groupby(col_name)[available_features]
                baselines[f"{vision}_median"] = grouped.median()
                baselines[f"{vision}_p05"] = grouped.quantile(0.05)
                baselines[f"{vision}_p95"] = grouped.quantile(0.95)
            except Exception:
                pass

    try:
        baselines["global_median"] = df[available_features].median()
        baselines["global_p05"] = df[available_features].quantile(0.05)
        baselines["global_p95"] = df[available_features].quantile(0.95)
    except Exception:
        baselines["global_median"] = default_series
        baselines["global_p05"] = default_series * 0.8
        baselines["global_p95"] = default_series * 1.2

    return baselines


def _ensure_controlled_keys(baseline: pd.Series, p05: pd.Series, p95: pd.Series):
    defaults = {
        "rainfall": 1000,
        "prev_rainfall": 900,
        "gdd": 2000,
        "MES_PLANTIO": 8,
        "QTDE_MESES": 12,
        "NDVI": 0.60,
        "EVI": 0.45,
        "GNDVI": 0.55,
        "NDWI": 0.10,
        "SAVI": 0.55,
        "TRATOS_CANA_PLANTA": 2.0,
        "PREPARO_SOLO": 1,
        "IRRIGADO": 0,
        "Reforma": 0,
        "Vinhaca_E": 0,
        "TORTA": 0,
    }

    for k, v in defaults.items():
        if k not in baseline.index or pd.isna(baseline.get(k, np.nan)):
            baseline.loc[k] = v
        if k not in p05.index or pd.isna(p05.get(k, np.nan)):
            p05.loc[k] = v * 0.8 if isinstance(v, (int, float)) else v
        if k not in p95.index or pd.isna(p95.get(k, np.nan)):
            p95.loc[k] = v * 1.2 if isinstance(v, (int, float)) else v

    return baseline, p05, p95


def get_baseline_values(baselines: dict, vision: str, selected_value, column_mapping: dict):
    default_baseline = pd.Series(
        {
            "rainfall": 1000,
            "prev_rainfall": 900,
            "gdd": 2000,
            "MES_PLANTIO": 8,
            "QTDE_MESES": 12,
            "NDVI": 0.60,
            "EVI": 0.45,
            "GNDVI": 0.55,
            "NDWI": 0.10,
            "SAVI": 0.55,
            "TRATOS_CANA_PLANTA": 2.0,
            "PREPARO_SOLO": 1,
            "IRRIGADO": 0,
            "Reforma": 0,
            "Vinhaca_E": 0,
            "TORTA": 0,
        }
    )
    default_p05 = default_baseline * 0.8
    default_p95 = default_baseline * 1.2

    if not baselines or "global_median" not in baselines:
        return default_baseline, default_p05, default_p95

    vision_col = column_mapping.get(vision)
    if vision_col and selected_value is not None:
        try:
            baseline = baselines[f"{vision}_median"].loc[selected_value]
            p05 = baselines[f"{vision}_p05"].loc[selected_value]
            p95 = baselines[f"{vision}_p95"].loc[selected_value]
            baseline, p05, p95 = _ensure_controlled_keys(baseline, p05, p95)
            return baseline, p05, p95
        except Exception:
            pass

    baseline = baselines["global_median"]
    p05 = baselines["global_p05"]
    p95 = baselines["global_p95"]

    if isinstance(baseline, pd.Series):
        baseline, p05, p95 = _ensure_controlled_keys(baseline.copy(), p05.copy(), p95.copy())
        return baseline, p05, p95

    return default_baseline, default_p05, default_p95


def _prepare_features_for_model(bundle, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o dataframe tenha TODAS as colunas do modelo e na ordem correta.
    """
    if not hasattr(bundle, "feature_columns") or bundle.feature_columns is None:
        return features_df

    cols = list(bundle.feature_columns)
    out = features_df.copy()

    for c in cols:
        if c not in out.columns:
            out[c] = 0

    out = out[cols]
    return out


def predict_tch(bundle, features_df: pd.DataFrame):
    if bundle is None:
        raise ValueError("Bundle do modelo não foi carregado.")
    if not hasattr(bundle, "pipeline") or bundle.pipeline is None:
        raise ValueError("Bundle não contém pipeline válido.")

    try:
        X = _prepare_features_for_model(bundle, features_df)
        pred = bundle.pipeline.predict(X)
        return float(pred[0]), None
    except Exception as e:
        st.error(f"Erro na predição: {e}")
        return None, None


def calculate_cascade_impacts(bundle, baseline_features: pd.DataFrame, controlled_changes: dict):
    """
    Impacto marginal do modelo: muda 1 variável por vez (ceteris paribus).
    Retorna deltas (test_pred - base_prediction).
    """
    impacts = {}
    base_prediction, _ = predict_tch(bundle, baseline_features)
    if base_prediction is None:
        st.error("Não foi possível calcular a predição base para impactos.")
        return {}

    for var, new_value in controlled_changes.items():
        if var in baseline_features.columns:
            test_features = baseline_features.copy()
            test_features.loc[:, var] = new_value
            test_pred, _ = predict_tch(bundle, test_features)
            if test_pred is not None:
                impacts[var] = test_pred - base_prediction

    return impacts


def _apply_factor_sign_rules(factor_key: str, base_value: float, mult: float) -> tuple[float, float]:
    """
    Aplica:
    - clamp universal de mult em [0.5, 1.5]
    - regras de sinal por tipo (apenas negativo / apenas positivo)
    Retorna: (final_value, delta)
    """
    k = _canon_key(factor_key)
    lo, hi = FACTOR_MULT_RANGE
    mult = float(_clamp(mult, lo, hi))

    base_value = float(base_value)
    final_value = base_value * mult
    delta = final_value - base_value

    # Penalizador: nunca pode ficar > 0 (pode ser 0)
    if k in NEGATIVE_ONLY_FACTORS:
        final_value = min(final_value, 0.0)
        delta = final_value - base_value

    # Ação: nunca pode ficar < 0 (pode ser 0)
    if k in POSITIVE_ONLY_FACTORS:
        final_value = max(final_value, 0.0)
        delta = final_value - base_value

    return float(final_value), float(delta)


def calculate_weighted_factors(
    baseline_features: pd.DataFrame,
    final_features: pd.DataFrame,
    factors_base: dict,
    factor_def: dict,
    metrics: list,
) -> tuple[dict, dict]:
    """
    Calcula:
      - factor_values: valor final do fator (base_value * mult com clamp [0.5,1.5])
      - factor_deltas: delta do fator (final - base)
    """
    b = baseline_features.iloc[0]
    f = final_features.iloc[0]

    deltas = {}
    for m in metrics:
        m_u = str(m).upper()
        b_val = float(b.get(m_u, b.get(m, 0.0))) if pd.notna(b.get(m_u, b.get(m, np.nan))) else 0.0
        f_val = float(f.get(m_u, f.get(m, 0.0))) if pd.notna(f.get(m_u, f.get(m, np.nan))) else 0.0
        if not np.isfinite(b_val):
            b_val = 0.0
        if not np.isfinite(f_val):
            f_val = 0.0
        deltas[m_u] = f_val - b_val

    factor_values = {}
    factor_deltas = {}

    for factor_name, base_value in factors_base.items():
        key = _canon_key(factor_name)
        base_value = float(base_value)

        # Sem definição: mult = 1 (mantém base)
        if key not in factor_def:
            final_val, delta = _apply_factor_sign_rules(key, base_value, mult=1.0)
            factor_values[key] = final_val
            factor_deltas[key] = delta
            continue

        weighted_delta = 0.0
        for metric, weight, sign in factor_def[key]:
            metric_u = str(metric).upper()
            if metric_u not in deltas:
                continue
            weighted_delta += float(weight) * float(sign) * float(deltas[metric_u])

        mult = 1.0 + float(weighted_delta)

        final_val, delta = _apply_factor_sign_rules(key, base_value, mult=mult)
        factor_values[key] = final_val
        factor_deltas[key] = delta

    return factor_values, factor_deltas


def create_waterfall_chart_pretty(
    impacts_dict: dict,
    base_value: float,
    final_value: float,
    title: str = "Waterfall - Análise Cascade",
    max_items: int = 18,
    base_label: str = "Base",
    final_label: str = "Final",
):
    """
    Waterfall horizontal:
    - Mostra Top N impactos (por |impacto|)
    - NÃO cria "OUTROS" (removido conforme requisito)
    """
    from matplotlib.ticker import FuncFormatter

    items = [(str(k), float(v)) for k, v in impacts_dict.items() if v is not None and np.isfinite(float(v))]
    items = [(k, v) for k, v in items if abs(v) > 1e-9]

    # Truncar sem "OUTROS"
    if len(items) > max_items:
        items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)
        items = items_sorted[:max_items]

    labels = [base_label] + [k for k, _ in items] + [final_label]
    y = np.arange(len(labels))
    bar_height = 0.62

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(labels) + 2)))

    color_base = "#6B7280"
    color_pos = "#2E7D32"
    color_neg = "#C62828"
    color_final = "#1E88E5"

    ax.barh(y[0], base_value, left=0, height=bar_height, color=color_base, edgecolor="white", linewidth=1.0)

    prev = base_value
    for i, (_, v) in enumerate(items, start=1):
        left = prev if v >= 0 else prev + v
        width = abs(v)

        ax.barh(
            y[i],
            width,
            left=left,
            height=bar_height,
            color=color_pos if v >= 0 else color_neg,
            edgecolor="white",
            linewidth=1.0,
        )

        ax.plot([prev, prev], [y[i] - bar_height / 2, y[i] + bar_height / 2], color="#9CA3AF", linewidth=1, alpha=0.6)

        txt = f"{v:+.3f}"
        x_txt = left + width + max(1.0, (abs(final_value) + abs(base_value)) * 0.006)
        ax.text(x_txt, y[i], txt, va="center", ha="left", fontsize=9, color="#111827")

        prev = prev + v

    ax.barh(y[-1], final_value, left=0, height=bar_height, color=color_final, edgecolor="white", linewidth=1.0)

    x_pad = max(1.0, (abs(final_value) + abs(base_value)) * 0.006)
    ax.text(base_value + x_pad, y[0], f"{base_value:.2f}", va="center", ha="left", fontsize=9, color="#111827", fontweight="bold")
    ax.text(final_value + x_pad, y[-1], f"{final_value:.2f}", va="center", ha="left", fontsize=9, color="#111827", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.2f}"))
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Impacto acumulado (t/ha)")

    cumul_vals = [base_value]
    for _, v in items:
        cumul_vals.append(cumul_vals[-1] + v)
    all_vals = [0, base_value, final_value] + cumul_vals
    x_min = min(all_vals)
    x_max = max(all_vals)
    pad = max(1.0, (x_max - x_min) * 0.08)
    ax.set_xlim(x_min - pad, x_max + pad)

    plt.tight_layout()
    return fig


# =========================
# ESTADO INICIAL
# =========================
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "selected_vision" not in st.session_state:
    st.session_state.selected_vision = "AMBIENTE"
if "selected_value" not in st.session_state:
    st.session_state.selected_value = None


# =========================
# CARREGA MODELO
# =========================
bundle = load_model()
if bundle is None:
    st.error("Modelo não carregado.")
    st.info("Você precisa do arquivo `tch_rf_bundle.joblib` contendo:")
    st.code("- bundle.pipeline: modelo treinado\n- bundle.feature_columns: lista das features do modelo")
    st.stop()


# =========================
# CARREGA BASELINE
# =========================
df_baseline, baseline_file = load_baseline_data()

if df_baseline is None:
    st.warning("Dados baseline não encontrados.")
    df_baseline = handle_baseline_upload()
    if df_baseline is None:
        st.info("Modo limitado: o simulador funcionará com valores padrão.")
        df_baseline = pd.DataFrame(columns=MODEL_FEATURES)

column_mapping = get_column_mapping(df_baseline)


# =========================
# HEADER
# =========================
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image("logo_atvos.png", width=120)

with col_title:
    st.markdown(
        """
<div style="
    background: linear-gradient(135deg, #124141 0%, #0D2F2F 100%);
    color: white;
    padding: 2rem 0;
    margin: -1rem -1rem 2rem -1rem;
    border-radius: 0 0 20px 20px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(18, 65, 65, 0.3);
">
    <h1 style="
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    ">🌾 Simulador de Cenários TCH</h1>
    <p style="
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-weight: 400;
    ">Simulação inteligente de produção de cana-de-açúcar com IA</p>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================
# SELEÇÃO VISÃO / VALOR
# =========================
st.markdown(
    """
<div style="margin-bottom: 2rem;">
    <h2 style="
        color: #124141;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #E3ECF2;
        padding-bottom: 0.5rem;
    ">🎛️ Configurações da Simulação</h2>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.session_state.selected_vision = st.selectbox(
        "Selecionar visão",
        VISIONS,
        index=VISIONS.index(st.session_state.selected_vision) if st.session_state.selected_vision in VISIONS else 0,
        help="Escolha a dimensão para agrupar os dados baseline",
    )

with col2:
    selected_vision = st.session_state.selected_vision
    vision_col = column_mapping.get(selected_vision)

    if vision_col and vision_col in df_baseline.columns and not df_baseline.empty:
        unique_values = sorted(df_baseline[vision_col].dropna().unique())
        if unique_values:
            if st.session_state.selected_value not in unique_values:
                st.session_state.selected_value = unique_values[0]
            st.session_state.selected_value = st.selectbox(
                f"Selecionar {selected_vision.lower()}",
                unique_values,
                index=unique_values.index(st.session_state.selected_value) if st.session_state.selected_value in unique_values else 0,
                help=f"Escolha o valor de {selected_vision} para calcular baseline",
            )
        else:
            st.session_state.selected_value = None
            st.warning("Sem valores disponíveis para esta visão.")
    else:
        st.session_state.selected_value = None
        st.warning(f"Coluna para {selected_vision} não encontrada ou dataset vazio.")

with col3:
    if st.button("🔄 Recalcular Baseline", use_container_width=True):
        st.session_state.sim_result = None
        st.rerun()


# =========================
# BASELINES
# =========================
selected_vision = st.session_state.selected_vision
selected_value = st.session_state.selected_value

baselines = precompute_baselines(df_baseline, column_mapping)
baseline, p05, p95 = get_baseline_values(baselines, selected_vision, selected_value, column_mapping)
if not isinstance(baseline, pd.Series):
    baseline = pd.Series(baseline)


# =========================
# VARIÁVEIS DE CONTROLE
# =========================
st.markdown(
    """
<div style="margin: 2rem 0 1rem 0;">
    <h3 style="
        color: #124141;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    ">⚙️ Variáveis de Controle</h3>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🌦️ Clima & Energia", "📅 Calendário", "🌿 Índices & Manejo"])

with tab1:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🌧️ Precipitação**")
        rainfall_min, rainfall_max, rainfall_val = _safe_minmax(
            mn=_to_float(p05.get("rainfall", 0.0), 0.0) * 0.8,
            mx=_to_float(p95.get("rainfall", 1000.0), 1000.0) * 1.2,
            value=_to_float(baseline.get("rainfall", 1000.0), 1000.0),
            min_range=10.0,
        )
        rainfall = st.slider(
            "Chuva acumulada (mm)",
            min_value=float(rainfall_min),
            max_value=float(rainfall_max),
            value=float(rainfall_val),
            step=10.0,
            format="%.0f",
            help="Precipitação acumulada em milímetros",
        )

    with col_b:
        st.markdown("**💧 Chuva Antecedente**")
        pr_min, pr_max, pr_val = _safe_minmax(
            mn=_to_float(p05.get("prev_rainfall", 0.0), 0.0) * 0.8,
            mx=_to_float(p95.get("prev_rainfall", 900.0), 900.0) * 1.2,
            value=_to_float(baseline.get("prev_rainfall", 900.0), 900.0),
            min_range=10.0,
        )
        prev_rainfall = st.slider(
            "Chuva antecedente (mm)",
            min_value=float(pr_min),
            max_value=float(pr_max),
            value=float(pr_val),
            step=10.0,
            format="%.0f",
            help="Precipitação anterior em milímetros",
        )

    with col_c:
        st.markdown("**🌡️ Energia Térmica**")
        gdd_min, gdd_max, gdd_val = _safe_minmax(
            mn=_to_float(p05.get("gdd", 0.0), 0.0) * 0.8,
            mx=_to_float(p95.get("gdd", 2000.0), 2000.0) * 1.2,
            value=_to_float(baseline.get("gdd", 2000.0), 2000.0),
            min_range=50.0,
        )
        gdd = st.slider(
            "GDD acumulados",
            min_value=float(gdd_min),
            max_value=float(gdd_max),
            value=float(gdd_val),
            step=50.0,
            format="%.0f",
            help="Graus-dia de crescimento acumulados",
        )

with tab2:
    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**📆 Plantio**")
        mes_plantio_default = int(_to_float(baseline.get("MES_PLANTIO", 8), 8))
        mes_plantio_default = min(max(mes_plantio_default, 1), 12)

        mes_plantio = st.selectbox(
            "Mês do plantio",
            options=list(range(1, 13)),
            index=mes_plantio_default - 1,
            format_func=lambda x: pd.to_datetime(str(x), format="%m").strftime("%B").capitalize(),
            help="Mês do plantio (1-12)",
        )

    with col_e:
        st.markdown("**⏱️ Ciclo**")
        qt_min = int(max(1, _to_float(p05.get("QTDE_MESES", 12), 12) * 0.8))
        qt_max = int(max(qt_min + 1, _to_float(p95.get("QTDE_MESES", 12), 12) * 1.2))
        qt_val = int(_to_float(baseline.get("QTDE_MESES", 12), 12))
        qt_val = min(max(qt_val, qt_min), qt_max)

        qtde_meses = st.slider(
            "Idade da cultura",
            min_value=int(qt_min),
            max_value=int(qt_max),
            value=int(qt_val),
            format="%d meses",
            help="Duração do ciclo em meses",
        )

with tab3:
    col_f, col_g, col_h = st.columns(3)

    def _metric_slider(name: str, baseline_s: pd.Series, p05_s: pd.Series, p95_s: pd.Series):
        """
        Slider robusto para índices:
        - aceita valores negativos (ex.: NDWI)
        - garante min < max
        - clampa em [-1, 1]
        """
        name_u = name.upper()

        base_v = _to_float(baseline_s.get(name_u, baseline_s.get(name, 0.0)), 0.0)
        p05_v = _to_float(p05_s.get(name_u, p05_s.get(name, np.nan)), np.nan)
        p95_v = _to_float(p95_s.get(name_u, p95_s.get(name, np.nan)), np.nan)

        if not np.isfinite(p05_v):
            p05_v = base_v - 0.15
        if not np.isfinite(p95_v):
            p95_v = base_v + 0.15

        base_v = _clamp(base_v, -1.0, 1.0)
        p05_v = _clamp(p05_v, -1.0, 1.0)
        p95_v = _clamp(p95_v, -1.0, 1.0)

        mn = min(p05_v, base_v - 0.10)
        mx = max(p95_v, base_v + 0.10)

        mn, mx, base_v = _safe_minmax(mn, mx, base_v, min_range=0.02)

        return st.slider(
            name_u,
            min_value=float(mn),
            max_value=float(mx),
            value=float(base_v),
            step=0.01,
            format="%.2f",
        )

    # Valores dos índices apenas "por trás", sem mostrar no grid
    ndvi = float(baseline.get("NDVI", 0.0))
    evi = float(baseline.get("EVI", 0.0))
    gndvi = float(baseline.get("GNDVI", 0.0))
    ndwi = float(baseline.get("NDWI", 0.0))
    savi = float(baseline.get("SAVI", 0.0))

    with col_f:
        st.markdown("**🚜 Manejo**")
        tratos_base = _to_float(baseline.get("TRATOS_CANA_PLANTA", 2.0), 2.0)
        tratos_pct = st.slider(
            "Tratos (% vs baseline)",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=1.0,
            format="%.0f%%",
            help=f"Percentual de variação vs baseline ({tratos_base:.1f})",
        )
        tratos_cana_planta = tratos_base * (1 + tratos_pct / 100)
        preparo_solo = st.checkbox("Preparo do Solo", value=bool(int(_to_float(baseline.get("PREPARO_SOLO", 1), 1))))

    with col_g:
        irrigado = st.checkbox("Sistema Irrigado", value=bool(int(_to_float(baseline.get("IRRIGADO", 0), 0))))
        reforma = st.checkbox("Reforma", value=bool(int(_to_float(baseline.get("Reforma", 0), 0))))

    with col_h:
        vinhaca_e = st.checkbox("Vinhaça", value=bool(int(_to_float(baseline.get("Vinhaca_E", 0), 0))))
        torta = st.checkbox("Torta", value=bool(int(_to_float(baseline.get("TORTA", 0), 0))))


# =========================
# BOTÃO SIMULAR
# =========================
if st.button("🚀 Simular Cenário", key="simulate", help="Executar simulação com os parâmetros atuais", use_container_width=True):
    baseline_df = pd.DataFrame([baseline])

    controlled_changes = {
        "rainfall": rainfall,
        "prev_rainfall": prev_rainfall,
        "gdd": gdd,
        "MES_PLANTIO": mes_plantio,
        "QTDE_MESES": qtde_meses,
        "NDVI": ndvi,
        "EVI": evi,
        "GNDVI": gndvi,
        "NDWI": ndwi,
        "SAVI": savi,
        "TRATOS_CANA_PLANTA": tratos_cana_planta,
        "PREPARO_SOLO": int(preparo_solo),
        "IRRIGADO": int(irrigado),
        "Reforma": int(reforma),
        "Vinhaca_E": int(vinhaca_e),
        "TORTA": int(torta),
    }

    final_features = baseline_df.copy()
    for var, value in controlled_changes.items():
        final_features[var] = value

    # Predições
    tch_base, _ = predict_tch(bundle, baseline_df)
    tch_final_model, _ = predict_tch(bundle, final_features)

    if tch_base is None or tch_final_model is None:
        st.error("Erro na predição. Verifique se o modelo e dados estão corretos.")
        st.stop()

    # Impactos do modelo (deltas marginais)
    model_impacts = calculate_cascade_impacts(bundle, baseline_df, controlled_changes)

    # Fatores ponderados (com clamp [0.5, 1.5] e regras de sinal)
    factor_values, factor_deltas = calculate_weighted_factors(
        baseline_features=baseline_df,
        final_features=final_features,
        factors_base=FACTORS_BASE,
        factor_def=FACTOR_DEF,
        metrics=METRICS,
    )

    # Cascade impacts (deltas) com labels simples
    cascade_impacts = {}

    # Fatores
    # Fatores (usar valor final do fator, não delta)
    for k, v in factor_values.items():
        k_norm = _canon_key(k)
        cascade_impacts[pretty_factor_label(k_norm)] = float(v)


    # Modelo
    for k, v in model_impacts.items():
        cascade_impacts[pretty_model_label(k)] = float(v)

    # TCH via cascade (baseline do modelo + soma de deltas)
    cascade_sum = float(np.nansum([float(v) for v in cascade_impacts.values()])) if cascade_impacts else 0.0
    tch_final_cascade = float(tch_base) + cascade_sum

    delta_model = float(tch_final_model) - float(tch_base)
    delta_cascade = float(tch_final_cascade) - float(tch_base)

    # Gap correto: Cascade vs Modelo
    gap_cascade_vs_model = float(tch_final_cascade) - float(tch_final_model)

    impacts_df = pd.DataFrame(
        {"Fator/Variável": list(cascade_impacts.keys()), "Impacto (t/ha)": list(cascade_impacts.values())}
    ).round(4)

    # Diagnóstico: baseline vs final vs delta
    cols_check = [
        "rainfall", "prev_rainfall", "gdd", "MES_PLANTIO", "QTDE_MESES",
        "NDVI", "EVI", "GNDVI", "NDWI", "SAVI",
        "TRATOS_CANA_PLANTA", "PREPARO_SOLO", "IRRIGADO", "Reforma", "Vinhaca_E", "TORTA"
    ]
    rows = []
    for c in cols_check:
        b = _to_float(baseline_df.iloc[0].get(c, np.nan), np.nan)
        f = _to_float(final_features.iloc[0].get(c, np.nan), np.nan)
        d = (f - b) if (np.isfinite(b) and np.isfinite(f)) else np.nan
        rows.append([c, b, f, d])
    debug_df = pd.DataFrame(rows, columns=["Variável", "Baseline", "Final", "Delta"])

    st.session_state.sim_result = {
        "tch_base": float(tch_base),
        "tch_final_model": float(tch_final_model),
        "tch_final_cascade": float(tch_final_cascade),
        "delta_model": float(delta_model),
        "delta_cascade": float(delta_cascade),
        "gap_cascade_vs_model": float(gap_cascade_vs_model),
        "baseline_df": baseline_df,
        "final_features": final_features,
        "controlled_changes": controlled_changes,
        "cascade_impacts": cascade_impacts,
        "impacts_df": impacts_df,
        "debug_df": debug_df,
        "selected_vision": selected_vision,
        "selected_value": selected_value,
        "column_mapping": column_mapping,
        "factor_values": factor_values,
        "factor_deltas": factor_deltas,
    }


# =========================
# RESULTADOS
# =========================
if st.session_state.sim_result is None:
    st.markdown(
        """
<div style="
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    padding: 3rem;
    margin: 2rem 0;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
    text-align: center;
">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
    <h3 style="color: #124141; margin: 0 0 1rem 0; font-size: 1.5rem; font-weight: 600;">Pronto para Simular!</h3>
    <p style="color: #6B7280; margin: 0; font-size: 1rem;">Configure os parâmetros acima e clique em "Simular Cenário" para ver os resultados</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()

tch_base = st.session_state.sim_result["tch_base"]
tch_final_model = st.session_state.sim_result["tch_final_model"]
tch_final_cascade = st.session_state.sim_result["tch_final_cascade"]
delta_model = st.session_state.sim_result["delta_model"]
delta_cascade = st.session_state.sim_result["delta_cascade"]
gap_cascade_vs_model = st.session_state.sim_result["gap_cascade_vs_model"]

cascade_impacts = st.session_state.sim_result["cascade_impacts"]
impacts_df = st.session_state.sim_result["impacts_df"]
debug_df = st.session_state.sim_result["debug_df"]

selected_vision = st.session_state.sim_result["selected_vision"]

st.markdown(
    """
<div style="
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    padding: 2rem;
    margin: 2rem -1rem;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
">
    <div style="
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #124141;
    ">
        <div style="
            background: #124141;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            margin-right: 1rem;
        ">📊 RESULTADOS</div>
        <h2 style="
            color: #124141;
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
        ">Análise de Impactos TCH (Fatores + Modelo)</h2>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns([1, 1, 1, 1])

with k1:
    st.markdown(
        f"""
    <div style="background: white; padding: 2rem; border-radius: 16px; border: 2px solid #E3ECF2; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1rem;">
        <div style="color: #6B7280; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">TCH Base</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #124141; margin-bottom: 0.5rem;">{tch_base:.1f}</div>
        <div style="color: #6B7280; font-size: 0.75rem;">t/ha</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k2:
    delta_color = "#407A4E" if delta_model >= 0 else "#C24656"
    st.markdown(
        f"""
    <div style="background: white; padding: 2rem; border-radius: 16px; border: 2px solid #E3ECF2; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1rem;">
        <div style="color: #6B7280; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">TCH Final (Modelo)</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #124141; margin-bottom: 0.5rem;">{tch_final_model:.1f}</div>
        <div style="color: {delta_color}; font-size: 1rem; font-weight: 700;">{delta_model:+.1f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k3:
    delta_color = "#407A4E" if delta_cascade >= 0 else "#C24656"
    st.markdown(
        f"""
    <div style="background: white; padding: 2rem; border-radius: 16px; border: 2px solid #E3ECF2; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1rem;">
        <div style="color: #6B7280; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">TCH Final (Cascade)</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #124141; margin-bottom: 0.5rem;">{tch_final_cascade:.1f}</div>
        <div style="color: {delta_color}; font-size: 1rem; font-weight: 700;">{delta_cascade:+.1f}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with k4:
    gap_color = "#407A4E" if gap_cascade_vs_model > 0 else "#C24656"
    st.markdown(
        f"""
    <div style="background: white; padding: 2rem; border-radius: 16px; border: 2px solid #E3ECF2; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 1rem;">
        <div style="color: #6B7280; font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem;">Diferença (Cascade - Modelo)</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: {gap_color}; margin-bottom: 0.5rem;">{gap_cascade_vs_model:+.1f}</div>
        <div style="color: #6B7280; font-size: 0.75rem;">t/ha</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

tab_results, tab_visual, tab_export = st.tabs(["📋 Análise Detalhada", "📊 Visualização", "💾 Exportar Dados"])

with tab_results:
    st.markdown(
        """
    <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0;">
        <h4 style="color: #124141; margin: 0 0 0.5rem 0; font-size: 1.1rem;">Tabela de Impactos</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.dataframe(impacts_df, use_container_width=True, height=420)

    st.markdown("### Diagnóstico de Entradas (Baseline vs Final)")
    st.dataframe(debug_df, use_container_width=True, height=360)

with tab_visual:
    st.markdown(
        """
    <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 1rem;">
        <h4 style="color: #124141; margin: 0 0 0.5rem 0; font-size: 1.1rem;">Waterfall - Cascade</h4>
        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Top impactos (sem “OUTROS”).</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    fig = create_waterfall_chart_pretty(
        impacts_dict=cascade_impacts,
        base_value=float(tch_base),
        final_value=float(tch_final_cascade),
        title=f"Waterfall (Visão {selected_vision})",
        max_items=18,
        base_label="Base",
        final_label="Final (Cascade)",
    )
    st.pyplot(fig, use_container_width=True)

with tab_export:
    st.markdown(
        """
    <div style="background: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0; margin-bottom: 1rem;">
        <h4 style="color: #124141; margin: 0 0 1rem 0; font-size: 1.1rem;">Exportar Dados</h4>
        <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">Download dos dados simulados em Excel</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if st.session_state.sim_result.get("selected_vision") == "TALHAO" and st.session_state.sim_result.get("selected_value") is not None:
        talhao_col = st.session_state.sim_result["column_mapping"].get("TALHAO")
        selected_value = st.session_state.sim_result.get("selected_value")

        if talhao_col and talhao_col in df_baseline.columns:
            try:
                talhao_data = df_baseline[df_baseline[talhao_col] == selected_value].copy()
            except Exception:
                talhao_data = df_baseline[df_baseline[talhao_col].astype(str) == str(selected_value)].copy()

            if not talhao_data.empty:
                export_data = talhao_data.copy()
                export_data["TCH_Predito_Modelo"] = tch_final_model
                export_data["TCH_Predito_Cascade"] = tch_final_cascade
                export_data["TCH_Base"] = tch_base
                export_data["Delta_Modelo"] = delta_model
                export_data["Delta_Cascade"] = delta_cascade
                export_data["Gap_Cascade_vs_Modelo"] = gap_cascade_vs_model

                baseline_df = st.session_state.sim_result["baseline_df"]
                final_features = st.session_state.sim_result["final_features"]

                for col in MODEL_FEATURES:
                    if col in baseline_df.columns:
                        export_data[f"Baseline_{col}"] = baseline_df[col].iloc[0]
                    if col in final_features.columns:
                        export_data[f"Final_{col}"] = final_features[col].iloc[0]

                for ofensor, impacto in cascade_impacts.items():
                    col_name = f'Impacto_{str(ofensor).replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_")}'
                    export_data[col_name] = impacto

                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    export_data.to_excel(writer, sheet_name="Dados_Talhao", index=False)
                    impacts_df.to_excel(writer, sheet_name="Impactos", index=False)
                    debug_df.to_excel(writer, sheet_name="Diagnostico_Inputs", index=False)

                st.download_button(
                    label="📥 Download Excel Completo",
                    data=buffer.getvalue(),
                    file_name=f"simulacao_talhao_{str(selected_value).replace('/', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.warning(f"Nenhum dado encontrado para o talhão '{selected_value}'.")
        else:
            st.warning("Coluna TALHAO não encontrada no baseline.")
    else:
        st.info("Selecione visão TALHAO para exportar dados específicos do talhão.")
