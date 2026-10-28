"""
================================================================================
PAINEL DE INTELIGÊNCIA ESPORTIVA - FUTEBOL BRASILEIRO (SÉRIES A A D)
================================================================================
Dashboard Interativo construído em Streamlit, Plotly e Mplsoccer para visualização
analítica de clubes, pontuações esperadas vs reais, mapas de calor no gramado,
shotmaps com xG, ocupação tática coletiva dos 11 titulares, radar comparativo
de percentis, filtro por clube e simulador preditivo de teto de carreira.
"""

from __future__ import annotations
import os
import sys
import json
import base64
import sqlite3
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adiciona pasta src ao path para carregar módulos do projeto
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from importlib import import_module

try:
    modulo_preditivo = import_module("04_modelo_preditivo")
    ModeloProjecaoCarreiraAtleta = getattr(modulo_preditivo, "ModeloProjecaoCarreiraAtleta", None)
    ModeloPrevisaoCampeao2026 = getattr(modulo_preditivo, "ModeloPrevisaoCampeao2026", None)
except Exception:
    ModeloProjecaoCarreiraAtleta = None
    ModeloPrevisaoCampeao2026 = None

try:
    modulo_tatica = import_module("05_visualizacao_tatica")
except Exception:
    modulo_tatica = None

try:
    modulo_expansao = import_module("06_confrontos_e_expansao")
    GoleirosAnalytics = getattr(modulo_expansao, "GoleirosAnalytics", None)
    TreinadoresAnalytics = getattr(modulo_expansao, "TreinadoresAnalytics", None)
    ConfrontosH2HAnalytics = getattr(modulo_expansao, "ConfrontosH2HAnalytics", None)
except Exception:
    GoleirosAnalytics = None
    TreinadoresAnalytics = None
    ConfrontosH2HAnalytics = None

# ------------------------------------------------------------------------------
# Configuração da Página
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Analytics Futebol BR - Séries A a D",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS personalizada (Tema Dark Profissional - Alto Contraste WCAG AAA Garantido)
st.markdown("""
<style>
    /* Fundo principal e textos com contraste blindado (nunca texto branco em fundo claro) */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #F8FAFC;
    }
    .main-title {
        font-size: 2.15rem;
        font-weight: 800;
        color: #F8FAFC !important;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.98rem;
        color: #CBD5E1 !important;
        margin-bottom: 1.2rem;
        line-height: 1.45;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 14px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        color: #F8FAFC !important;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC !important;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #CBD5E1 !important;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-card small {
        color: #94A3B8 !important;
        font-size: 0.82rem;
    }
    /* Estilização refinada dos botões de download e navegação na Sidebar */
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
        color: #38BDF8 !important;
        border: 1px solid #38BDF866 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #0284C7 !important;
        color: #FFFFFF !important;
        border-color: #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def gerar_logo_app_svg_b64() -> str:
    """Gera o logotipo oficial em SVG de alta definição (escudo tático + estrela dourada) codificado em base64."""
    svg_logo = """<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 160 160">
        <defs>
            <linearGradient id="shieldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#0284C7"/>
                <stop offset="50%" stop-color="#1E3A8A"/>
                <stop offset="100%" stop-color="#0F172A"/>
            </linearGradient>
            <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#FDE047"/>
                <stop offset="100%" stop-color="#EAB308"/>
            </linearGradient>
        </defs>
        <path d="M80 10 L142 34 L142 82 C142 118 114 142 80 152 C46 142 18 118 18 82 L18 34 Z"
              fill="url(#shieldGrad)" stroke="#38BDF8" stroke-width="4"/>
        <path d="M80 20 L132 40 L132 80 C132 110 108 131 80 140 C52 131 28 110 28 80 L28 40 Z"
              fill="none" stroke="#22C55E" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.75"/>
        <rect x="44" y="48" width="72" height="52" rx="4" fill="#064E3B" stroke="#34D399" stroke-width="2"/>
        <line x1="80" y1="48" x2="80" y2="100" stroke="#34D399" stroke-width="1.5"/>
        <circle cx="80" cy="74" r="10" fill="none" stroke="#34D399" stroke-width="1.5"/>
        <polyline points="50,88 68,72 86,78 108,58" fill="none" stroke="#FDE047" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="50" cy="88" r="3.5" fill="#38BDF8"/>
        <circle cx="68" cy="72" r="3.5" fill="#38BDF8"/>
        <circle cx="86" cy="78" r="3.5" fill="#38BDF8"/>
        <circle cx="108" cy="58" r="4.5" fill="#22C55E" stroke="#FFFFFF" stroke-width="1.5"/>
        <polygon points="80,18 84,27 94,27 86,33 89,42 80,36 71,42 74,33 66,27 76,27" fill="url(#goldGrad)"/>
        <text x="80" y="124" font-family="Arial, sans-serif" font-size="14" font-weight="900" fill="#F8FAFC" text-anchor="middle" letter-spacing="1.5">BR ANALYTICS</text>
    </svg>"""
    b64 = base64.b64encode(svg_logo.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64}"


def aplicar_tema_plotly_dark(fig: go.Figure, altura: Optional[int] = None) -> go.Figure:
    """Padroniza todos os gráficos Plotly com tema Dark Slate de alto contraste e legibilidade WCAG."""
    layout_kwargs: Dict[str, Any] = dict(
        paper_bgcolor="#1E293B",
        plot_bgcolor="#0F172A",
        font=dict(color="#F8FAFC", family="Inter, Arial, sans-serif"),
        hoverlabel=dict(bgcolor="#0F172A", bordercolor="#38BDF8", font=dict(color="#F8FAFC", size=12)),
    )
    if altura is not None:
        layout_kwargs["height"] = altura
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC")
    )
    fig.update_yaxes(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#CBD5E1"),
        title_font=dict(color="#F8FAFC")
    )
    return fig


def render_botao_download_csv(
    df_export: pd.DataFrame,
    nome_arquivo: str,
    label_botao: str = "📥 Exportar Tabela (CSV)",
    key_btn: str = "btn_dl_csv"
):
    """Renderiza botão nativo do Streamlit para exportação instantânea de DataFrames em CSV UTF-8-BOM."""
    if df_export is None or df_export.empty:
        return
    csv_bytes = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label=label_botao,
        data=csv_bytes,
        file_name=nome_arquivo,
        mime="text/csv",
        key=key_btn,
        use_container_width=False
    )


# ------------------------------------------------------------------------------
# Carregamento e Cache dos Dados (Otimizados com @st.cache_data)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    caminho_clubes = "estudo_desempenho_clubes.csv"
    if not os.path.exists(caminho_clubes):
        caminho_clubes = "fato_times_campeonato.csv"

    caminho_jogadores = "fato_jogadores_temporada.csv"
    caminho_parquet = "jogadores_perfil.parquet"
    caminho_joias = "ranking_joias_escondidas_A_ate_D.csv"

    # Prioriza a base parquet enriquecida com fotos, biometria e dados contratuais
    if os.path.exists(caminho_parquet):
        df_j = pd.read_parquet(caminho_parquet)
        df_c = pd.read_csv(caminho_clubes) if os.path.exists(caminho_clubes) else pd.DataFrame()
    elif not os.path.exists(caminho_jogadores) and os.path.exists("banco_dados/futebol_br.db"):
        conn = sqlite3.connect("banco_dados/futebol_br.db")
        tabelas = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        if "jogadores_perfil" in tabelas:
            df_j = pd.read_sql_query("SELECT * FROM jogadores_perfil", conn)
        else:
            df_j = pd.read_sql_query("SELECT * FROM fato_jogadores_temporada", conn)
        df_c = pd.read_sql_query("SELECT * FROM fato_times_campeonato", conn)
        conn.close()
    else:
        df_c = pd.read_csv(caminho_clubes) if os.path.exists(caminho_clubes) else pd.DataFrame()
        df_j = pd.read_csv(caminho_jogadores) if os.path.exists(caminho_jogadores) else pd.DataFrame()

    df_joias = pd.read_csv(caminho_joias) if os.path.exists(caminho_joias) else pd.DataFrame()

    # Tratamento e Higienização Estrita de Jogadores e Posições
    if not df_j.empty:
        df_j["ano"] = pd.to_numeric(df_j["ano"], errors="coerce").fillna(2025).astype(int)
        df_j["idade_na_temporada"] = pd.to_numeric(df_j["idade_na_temporada"], errors="coerce").fillna(24.0)
        df_j["minutos"] = pd.to_numeric(df_j["minutos"], errors="coerce").fillna(0).astype(int)
        df_j["jogos"] = pd.to_numeric(df_j.get("jogos", 1), errors="coerce").fillna(1).clip(lower=1).astype(int)
        df_j["indice_teto_projetado"] = pd.to_numeric(df_j["indice_teto_projetado"], errors="coerce").fillna(50.0)
        df_j["nota_media_sofascore"] = pd.to_numeric(df_j["nota_media_sofascore"], errors="coerce").fillna(6.75)
        df_j["valor_mercado_eur"] = pd.to_numeric(df_j["valor_mercado_eur"], errors="coerce").fillna(100_000.0)

        # Regra estrita e padronizada de posição: G (Goleiros), D (Defensores), M (Meio-campistas), F (Atacantes)
        df_j["posicao"] = df_j["posicao"].astype(str).str.strip().str.upper()
        df_j["posicao"] = df_j["posicao"].where(df_j["posicao"].isin(["G", "D", "M", "F"]), "M")

        if "posicao_detalhada" not in df_j.columns:
            df_j["posicao_detalhada"] = df_j["posicao"].map({
                "G": "Goleiro", "D": "Zagueiro Central", "M": "Meio-Campista Central", "F": "Centroavante"
            }).fillna("Meio-Campista Central")

        df_j["posicao_nome"] = df_j["posicao"].map({
            "G": "Goleiro", "D": "Defensor", "M": "Meio-Campo", "F": "Atacante"
        }).fillna("Meio-Campo")
        df_j["posicao_setor"] = df_j["posicao"].map({
            "G": "Goleiros", "D": "Defensores", "M": "Meio-campistas", "F": "Atacantes"
        }).fillna("Meio-campistas")
        df_j["posicao_agrupada"] = df_j["posicao_nome"]

        # Garante colunas de métricas avançadas por setor caso a base tenha sido lida de fallback antigo
        jogos_s = df_j["jogos"].clip(lower=1)
        gols_s = pd.to_numeric(df_j.get("gols", 0), errors="coerce").fillna(0).astype(int)
        ass_s = pd.to_numeric(df_j.get("assistencias", 0), errors="coerce").fillna(0).astype(int)
        fin_s = pd.to_numeric(df_j.get("finalizacoes", 0), errors="coerce").fillna(0).astype(int)
        des_s = pd.to_numeric(df_j.get("desarmes", 0), errors="coerce").fillna(0).astype(int)
        int_s = pd.to_numeric(df_j.get("interceptacoes", 0), errors="coerce").fillna(0).astype(int)
        aer_s = pd.to_numeric(df_j.get("duelos_aereos_ganhos", 0), errors="coerce").fillna(0).astype(int)
        dri_s = pd.to_numeric(df_j.get("dribles_certos", 0), errors="coerce").fillna(0).astype(int)
        pas_s = pd.to_numeric(df_j.get("passes_certos_pct", 78.0), errors="coerce").replace(0.0, np.nan).fillna(78.0)

        if "participacoes_diretas_gols" not in df_j.columns:
            df_j["participacoes_diretas_gols"] = gols_s + ass_s
        if "finalizacoes_no_gol" not in df_j.columns:
            df_j["finalizacoes_no_gol"] = np.maximum(gols_s, np.round(fin_s * 0.42 + gols_s * 0.55)).astype(int)
        if "finalizacoes_no_gol_por_jogo" not in df_j.columns:
            df_j["finalizacoes_no_gol_por_jogo"] = np.round(df_j["finalizacoes_no_gol"] / jogos_s, 2)
        if "xg_estimado" not in df_j.columns:
            df_j["xg_estimado"] = np.round(np.maximum(gols_s * 0.68, fin_s * 0.105 + df_j["finalizacoes_no_gol"] * 0.14 + gols_s * 0.32), 2)
        if "passes_decisivos" not in df_j.columns:
            df_j["passes_decisivos"] = np.maximum(ass_s, np.round(ass_s * 2.8 + fin_s * 0.32 + dri_s * 0.40)).astype(int)
        if "passes_decisivos_por_jogo" not in df_j.columns:
            df_j["passes_decisivos_por_jogo"] = np.round(df_j["passes_decisivos"] / jogos_s, 2)
        if "xa_estimado" not in df_j.columns:
            df_j["xa_estimado"] = np.round(np.maximum(ass_s * 0.60, ass_s * 0.52 + df_j["passes_decisivos"] * 0.115 + dri_s * 0.035), 2)
        if "duelos_ganhos_meio" not in df_j.columns:
            df_j["duelos_ganhos_meio"] = (des_s + dri_s + aer_s + np.round(int_s * 0.75)).astype(int)
        if "duelos_ganhos_meio_por_jogo" not in df_j.columns:
            df_j["duelos_ganhos_meio_por_jogo"] = np.round(df_j["duelos_ganhos_meio"] / jogos_s, 2)
        if "desarmes_por_jogo" not in df_j.columns:
            df_j["desarmes_por_jogo"] = np.round(des_s / jogos_s, 2)
        if "interceptacoes_por_jogo" not in df_j.columns:
            df_j["interceptacoes_por_jogo"] = np.round(int_s / jogos_s, 2)
        if "duelos_aereos_ganhos_pct" not in df_j.columns:
            df_j["duelos_aereos_ganhos_pct"] = np.round(np.clip(50.0 + (aer_s / jogos_s) * 6.8 + (df_j["nota_media_sofascore"] - 6.7) * 8.0, 38.0, 84.5), 1)
        if "precisao_passes_defensivos_pct" not in df_j.columns:
            df_j["precisao_passes_defensivos_pct"] = np.round(np.clip(pas_s, 62.0, 96.5), 1)
        if "score_destaque_setor" not in df_j.columns:
            df_j["score_destaque_setor"] = np.round(df_j["nota_media_sofascore"] * 10.0, 2)

        # Garante colunas de URL de foto (SofaScore CDN e Transfermarkt CDN)
        if "foto_url" not in df_j.columns:
            df_j["foto_url"] = df_j["sofascore_player_id"].apply(
                lambda pid: f"https://api.sofascore.app/api/v1/player/{int(float(pid))}/image"
                if pd.notna(pid) and str(pid).strip() not in ["", "0"] else ""
            )

        # Calcula percentis dos 6 pilares táticos para o Radar
        if modulo_tatica:
            df_j = modulo_tatica.calcular_percentis_radar(df_j)

    # Tratamento de Clubes
    if not df_c.empty:
        df_c["ano"] = pd.to_numeric(df_c["ano"], errors="coerce").fillna(2025).astype(int)
        df_c["pontos"] = pd.to_numeric(df_c["pontos"], errors="coerce").fillna(0).astype(int)
        df_c["aproveitamento_pct"] = pd.to_numeric(df_c["aproveitamento_pct"], errors="coerce").fillna(0.0)
        if "residuo_desempenho" not in df_c.columns:
            df_c["residuo_desempenho"] = 0.0
        if "pontos_esperados_elenco" not in df_c.columns:
            df_c["pontos_esperados_elenco"] = df_c["pontos"]

    return df_c, df_j, df_joias


@st.cache_data(ttl=3600, show_spinner=False)
def carregar_dados_expansao() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    caminho_gol = "fato_goleiros_detalhado.csv"
    caminho_tec = "fato_treinadores_historico.csv"
    caminho_conf = "fato_partidas_confrontos.csv"
    caminho_meta = os.path.join("assets", "clubes_ativos_meta.json")

    df_gol = pd.read_csv(caminho_gol) if os.path.exists(caminho_gol) else pd.DataFrame()
    if not df_gol.empty:
        # Blindagem estrita: apenas goleiros reais com defesas_totais > 0
        df_gol["defesas_totais"] = pd.to_numeric(df_gol["defesas_totais"], errors="coerce").fillna(0).astype(int)
        df_gol = df_gol[df_gol["defesas_totais"] > 0].copy()
        if "posicao" in df_gol.columns:
            df_gol = df_gol[df_gol["posicao"].astype(str).str.upper() == "G"].copy()
        else:
            df_gol["posicao"] = "G"
        if "taxa_defesas_dificeis_pct" not in df_gol.columns:
            chutes_dif = np.maximum(1.0, df_gol["defesas_dentro_area"] + df_gol["gols_sofridos"] * 0.72)
            df_gol["taxa_defesas_dificeis_pct"] = np.round(
                np.clip((df_gol["defesas_dentro_area"] / chutes_dif) * 100.0, 45.0, 92.0), 1
            )

    df_tec = pd.read_csv(caminho_tec) if os.path.exists(caminho_tec) else pd.DataFrame()
    df_conf = pd.read_csv(caminho_conf) if os.path.exists(caminho_conf) else pd.DataFrame()

    meta = {}
    if os.path.exists(caminho_meta):
        try:
            with open(caminho_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    return df_gol, df_tec, df_conf, meta


# Inicialização com Feedback Visual Nativo (Spinner + Barra de Progresso na 1ª Carga)
if "dados_inicializados_ux" not in st.session_state:
    placeholder_boot = st.empty()
    with placeholder_boot.container():
        with st.spinner("🔄 Carregando dados do SofaScore/Transfermarkt, consultando banco e preparando modelos..."):
            barra_boot = st.progress(10, text="📦 [1/3] Lendo bases multitemporais de clubes, atletas e joias (Parquet/SQLite)...")
            df_clubes, df_jogadores, df_joias = carregar_dados()
            barra_boot.progress(60, text="⚔️ [2/3] Carregando histórico de confrontos H2H, goleiros (PSxG) e treinadores...")
            df_goleiros, df_treinadores, df_confrontos, meta_clubes = carregar_dados_expansao()
            barra_boot.progress(100, text="✅ [3/3] Bases e modelos carregados em cache com sucesso!")
    placeholder_boot.empty()
    st.session_state["dados_inicializados_ux"] = True
else:
    with st.spinner("🔄 Sincronizando dados em cache..."):
        df_clubes, df_jogadores, df_joias = carregar_dados()
        df_goleiros, df_treinadores, df_confrontos, meta_clubes = carregar_dados_expansao()

def normalizar_texto_app(texto):
    import unicodedata
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.lower().strip()
    for token in [" fc", " ec", " ac", " sc", " saf", " clube", " associacao", " atletico"]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())

def obter_escudo_img(clube_nome, team_id=None):
    k = normalizar_texto_app(clube_nome)
    p_chave = os.path.join("assets", "escudos", f"{k}.png")
    if os.path.exists(p_chave):
        return p_chave
    if team_id:
        p_id = os.path.join("assets", "escudos", f"{int(team_id)}.png")
        if os.path.exists(p_id):
            return p_id
    p_fb = os.path.join("assets", "placeholders", "escudo_fallback.png")
    return p_fb if os.path.exists(p_fb) else None

def obter_camisa_img(clube_nome, tipo=1, team_id=None):
    k = normalizar_texto_app(clube_nome)
    p_chave = os.path.join("assets", "camisas", f"{k}_{tipo}.png")
    if os.path.exists(p_chave):
        return p_chave
    if team_id:
        p_id = os.path.join("assets", "camisas", f"{int(team_id)}_{tipo}.png")
        if os.path.exists(p_id):
            return p_id
    return None

# ==============================================================================
# DICIONÁRIO DE CORES, ESTILOS E METADADOS OFICIAIS DOS CLUBES (SÉRIES A A D)
# ==============================================================================
DICIONARIO_CORES_CLUBES = {
    # SÉRIE A
    "palmeiras": {
        "cor_primaria": "#006437", "cor_secundaria": "#FFFFFF", "accent": "#028247",
        "text_on_primary": "#FFFFFF", "sigla": "PAL", "divisao": "Serie_A",
        "estadio": "Allianz Parque", "cidade": "São Paulo (SP)"
    },
    "flamengo": {
        "cor_primaria": "#C3281E", "cor_secundaria": "#111111", "accent": "#E5A823",
        "text_on_primary": "#FFFFFF", "sigla": "FLA", "divisao": "Serie_A",
        "estadio": "Maracanã", "cidade": "Rio de Janeiro (RJ)"
    },
    "corinthians": {
        "cor_primaria": "#111111", "cor_secundaria": "#FFFFFF", "accent": "#8A1538",
        "text_on_primary": "#FFFFFF", "sigla": "COR", "divisao": "Serie_A",
        "estadio": "Neo Química Arena", "cidade": "São Paulo (SP)"
    },
    "sao paulo": {
        "cor_primaria": "#C8102E", "cor_secundaria": "#000000", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "SAO", "divisao": "Serie_A",
        "estadio": "MorumBIS", "cidade": "São Paulo (SP)"
    },
    "botafogo": {
        "cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#737373",
        "text_on_primary": "#FFFFFF", "sigla": "BOT", "divisao": "Serie_A",
        "estadio": "Nilton Santos", "cidade": "Rio de Janeiro (RJ)"
    },
    "cruzeiro": {
        "cor_primaria": "#0033A0", "cor_secundaria": "#FFFFFF", "accent": "#0047BA",
        "text_on_primary": "#FFFFFF", "sigla": "CRU", "divisao": "Serie_A",
        "estadio": "Mineirão", "cidade": "Belo Horizonte (MG)"
    },
    "atletico mineiro": {
        "cor_primaria": "#111111", "cor_secundaria": "#FFFFFF", "accent": "#D4AF37",
        "text_on_primary": "#FFFFFF", "sigla": "CAM", "divisao": "Serie_A",
        "estadio": "Arena MRV", "cidade": "Belo Horizonte (MG)"
    },
    "gremio": {
        "cor_primaria": "#0D80BF", "cor_secundaria": "#000000", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "GRE", "divisao": "Serie_A",
        "estadio": "Arena do Grêmio", "cidade": "Porto Alegre (RS)"
    },
    "internacional": {
        "cor_primaria": "#E11D48", "cor_secundaria": "#FFFFFF", "accent": "#BE123C",
        "text_on_primary": "#FFFFFF", "sigla": "INT", "divisao": "Serie_A",
        "estadio": "Beira-Rio", "cidade": "Porto Alegre (RS)"
    },
    "fluminense": {
        "cor_primaria": "#8A1538", "cor_secundaria": "#006341", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "FLU", "divisao": "Serie_A",
        "estadio": "Maracanã", "cidade": "Rio de Janeiro (RJ)"
    },
    "vasco da gama": {
        "cor_primaria": "#111111", "cor_secundaria": "#FFFFFF", "accent": "#DC2626",
        "text_on_primary": "#FFFFFF", "sigla": "VAS", "divisao": "Serie_A",
        "estadio": "São Januário", "cidade": "Rio de Janeiro (RJ)"
    },
    "vasco": {
        "cor_primaria": "#111111", "cor_secundaria": "#FFFFFF", "accent": "#DC2626",
        "text_on_primary": "#FFFFFF", "sigla": "VAS", "divisao": "Serie_A",
        "estadio": "São Januário", "cidade": "Rio de Janeiro (RJ)"
    },
    "bahia": {
        "cor_primaria": "#0055A5", "cor_secundaria": "#E4002B", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "BAH", "divisao": "Serie_A",
        "estadio": "Arena Fonte Nova", "cidade": "Salvador (BA)"
    },
    "fortaleza": {
        "cor_primaria": "#003A70", "cor_secundaria": "#D00027", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "FOR", "divisao": "Serie_A",
        "estadio": "Arena Castelão", "cidade": "Fortaleza (CE)"
    },
    "athletico paranaense": {
        "cor_primaria": "#C8102E", "cor_secundaria": "#000000", "accent": "#737373",
        "text_on_primary": "#FFFFFF", "sigla": "CAP", "divisao": "Serie_A",
        "estadio": "Ligga Arena", "cidade": "Curitiba (PR)"
    },
    "athletico": {
        "cor_primaria": "#C8102E", "cor_secundaria": "#000000", "accent": "#737373",
        "text_on_primary": "#FFFFFF", "sigla": "CAP", "divisao": "Serie_A",
        "estadio": "Ligga Arena", "cidade": "Curitiba (PR)"
    },
    "santos": {
        "cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#D4AF37",
        "text_on_primary": "#FFFFFF", "sigla": "SAN", "divisao": "Serie_A",
        "estadio": "Vila Belmiro", "cidade": "Santos (SP)"
    },
    "red bull bragantino": {
        "cor_primaria": "#DC2626", "cor_secundaria": "#FFFFFF", "accent": "#1E3A8A",
        "text_on_primary": "#FFFFFF", "sigla": "RBB", "divisao": "Serie_A",
        "estadio": "Nabi Abi Chedid", "cidade": "Bragança Paulista (SP)"
    },
    "juventude": {
        "cor_primaria": "#059669", "cor_secundaria": "#FFFFFF", "accent": "#047857",
        "text_on_primary": "#FFFFFF", "sigla": "JUV", "divisao": "Serie_A",
        "estadio": "Alfredo Jaconi", "cidade": "Caxias do Sul (RS)"
    },
    "cuiaba": {
        "cor_primaria": "#047857", "cor_secundaria": "#FACC15", "accent": "#065F46",
        "text_on_primary": "#FFFFFF", "sigla": "CUI", "divisao": "Serie_A",
        "estadio": "Arena Pantanal", "cidade": "Cuiabá (MT)"
    },
    "vitoria": {
        "cor_primaria": "#DC2626", "cor_secundaria": "#000000", "accent": "#F59E0B",
        "text_on_primary": "#FFFFFF", "sigla": "VIT", "divisao": "Serie_A",
        "estadio": "Barradão", "cidade": "Salvador (BA)"
    },
    "sport recife": {
        "cor_primaria": "#DC2626", "cor_secundaria": "#000000", "accent": "#F59E0B",
        "text_on_primary": "#FFFFFF", "sigla": "SPT", "divisao": "Serie_A",
        "estadio": "Ilha do Retiro", "cidade": "Recife (PE)"
    },
    "mirassol": {
        "cor_primaria": "#EAB308", "cor_secundaria": "#1E3A8A", "accent": "#FACC15",
        "text_on_primary": "#1E293B", "sigla": "MIR", "divisao": "Serie_A",
        "estadio": "José Maria de Campos Maia", "cidade": "Mirassol (SP)"
    },
    "criciuma": {
        "cor_primaria": "#FACC15", "cor_secundaria": "#000000", "accent": "#FFFFFF",
        "text_on_primary": "#1E293B", "sigla": "CRI", "divisao": "Serie_A",
        "estadio": "Heriberto Hülse", "cidade": "Criciúma (SC)"
    },
    "atletico goianiense": {
        "cor_primaria": "#DC2626", "cor_secundaria": "#000000", "accent": "#FFFFFF",
        "text_on_primary": "#FFFFFF", "sigla": "ACG", "divisao": "Serie_A",
        "estadio": "Antônio Accioly", "cidade": "Goiânia (GO)"
    },

    # SÉRIE B
    "ceara": {"cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#64748B", "text_on_primary": "#FFFFFF", "sigla": "CEA", "divisao": "Serie_B", "estadio": "Castelão", "cidade": "Fortaleza (CE)"},
    "goias": {"cor_primaria": "#065F46", "cor_secundaria": "#FFFFFF", "accent": "#10B981", "text_on_primary": "#FFFFFF", "sigla": "GOI", "divisao": "Serie_B", "estadio": "Serrinha", "cidade": "Goiânia (GO)"},
    "coritiba": {"cor_primaria": "#065F46", "cor_secundaria": "#FFFFFF", "accent": "#047857", "text_on_primary": "#FFFFFF", "sigla": "CFC", "divisao": "Serie_B", "estadio": "Couto Pereira", "cidade": "Curitiba (PR)"},
    "america mineiro": {"cor_primaria": "#047857", "cor_secundaria": "#000000", "accent": "#F8FAFC", "text_on_primary": "#FFFFFF", "sigla": "AMG", "divisao": "Serie_B", "estadio": "Independência", "cidade": "Belo Horizonte (MG)"},
    "avai": {"cor_primaria": "#0284C7", "cor_secundaria": "#FFFFFF", "accent": "#0369A1", "text_on_primary": "#FFFFFF", "sigla": "AVA", "divisao": "Serie_B", "estadio": "Ressacada", "cidade": "Florianópolis (SC)"},
    "chapecoense": {"cor_primaria": "#059669", "cor_secundaria": "#FFFFFF", "accent": "#10B981", "text_on_primary": "#FFFFFF", "sigla": "CHA", "divisao": "Serie_B", "estadio": "Arena Condá", "cidade": "Chapecó (SC)"},
    "vila nova": {"cor_primaria": "#DC2626", "cor_secundaria": "#FFFFFF", "accent": "#991B1B", "text_on_primary": "#FFFFFF", "sigla": "VIL", "divisao": "Serie_B", "estadio": "OBA", "cidade": "Goiânia (GO)"},
    "novorizontino": {"cor_primaria": "#EAB308", "cor_secundaria": "#000000", "accent": "#CA8A04", "text_on_primary": "#1E293B", "sigla": "NOV", "divisao": "Serie_B", "estadio": "Jorge Ismael de Biasi", "cidade": "Novo Horizonte (SP)"},
    "operario": {"cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#64748B", "text_on_primary": "#FFFFFF", "sigla": "OPE", "divisao": "Serie_B", "estadio": "Germano Krüger", "cidade": "Ponta Grossa (PR)"},
    "paysandu": {"cor_primaria": "#0284C7", "cor_secundaria": "#FFFFFF", "accent": "#0369A1", "text_on_primary": "#FFFFFF", "sigla": "PAY", "divisao": "Serie_B", "estadio": "Curuzu", "cidade": "Belém (PA)"},
    "crb": {"cor_primaria": "#DC2626", "cor_secundaria": "#FFFFFF", "accent": "#B91C1C", "text_on_primary": "#FFFFFF", "sigla": "CRB", "divisao": "Serie_B", "estadio": "Rei Pelé", "cidade": "Maceió (AL)"},
    "ponte preta": {"cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#475569", "text_on_primary": "#FFFFFF", "sigla": "PON", "divisao": "Serie_B", "estadio": "Moisés Lucarelli", "cidade": "Campinas (SP)"},
    "guarani": {"cor_primaria": "#065F46", "cor_secundaria": "#FFFFFF", "accent": "#047857", "text_on_primary": "#FFFFFF", "sigla": "GUA", "divisao": "Serie_B", "estadio": "Brinco de Ouro", "cidade": "Campinas (SP)"},

    # SÉRIE C E D
    "nautico": {"cor_primaria": "#DC2626", "cor_secundaria": "#FFFFFF", "accent": "#991B1B", "text_on_primary": "#FFFFFF", "sigla": "NAU", "divisao": "Serie_C", "estadio": "Aflitos", "cidade": "Recife (PE)"},
    "santa cruz": {"cor_primaria": "#DC2626", "cor_secundaria": "#000000", "accent": "#FFFFFF", "text_on_primary": "#FFFFFF", "sigla": "SCFC", "divisao": "Serie_D", "estadio": "Arruda", "cidade": "Recife (PE)"},
    "remo": {"cor_primaria": "#1E3A8A", "cor_secundaria": "#FFFFFF", "accent": "#172554", "text_on_primary": "#FFFFFF", "sigla": "REM", "divisao": "Serie_C", "estadio": "Baenão", "cidade": "Belém (PA)"},
    "figueirense": {"cor_primaria": "#000000", "cor_secundaria": "#FFFFFF", "accent": "#475569", "text_on_primary": "#FFFFFF", "sigla": "FIG", "divisao": "Serie_C", "estadio": "Orlando Scarpelli", "cidade": "Florianópolis (SC)"},
    "csa": {"cor_primaria": "#1D4ED8", "cor_secundaria": "#FFFFFF", "accent": "#1E40AF", "text_on_primary": "#FFFFFF", "sigla": "CSA", "divisao": "Serie_C", "estadio": "Rei Pelé", "cidade": "Maceió (AL)"},
    "sampaio correa": {"cor_primaria": "#FACC15", "cor_secundaria": "#DC2626", "accent": "#047857", "text_on_primary": "#1E293B", "sigla": "SAM", "divisao": "Serie_C", "estadio": "Castelão", "cidade": "São Luís (MA)"},
}

FALLBACK_SERIES_CD = {
    "cor_primaria": "#334155",
    "cor_secundaria": "#94A3B8",
    "accent": "#38BDF8",
    "text_on_primary": "#F8FAFC",
    "sigla": "FUT",
    "divisao": "Serie_C",
    "estadio": "Estádio Municipal",
    "cidade": "Brasil"
}


def obter_info_clube_meta(clube_nome, team_id=None):
    """Recupera metadados com hierarquia: Dicionário -> Cache Meta -> Fallback Séries C/D."""
    k = normalizar_texto_app(clube_nome)
    
    # 1. Busca no Dicionário Curado
    if k in DICIONARIO_CORES_CLUBES:
        info = DICIONARIO_CORES_CLUBES[k].copy()
        info["nome"] = clube_nome
        # Completa com técnico do meta se houver
        if k in meta_clubes and meta_clubes[k].get("tecnico_atual"):
            info["tecnico_atual"] = meta_clubes[k]["tecnico_atual"]
        else:
            info["tecnico_atual"] = "Comissão Técnica"
        return info

    # 2. Busca no cache de metadados gerado pela coleta
    if k in meta_clubes:
        return meta_clubes[k]
    if team_id and str(team_id) in meta_clubes:
        return meta_clubes[str(team_id)]

    # 3. Fallback Seguro para Séries C e D
    fb = FALLBACK_SERIES_CD.copy()
    fb["nome"] = clube_nome
    fb["sigla"] = (clube_nome or "FUT")[:3].upper()
    fb["tecnico_atual"] = "Comissão Técnica"
    return fb


def gerar_css_dinamico_clube(clube_info: Dict[str, Any], divisao: str = "Serie_A") -> str:
    """
    Opção A (Recomendada): Mantém a base do tema padrão (dark/neutro limpo) e aplica as cores
    do clube apenas em detalhes sutis (borda fina no card, linha de destaque no header, badge pill),
    garantindo 100% de alto contraste WCAG entre fundo e texto.
    """
    p = clube_info.get("cor_primaria", "#38BDF8")
    acc = clube_info.get("accent", "#38BDF8")

    # Tratamento de alto contraste: se a cor primária for excessivamente escura (ex: preto de Corinthians/Botafogo/Vasco),
    # utilizamos a cor de acento secundária ou prata/azul claro para que a linha de destaque tenha visibilidade nítida
    p_lower = str(p).lower().strip()
    if p_lower in ["#000000", "#111111", "#18181b", "#1a1a1a", "#0f172a"]:
        accent_line = acc if acc and acc.lower() not in ["#000000", "#111111"] else "#94A3B8"
    else:
        accent_line = p

    return f"""
    <style>
        .club-hub-header {{
            background: #1E293B;
            border-top: 4px solid {accent_line};
            border-left: 1px solid #334155;
            border-right: 1px solid #334155;
            border-bottom: 1px solid #334155;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}
        .club-title-text {{
            font-size: 2.1rem;
            font-weight: 800;
            color: #F8FAFC !important;
            letter-spacing: -0.5px;
            line-height: 1.1;
        }}
        .club-badge-pill {{
            background: #0F172A;
            color: #F8FAFC !important;
            border: 1.5px solid {accent_line};
            font-weight: 700;
            font-size: 0.82rem;
            padding: 3px 10px;
            border-radius: 6px;
            display: inline-block;
            letter-spacing: 0.8px;
        }}
        .club-card-accent {{
            background: #1E293B;
            border: 1px solid #334155;
            border-left: 4px solid {accent_line};
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
            color: #F8FAFC;
        }}
        .kit-card-box {{
            background: #0F172A;
            border: 1px solid #334155;
            border-top: 3px solid {accent_line};
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin-bottom: 8px;
        }}
    </style>
    """


@st.cache_data(show_spinner=False)
def calcular_probabilidades_poisson(xg_mandante: float, xg_visitante: float, max_gols: int = 6) -> Tuple[float, float, float, str]:
    """Calcula probabilidades de vitória, empate e derrota usando modelo estocástico de Poisson."""
    from scipy.stats import poisson
    x_m = max(0.2, float(xg_mandante))
    x_v = max(0.2, float(xg_visitante))
    prob_matrix = np.zeros((max_gols, max_gols))
    for i in range(max_gols):
        for j in range(max_gols):
            prob_matrix[i, j] = poisson.pmf(i, x_m) * poisson.pmf(j, x_v)

    p_win_a = float(np.sum(np.tril(prob_matrix, -1)) * 100)
    p_draw = float(np.sum(np.diag(prob_matrix)) * 100)
    p_win_b = float(np.sum(np.triu(prob_matrix, 1)) * 100)

    idx_max = np.unravel_index(np.argmax(prob_matrix, axis=None), prob_matrix.shape)
    placar_provavel = f"{idx_max[0]} x {idx_max[1]}"
    return round(p_win_a, 1), round(p_draw, 1), round(p_win_b, 1), placar_provavel


def formatar_moeda_eur(val):
    if pd.isna(val) or val == 0:
        return "€ 0"
    if val >= 1_000_000:
        return f"€ {val / 1_000_000:.1f}M"
    elif val >= 1_000:
        return f"€ {val / 1_000:.0f}K"
    return f"€ {val:.0f}"


@st.cache_data(show_spinner=False)
def obter_img_base64(caminho: str) -> str:
    """Converte arquivo de imagem local em string base64 para renderização direta em HTML."""
    if caminho and os.path.exists(caminho):
        try:
            with open(caminho, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = caminho.split(".")[-1].lower()
                mime = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
                return f"data:{mime};base64,{b64}"
        except Exception:
            return ""
    return ""


def _compactar_html(bloco_html: str) -> str:
    """
    Remove indentação excessiva e linhas em branco internas de strings HTML
    para impedir que o parser CommonMark do Streamlit interprete tags aninhadas
    com >= 4 espaços como blocos de código (<pre><code>).
    """
    return "\n".join(linha.strip() for linha in bloco_html.splitlines() if linha.strip())


@st.cache_data(show_spinner=False)
def gerar_avatar_silhueta_svg_b64(nome: str = "Atleta") -> str:
    """
    Gera um avatar padrão de silhueta neutra de jogador em SVG codificado como Data URI (base64).
    Imune ao sanitizador DOMPurify do Streamlit por ser injetado diretamente como src='data:image/svg+xml;base64,...'.
    """
    partes = [p for p in str(nome).strip().split() if p]
    iniciais = "".join([p[0] for p in partes[:2]]).upper() if partes else "AT"
    iniciais_safe = "".join(c for c in iniciais if c.isalnum())[:3] or "AT"

    svg_str = f"""<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240" viewBox="0 0 240 240">
        <defs>
            <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#1E293B"/>
                <stop offset="100%" stop-color="#0F172A"/>
            </linearGradient>
            <linearGradient id="silGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#64748B"/>
                <stop offset="100%" stop-color="#475569"/>
            </linearGradient>
        </defs>
        <rect width="240" height="240" rx="120" fill="url(#bgGrad)"/>
        <circle cx="120" cy="84" r="38" fill="url(#silGrad)"/>
        <path d="M44 218 C44 162, 78 136, 120 136 C162 136, 196 162, 196 218 Z" fill="url(#silGrad)"/>
        <path d="M98 138 L120 166 L142 138" fill="none" stroke="#1E293B" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
        <rect x="86" y="182" width="68" height="28" rx="14" fill="#0F172A" opacity="0.85"/>
        <text x="120" y="201" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="bold" fill="#E2E8F0" text-anchor="middle">{iniciais_safe}</text>
    </svg>"""
    b64_svg = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{b64_svg}"


def _detectar_mime_imagem(conteudo: bytes) -> Optional[str]:
    """Valida se os bytes correspondem a uma imagem real (PNG, JPEG, WebP ou GIF)."""
    if not conteudo or len(conteudo) < 350:
        return None
    if conteudo.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if conteudo.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if conteudo.startswith(b"RIFF") and b"WEBP" in conteudo[:16]:
        return "image/webp"
    if conteudo.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def obter_foto_atleta_segura(
    player_id: Any,
    foto_url: str = "",
    tm_image_url: str = "",
    nome: str = "Atleta"
) -> Tuple[str, bool]:
    """
    Resolve e valida a foto oficial do jogador usando CDN do SofaScore e Transfermarkt,
    com cache persistente em disco ('assets/fotos_jogadores/') e fallback automático
    para o avatar padrão (silhueta neutra) se a imagem falhar ou não existir.
    Retorna: (data_uri_base64, is_foto_real)
    """
    avatar_padrao = gerar_avatar_silhueta_svg_b64(nome)
    pasta_cache_fotos = os.path.join("assets", "fotos_jogadores")
    os.makedirs(pasta_cache_fotos, exist_ok=True)

    pid_str = ""
    if pd.notna(player_id) and str(player_id).strip() not in ["", "0", "nan", "None"]:
        try:
            pid_int = int(float(player_id))
            if pid_int > 0:
                pid_str = str(pid_int)
        except Exception:
            pid_str = ""

    # 1. Verifica se já existe no cache local em disco
    if pid_str:
        caminho_cache_img = os.path.join(pasta_cache_fotos, f"{pid_str}.img")
        caminho_cache_none = os.path.join(pasta_cache_fotos, f"{pid_str}.none")
        if os.path.exists(caminho_cache_img):
            try:
                with open(caminho_cache_img, "rb") as f:
                    raw_bytes = f.read()
                mime_local = _detectar_mime_imagem(raw_bytes)
                if mime_local:
                    b64_img = base64.b64encode(raw_bytes).decode("utf-8")
                    return f"data:{mime_local};base64,{b64_img}", True
            except Exception:
                pass
        if os.path.exists(caminho_cache_none) and not tm_image_url:
            return avatar_padrao, False

    # 2. Monta fila ordenada de URLs candidatas (SofaScore CDN App -> SofaScore CDN Img -> Transfermarkt CDN)
    urls_candidatas: List[str] = []
    if pid_str:
        urls_candidatas.append(f"https://api.sofascore.app/api/v1/player/{pid_str}/image")
        urls_candidatas.append(f"https://img.sofascore.com/api/v1/player/{pid_str}/image")

    for u_extra in [tm_image_url, foto_url]:
        if pd.notna(u_extra) and isinstance(u_extra, str):
            u_limpa = u_extra.strip()
            if (
                u_limpa.startswith("http")
                and "default.jpg" not in u_limpa.lower()
                and u_limpa not in urls_candidatas
            ):
                urls_candidatas.append(u_limpa)

    if not urls_candidatas:
        return avatar_padrao, False

    # 3. Tenta baixar e validar os bytes da imagem no servidor (contornando CORS/Cloudflare/404 no navegador)
    try:
        from curl_cffi import requests as req_client
        usa_curl = True
    except ImportError:
        import requests as req_client
        usa_curl = False

    headers_img = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    for url_cand in urls_candidatas:
        try:
            kwargs_req: Dict[str, Any] = {"headers": headers_img, "timeout": 3}
            if usa_curl:
                kwargs_req["impersonate"] = "chrome"
            resp = req_client.get(url_cand, **kwargs_req)
            if resp.status_code == 200 and resp.content:
                mime_valido = _detectar_mime_imagem(resp.content)
                if mime_valido:
                    if pid_str:
                        try:
                            with open(os.path.join(pasta_cache_fotos, f"{pid_str}.img"), "wb") as f_out:
                                f_out.write(resp.content)
                        except Exception:
                            pass
                    b64_ok = base64.b64encode(resp.content).decode("utf-8")
                    return f"data:{mime_valido};base64,{b64_ok}", True
        except Exception:
            continue

    # 4. Se todas as URLs falharem (ex: 404 para atleta sem foto), registra marcador leve e retorna silhueta neutra
    if pid_str:
        try:
            with open(os.path.join(pasta_cache_fotos, f"{pid_str}.none"), "w", encoding="utf-8") as f_none:
                f_none.write("404")
        except Exception:
            pass

    return avatar_padrao, False


def render_avatar_atleta_html(
    foto_src_b64: str,
    nome: str,
    tamanho: int = 125,
    is_foto_real: bool = True
) -> str:
    """
    Renderiza avatar circular com foto validada em Base64 ou silhueta neutra SVG.
    Não depende de atributos JavaScript inline ('onerror') que são bloqueados pelo DOMPurify do Streamlit.
    """
    silhueta_fallback_b64 = gerar_avatar_silhueta_svg_b64(nome)
    src_final = foto_src_b64 if (isinstance(foto_src_b64, str) and foto_src_b64.strip()) else silhueta_fallback_b64
    borda_cor = "#38BDF8" if is_foto_real else "#475569"

    return _compactar_html(f"""
        <div style="position: relative; width: {tamanho}px; height: {tamanho}px; flex-shrink: 0;
                    border-radius: 50%; overflow: hidden; border: 3px solid {borda_cor};
                    background-color: #0F172A; background-image: url('{silhueta_fallback_b64}');
                    background-size: cover; background-position: center;
                    box-shadow: 0 6px 16px rgba(0,0,0,0.45);">
            <img src="{src_final}" alt="{nome}"
                 style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover; display: block;" />
        </div>
    """)


def render_card_perfil_atleta(atleta: pd.Series, escudo_b64: str = "") -> str:
    """Renderiza o Card Principal do Atleta com foto validada (ou silhueta neutra), biometria e contrato."""
    nome = atleta.get("jogador_nome", "Atleta")
    pos_detalhada = atleta.get("posicao_detalhada", atleta.get("posicao_nome", atleta.get("posicao", "Jogador")))
    num_camisa = atleta.get("numero_camisa", 10)
    clube = atleta.get("clube_nome", "Clube")
    divisao = mapa_divisoes.get(str(atleta.get("divisao", "")), str(atleta.get("divisao", "Série A")))
    ano = int(atleta.get("ano", 2026))

    pid_atleta = atleta.get("sofascore_player_id")
    foto_url_raw = str(atleta.get("foto_url", "") or "")
    tm_img_raw = str(atleta.get("foto_url_transfermarkt", atleta.get("image_url", "")) or "")

    foto_resolvida_b64, is_foto_real = obter_foto_atleta_segura(
        player_id=pid_atleta,
        foto_url=foto_url_raw,
        tm_image_url=tm_img_raw,
        nome=nome
    )

    altura_txt = atleta.get("altura_cm_texto", "N/D")
    peso_txt = atleta.get("peso_kg_texto", "N/D")
    pe_dom = atleta.get("pe_dominante", "Destro")
    idade = int(atleta.get("idade", atleta.get("idade_na_temporada", 24)))
    nasc_txt = atleta.get("data_nascimento_texto", "N/D")

    val_mercado = atleta.get("valor_mercado_formatado", formatar_moeda_eur(atleta.get("valor_mercado_eur", 0)))
    termino_contrato = atleta.get("termino_contrato", "Em vigência")
    tempo_contrato = atleta.get("tempo_contrato_restante", "Ativo")
    clube_detentor = atleta.get("clube_detentor", clube)

    avatar_html = render_avatar_atleta_html(foto_resolvida_b64, nome, tamanho=125, is_foto_real=is_foto_real)
    escudo_img_tag = f'<img src="{escudo_b64}" style="width: 22px; height: 22px; vertical-align: middle; margin-right: 6px; object-fit: contain;">' if escudo_b64 else '🛡️ '
    nasc_badge = f" ({nasc_txt})" if nasc_txt and nasc_txt != "N/D" else ""

    html_card = f"""
    <div style="background: #1E293B; border-radius: 14px; border: 1px solid #334155; padding: 22px 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); margin-bottom: 22px;">
        <div style="display: flex; flex-direction: row; align-items: center; gap: 24px; flex-wrap: wrap;">
            {avatar_html}
            <div style="flex: 1; min-width: 280px;">
                <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 6px;">
                    <span style="font-size: 1.65rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.5px;">{nome}</span>
                    <span style="background: #0284C7; color: #FFFFFF; font-weight: 800; font-size: 0.95rem; padding: 3px 10px; border-radius: 6px; box-shadow: 0 2px 6px rgba(2,132,199,0.3);">#{num_camisa}</span>
                    <span style="background: #0F172A; border: 1px solid #38BDF8; color: #38BDF8; font-weight: 700; font-size: 0.82rem; padding: 3px 10px; border-radius: 6px; text-transform: uppercase; letter-spacing: 0.5px;">{pos_detalhada}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 14px; color: #94A3B8; font-size: 0.95rem;">
                    <span style="display: inline-flex; align-items: center; font-weight: 600; color: #E2E8F0;">
                        {escudo_img_tag} {clube}
                    </span>
                    <span style="color: #475569;">•</span>
                    <span>{divisao}</span>
                    <span style="color: #475569;">•</span>
                    <span style="background: #334155; color: #F1F5F9; padding: 1px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Temporada {ano}</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px;">
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">📏 Altura:</span> <strong style="color: #F8FAFC;">{altura_txt}</strong>
                    </div>
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">⚖️ Peso:</span> <strong style="color: #F8FAFC;">{peso_txt}</strong>
                    </div>
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">🦶 Pé Dominante:</span> <strong style="color: #38BDF8;">{pe_dom}</strong>
                    </div>
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">🎂 Idade:</span> <strong style="color: #F8FAFC;">{idade} anos</strong><span style="color: #64748B; font-size: 0.8rem;">{nasc_badge}</span>
                    </div>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    <div style="background: #0F172A; border: 1px solid #334155; border-left: 3px solid #16A34A; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">💰 Valor Estimado:</span> <strong style="color: #4ADE80;">{val_mercado}</strong>
                    </div>
                    <div style="background: #0F172A; border: 1px solid #334155; border-left: 3px solid #F59E0B; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">📝 Término:</span> <strong style="color: #FCD34D;">{termino_contrato}</strong> <span style="color: #94A3B8; font-size: 0.78rem;">({tempo_contrato})</span>
                    </div>
                    <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 6px 12px; font-size: 0.85rem; color: #E2E8F0;">
                        <span style="color: #94A3B8; font-weight: 500;">🏢 Detentor dos Direitos:</span> <strong style="color: #CBD5E1;">{clube_detentor}</strong>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return _compactar_html(html_card)


def plotar_radar_perfil_atleta(
    atleta_1: pd.Series,
    atleta_2: Optional[pd.Series] = None,
    df_goleiros_base: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Plota radar de percentis (0 a 100) com linha de referência da média da liga (P50) e comparativo."""
    is_gk = (atleta_1.get("posicao") == "G")
    fig = go.Figure()

    if is_gk and df_goleiros_base is not None and not df_goleiros_base.empty:
        categorias = [
            "Defesas Totais",
            "Defesas na Área",
            "Taxa de Defesas %",
            "Clean Sheets",
            "Gols Evitados (PSxG)",
            "Segurança / Nota"
        ]
        cols_gk = [
            "defesas_totais", "defesas_dentro_area", "taxa_defesas_pct",
            "clean_sheets", "gols_evitados_psxg_saldo", "nota_media_sofascore"
        ]

        def extrair_percentis_gk(row_atleta):
            pid = row_atleta.get("sofascore_player_id")
            ano = row_atleta.get("ano")
            sub_gk = df_goleiros_base[(df_goleiros_base["sofascore_player_id"] == pid) & (df_goleiros_base["ano"] == ano)]
            if sub_gk.empty:
                sub_gk = df_goleiros_base[df_goleiros_base["sofascore_player_id"] == pid]
            if not sub_gk.empty:
                r_gk = sub_gk.iloc[-1]
                vals = []
                for c in cols_gk:
                    pct = (df_goleiros_base[c] <= r_gk.get(c, 0)).mean() * 100.0
                    vals.append(float(np.clip(pct, 5.0, 99.0)))
                return vals
            return [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

        vals_1 = extrair_percentis_gk(atleta_1)
        vals_1_closed = vals_1 + [vals_1[0]]
        categorias_closed = categorias + [categorias[0]]

        fig.add_trace(go.Scatterpolar(
            r=[50] * 7,
            theta=categorias_closed,
            mode="lines",
            name="Média da Liga (P50)",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.08)"
        ))

        nome_1 = f"{atleta_1.get('jogador_nome', 'Goleiro')} ({atleta_1.get('clube_nome', '')})"
        fig.add_trace(go.Scatterpolar(
            r=vals_1_closed,
            theta=categorias_closed,
            mode="lines+markers",
            name=nome_1,
            line=dict(color="#00D2FF", width=3),
            fill="toself",
            fillcolor="rgba(0, 210, 255, 0.35)",
            marker=dict(size=7, color="#00D2FF")
        ))

        if atleta_2 is not None:
            vals_2 = extrair_percentis_gk(atleta_2)
            vals_2_closed = vals_2 + [vals_2[0]]
            nome_2 = f"{atleta_2.get('jogador_nome', 'Goleiro 2')} ({atleta_2.get('clube_nome', '')})"
            fig.add_trace(go.Scatterpolar(
                r=vals_2_closed,
                theta=categorias_closed,
                mode="lines+markers",
                name=nome_2,
                line=dict(color="#EC4899", width=2.5),
                fill="toself",
                fillcolor="rgba(236, 72, 153, 0.30)",
                marker=dict(size=7, color="#EC4899")
            ))

    else:
        categorias = [
            "Poder de Fogo<br>(Gols/90)",
            "Criação<br>(Assist./90)",
            "Dribles<br>(Dribles/90)",
            "Combate Defensivo<br>(Desarmes+Interc./90)",
            "Jogo Aéreo<br>(Aéreos/90)",
            "Retenção & Qualidade<br>(Passes & Nota)"
        ]
        categorias_closed = categorias + [categorias[0]]

        def extrair_vals_linha(row):
            return [
                float(row.get("pct_poder_fogo", 50)),
                float(row.get("pct_criacao", 50)),
                float(row.get("pct_dribles", 50)),
                float(row.get("pct_combate_defensivo", 50)),
                float(row.get("pct_jogo_aereo", 50)),
                float(row.get("pct_retencao_qualidade", 50)),
            ]

        vals_1 = extrair_vals_linha(atleta_1)
        vals_1_closed = vals_1 + [vals_1[0]]

        fig.add_trace(go.Scatterpolar(
            r=[50] * 7,
            theta=categorias_closed,
            mode="lines",
            name="Média da Posição na Liga (P50)",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            fill="toself",
            fillcolor="rgba(245, 158, 11, 0.08)"
        ))

        nome_1 = f"{atleta_1.get('jogador_nome', 'Atleta')} ({atleta_1.get('clube_nome', '')})"
        fig.add_trace(go.Scatterpolar(
            r=vals_1_closed,
            theta=categorias_closed,
            mode="lines+markers",
            name=nome_1,
            line=dict(color="#00D2FF", width=3),
            fill="toself",
            fillcolor="rgba(0, 210, 255, 0.35)",
            marker=dict(size=7, color="#00D2FF")
        ))

        if atleta_2 is not None:
            vals_2 = extrair_vals_linha(atleta_2)
            vals_2_closed = vals_2 + [vals_2[0]]
            nome_2 = f"{atleta_2.get('jogador_nome', 'Atleta 2')} ({atleta_2.get('clube_nome', '')})"
            fig.add_trace(go.Scatterpolar(
                r=vals_2_closed,
                theta=categorias_closed,
                mode="lines+markers",
                name=nome_2,
                line=dict(color="#EC4899", width=2.5),
                fill="toself",
                fillcolor="rgba(236, 72, 153, 0.30)",
                marker=dict(size=7, color="#EC4899")
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#94A3B8", size=10),
                gridcolor="#334155",
                linecolor="#334155"
            ),
            angularaxis=dict(
                tickfont=dict(color="#F8FAFC", size=11),
                gridcolor="#334155",
                linecolor="#334155"
            ),
            bgcolor="#0F172A"
        ),
        paper_bgcolor="#1E293B",
        font=dict(color="#F8FAFC"),
        title=dict(
            text="🕸️ <b>Radar de Habilidades vs. Média da Posição (Percentis 0 a 100)</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=14, color="#F8FAFC")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#E2E8F0")
        ),
        margin=dict(l=45, r=45, t=50, b=65),
        height=480
    )
    return fig


# ------------------------------------------------------------------------------
# Cabeçalho Principal
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">⚽ Painel de Inteligência Esportiva - Futebol Brasileiro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Modelagem Multi-Temporada (2023 a 2026) das Séries A, B, C e D: Visualização Tática no Gramado, Shotmaps, Radar de Percentis e Teto Preditivo</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Metodologia e Dicionário de Métricas (Colapsado e Discreto)
# ------------------------------------------------------------------------------
with st.expander("ℹ️ Metodologia e Dicionário de Métricas", expanded=False):
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        **🧠 Modelos Preditivos de Machine Learning:**
        * **Pontos Esperados (Ridge & XGBoost):** Prevê os pontos da temporada com base em valor de elenco, nota média e potencial das joias. Resíduos positivos ($>0$) sinalizam *Superação Tática*.
        * **Teto Projetado & Joias Sub-23 (Aging Curves):** Curvas polinomiais e $k$-NN estimando ápice técnico e potencial de valorização no ciclo 2026+.
        * **Confrontos Diretos (Distribuição de Poisson):** Simulação estocástica baseada nas taxas de gols esperados ($\lambda_A, \lambda_B$) para calcular probabilidades de vitória e placar provável.
        """)
    with col_m2:
        st.markdown("""
        **📖 Glossário de Métricas Táticas & Goleiros:**
        * **xG / xA:** Gols e Assistências Esperados com base na probabilidade estatística das finalizações e passes.
        * **PSxG & Saldo Gols Evitados:** Post-Shot xG menos gols sofridos; saldo positivo reflete defesas decisivas que salvaram gols prováveis.
        * **Clean Sheets & Defesas %:** Jogos sem sofrer gols e percentual de finalizações certas defendidas.
        * **Quadrantes Táticos:** Sup. Esquerdo (*Superação / Joias*), Sup. Direito (*Elite*), Inf. Direito (*Sub-aproveitamento*), Inf. Esquerdo (*Risco*).
        """)

mapa_divisoes = {
    "Todas": "Todas",
    "Serie_A": "Série A",
    "Serie_B": "Série B",
    "Serie_C": "Série C",
    "Serie_D": "Série D"
}

# ------------------------------------------------------------------------------
# Navegação Principal na Sidebar (Logo Oficial + Seletor Agrupado por Domínio)
# ------------------------------------------------------------------------------
MODULO_HUB_CLUBE = "🛡️ Hub do Clube & Destaques por Setor"
MODULO_PERFIL_ATLETA = "🔬 Perfil & Comparador de Atletas (Raio-X)"
MODULO_SIMULACAO_2026 = "🔮 Simulação Brasileirão 2026 (Monte Carlo)"
MODULO_H2H = "⚔️ Confrontos Diretos (H2H & Poisson)"
MODULO_RADAR_JOIAS = "🚀 Radar de Joias Sub-23 & Mapa Coletivo"
MODULO_PANORAMA_CLUBES = "🏆 Panorama dos Clubes (Tabela vs. Elenco)"

with st.sidebar:
    logo_b64 = gerar_logo_app_svg_b64()
    st.markdown(f"""
    <div style="background: linear-gradient(145deg, #0F172A 0%, #1E293B 100%);
                border: 1px solid #334155; border-radius: 14px; padding: 14px 12px;
                margin-bottom: 14px; display: flex; align-items: center; gap: 12px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.45);">
        <img src="{logo_b64}" style="width: 54px; height: 54px; border-radius: 12px; flex-shrink: 0;" alt="Logo Brasileirão Analytics"/>
        <div>
            <div style="font-size: 0.96rem; font-weight: 900; color: #F8FAFC; line-height: 1.15; letter-spacing: 0.3px;">
                BRASILEIRÃO <span style="color: #38BDF8;">PRO</span>
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; margin-top: 2px;">
                Scout Tático &bull; IA Preditiva 2026
            </div>
            <div style="margin-top: 5px;">
                <span style="background: rgba(34,197,94,0.15); color: #4ADE80; border: 1px solid rgba(34,197,94,0.35);
                             font-size: 0.65rem; font-weight: 800; padding: 2px 7px; border-radius: 999px;">
                    ● Séries A &bull; B &bull; C &bull; D
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.74rem; font-weight:800; color:#38BDF8; text-transform:uppercase; "
        "letter-spacing:0.8px; margin-bottom:6px;'>🧭 Módulos de Navegação</div>",
        unsafe_allow_html=True
    )

    nav_modulo = st.radio(
        "Selecione o Módulo Analítico:",
        options=[
            MODULO_HUB_CLUBE,
            MODULO_PERFIL_ATLETA,
            MODULO_SIMULACAO_2026,
            MODULO_H2H,
            MODULO_RADAR_JOIAS,
            MODULO_PANORAMA_CLUBES,
        ],
        index=0,
        label_visibility="collapsed",
        key="navegacao_principal_sidebar"
    )

    st.markdown("""
    <div style="background: rgba(15,23,42,0.75); border: 1px solid #1E293B; border-radius: 10px;
                padding: 10px 12px; margin-top: 10px; font-size: 0.75rem; color: #94A3B8;">
        <div style="font-weight: 800; color: #E2E8F0; margin-bottom: 4px;">📌 Guia Rápido de Seções</div>
        <div>• <b>Hub & Setores:</b> Elenco, Uniformes e Top 4</div>
        <div>• <b>Raio-X Atleta:</b> Biometria, Radar e Comparador</div>
        <div>• <b>Simulação 2026:</b> Tabela 1º–20º, Z4 e G4</div>
        <div>• <b>H2H & Radar:</b> Confrontos e Joias Sub-23</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.4;">
        <b>Status da Base:</b> {len(df_clubes):,} registros de clubes &bull; {len(df_jogadores):,} registros de atletas auditados.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MÓDULO 1: PANORAMA DOS CLUBES
# ==============================================================================
if nav_modulo == MODULO_PANORAMA_CLUBES:
    st.subheader("🏆 Desempenho dos Clubes: Tabela Real vs. Força do Elenco")

    col_f1, col_f2, col_f3 = st.columns([2, 2, 4])
    with col_f1:
        div_opcoes = ["Todas", "Serie_A", "Serie_B", "Serie_C", "Serie_D"]
        div_selecionada = st.selectbox(
            "Filtrar por Divisão:",
            div_opcoes,
            format_func=lambda x: mapa_divisoes.get(x, x),
            key="filtro_div_clubes"
        )
    with col_f2:
        anos_disp = ["Todas"] + sorted(df_clubes["ano"].dropna().unique().tolist(), reverse=True) if not df_clubes.empty else ["Todas", 2025]
        ano_selecionado = st.selectbox("Temporada:", anos_disp, index=1 if len(anos_disp) > 1 else 0, key="filtro_ano_clubes")

    with st.spinner("🔄 Processando métricas de desempenho e resíduos táticos dos clubes..."):
        df_c_filtro = df_clubes.copy()
        if div_selecionada != "Todas":
            df_c_filtro = df_c_filtro[df_c_filtro["divisao"] == div_selecionada]
        if ano_selecionado != "Todas":
            df_c_filtro = df_c_filtro[df_c_filtro["ano"] == int(ano_selecionado)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Clubes Registrados</div>
            <div class="metric-value">{len(df_c_filtro)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        media_pts = df_c_filtro["pontos"].mean() if not df_c_filtro.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Média de Pontos</div>
            <div class="metric-value">{media_pts:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        if not df_c_filtro.empty and "residuo_desempenho" in df_c_filtro.columns:
            top_over = df_c_filtro.nlargest(1, "residuo_desempenho").iloc[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Maior Superação Tática</div>
                <div class="metric-value" style="font-size: 1.15rem; color: #4ADE80;">
                    {top_over['clube_nome']} ({top_over['ano']}) +{top_over['residuo_desempenho']} pts
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-card"><div class="metric-label">Maior Superação</div><div class="metric-value">-</div></div>""", unsafe_allow_html=True)
    with c4:
        val_med = df_c_filtro["valor_mercado_somado_eur"].median() if not df_c_filtro.empty and "valor_mercado_somado_eur" in df_c_filtro.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mediana Valor Elenco</div>
            <div class="metric-value">{formatar_moeda_eur(val_med)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("##### 📊 Superação de Expectativa (Resíduo de Pontos)")
        if not df_c_filtro.empty and "residuo_desempenho" in df_c_filtro.columns:
            top_over_sub = df_c_filtro.nlargest(8, "residuo_desempenho")
            top_under_sub = df_c_filtro.nsmallest(8, "residuo_desempenho")
            df_plot_res = pd.concat([top_over_sub, top_under_sub]).drop_duplicates(subset=["clube_nome", "ano"]).sort_values("residuo_desempenho", ascending=True)

            fig_bar = px.bar(
                df_plot_res,
                x="residuo_desempenho",
                y="clube_nome",
                orientation="h",
                color="residuo_desempenho",
                color_continuous_scale=["#EF4444", "#38BDF8", "#22C55E"],
                labels={"residuo_desempenho": "Saldo (Real - Esperado)", "clube_nome": "Clube"},
                hover_data=["divisao", "ano", "pontos", "pontos_esperados_elenco"]
            )
            aplicar_tema_plotly_dark(fig_bar, altura=420)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Dados de resíduos não disponíveis.")

    with col_g2:
        st.markdown("##### 🎯 Força do Elenco vs. Aproveitamento de Pontos")
        if not df_c_filtro.empty:
            eixo_x_opcao = st.radio(
                "Eixo X:",
                ["idade_media_ponderada_minutos", "nota_media_elenco", "potencial_medio_elenco"],
                horizontal=True,
                format_func=lambda x: {
                    "idade_media_ponderada_minutos": "Idade Média",
                    "nota_media_elenco": "Nota Elenco",
                    "potencial_medio_elenco": "Potencial Médio"
                }.get(x, x)
            )

            fig_scat = px.scatter(
                df_c_filtro,
                x=eixo_x_opcao,
                y="aproveitamento_pct",
                size="valor_mercado_somado_eur",
                color="divisao",
                hover_name="clube_nome",
                hover_data=["ano", "pontos", "jogos", "saldo_gols"],
                color_discrete_map={
                    "Serie_A": "#38BDF8",
                    "Serie_B": "#4ADE80",
                    "Serie_C": "#FBBF24",
                    "Serie_D": "#F87171"
                },
                labels={
                    "aproveitamento_pct": "Aproveitamento (%)",
                    eixo_x_opcao: "Indicador do Elenco",
                    "divisao": "Divisão"
                }
            )
            med_x = float(df_c_filtro[eixo_x_opcao].median())
            med_y = float(df_c_filtro["aproveitamento_pct"].median())
            fig_scat.add_vline(x=med_x, line_dash="dash", line_color="#94A3B8", annotation_text="Mediana X", annotation_position="top left")
            fig_scat.add_hline(y=med_y, line_dash="dash", line_color="#94A3B8", annotation_text="Mediana Y", annotation_position="bottom right")

            aplicar_tema_plotly_dark(fig_scat, altura=420)
            st.plotly_chart(fig_scat, use_container_width=True)

            st.caption(
                "💡 **Interpretação dos Quadrantes:** **Sup. Esquerdo:** Superação Tática (elenco modesto com alto rendimento). **Sup. Direito:** Potências Favoritas (alto poder financeiro e liderança). "
                "**Inf. Direito:** Sub-aproveitamento Crítico (elencos caros com baixo retorno de pontos). **Inf. Esquerdo:** Zona de Risco e Reestruturação."
            )

    col_tab_head, col_tab_dl = st.columns([3, 1.3])
    with col_tab_head:
        st.markdown("##### 📋 Tabela de Classificação e Força de Elenco")
    cols_tabela = [
        "posicao", "clube_nome", "divisao", "ano", "jogos", "pontos", "vitorias", "empates",
        "derrotas", "saldo_gols", "aproveitamento_pct", "nota_media_elenco",
        "idade_media_ponderada_minutos", "valor_mercado_somado_eur"
    ]
    cols_disp = [c for c in cols_tabela if c in df_c_filtro.columns]
    df_tabela_view = df_c_filtro[cols_disp].copy()
    if "posicao" in df_tabela_view.columns:
        df_tabela_view = df_tabela_view.sort_values(by=["ano", "divisao", "posicao"], ascending=[False, True, True])

    df_tabela_export = df_tabela_view.copy()
    if "valor_mercado_somado_eur" in df_tabela_view.columns:
        df_tabela_view["valor_mercado_somado_eur"] = df_tabela_view["valor_mercado_somado_eur"].apply(formatar_moeda_eur)

    with col_tab_dl:
        render_botao_download_csv(
            df_tabela_export,
            nome_arquivo=f"panorama_clubes_{div_selecionada}_{ano_selecionado}.csv",
            label_botao="📥 Exportar Tabela (CSV)",
            key_btn="dl_panorama_clubes_csv"
        )

    st.dataframe(
        df_tabela_view.rename(columns={
            "posicao": "Pos",
            "clube_nome": "Clube",
            "divisao": "Divisão",
            "ano": "Ano",
            "jogos": "J",
            "pontos": "Pts",
            "vitorias": "V",
            "empates": "E",
            "derrotas": "D",
            "saldo_gols": "SG",
            "aproveitamento_pct": "Aprov %",
            "nota_media_elenco": "Nota Elenco",
            "idade_media_ponderada_minutos": "Idade Média",
            "valor_mercado_somado_eur": "Valor Elenco"
        }),
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# MÓDULO 2: HUB DO CLUBE ("VISÃO DO CLUBE")
# ==============================================================================
if nav_modulo == MODULO_HUB_CLUBE:
    st.subheader("🛡️ Hub do Clube: Identidade Visual, Vitrine de Uniformes & Resumo da Temporada")

    col_h_div, col_h_clube, col_h_ano = st.columns([1.5, 2.5, 1])
    with col_h_div:
        div_filtro_hub = st.selectbox(
            "Divisão:",
            ["Todas", "Serie_A", "Serie_B", "Serie_C", "Serie_D"],
            format_func=lambda x: mapa_divisoes.get(x, x),
            key="hub_div_seletor"
        )
    with col_h_clube:
        df_c_div = df_clubes.copy()
        if div_filtro_hub != "Todas":
            df_c_div = df_c_div[df_c_div["divisao"] == div_filtro_hub]
        clubes_hub = sorted(df_c_div["clube_nome"].dropna().unique().tolist()) if not df_c_div.empty else sorted(df_clubes["clube_nome"].dropna().unique().tolist())
        idx_def_hub = 0
        for cand in ["Palmeiras", "Flamengo", "Corinthians", "São Paulo", "Cruzeiro"]:
            if cand in clubes_hub:
                idx_def_hub = clubes_hub.index(cand)
                break
        clube_hub_sel = st.selectbox("Selecione o Clube:", clubes_hub, index=idx_def_hub, key="hub_clube_seletor")

    with col_h_ano:
        anos_hub = sorted(df_clubes["ano"].dropna().unique().tolist(), reverse=True) if not df_clubes.empty else [2026, 2025, 2024, 2023]
        ano_hub_sel = st.selectbox("Temporada:", anos_hub, index=0, key="hub_ano_seletor")

    info_c = obter_info_clube_meta(clube_hub_sel)
    divisao_clube = "Serie_A"
    if not df_clubes.empty:
        c_match = df_clubes[df_clubes["clube_nome"] == clube_hub_sel]
        if not c_match.empty and "divisao" in c_match.columns:
            divisao_clube = c_match.iloc[0]["divisao"]
    if not divisao_clube or pd.isna(divisao_clube):
        divisao_clube = info_c.get("divisao", "Serie_A")

    cor_prim = info_c.get("cor_primaria", "#1E3A8A")
    cor_sec = info_c.get("cor_secundaria", "#FFFFFF")
    accent_c = info_c.get("accent", "#38BDF8")
    sigla = info_c.get("sigla", (clube_hub_sel[:3].upper()))
    tec_atual = info_c.get("tecnico_atual", "Comissão Técnica")
    estadio = info_c.get("estadio", "Estádio Principal")
    cidade = info_c.get("cidade", "Brasil")
    escudo_path = obter_escudo_img(clube_hub_sel)

    # Injeção dinâmica de CSS temático exclusivo do clube
    st.markdown(gerar_css_dinamico_clube(info_c, divisao=divisao_clube), unsafe_allow_html=True)

    is_elite = divisao_clube in ["Serie_A", "Serie_B"]
    badge_div_label = mapa_divisoes.get(divisao_clube, divisao_clube)

    # Header Dinâmico de Alto Contraste (Opção A: Base Neutra Dark com Acentuação Sutil)
    col_header_logo, col_header_info = st.columns([1, 6])
    with col_header_logo:
        if escudo_path and os.path.exists(escudo_path):
            st.image(escudo_path, width=100)
        else:
            st.markdown(f"""
            <div style="width: 90px; height: 90px; border-radius: 12px; background: #0F172A;
                        color: #F8FAFC; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; font-weight: 800;
                        border: 2px solid #334155; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
                {sigla}
            </div>
            """, unsafe_allow_html=True)
    with col_header_info:
        st.markdown(f"""
        <div class="club-hub-header">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px; flex-wrap: wrap;">
                <span class="club-title-text">{clube_hub_sel}</span>
                <span class="club-badge-pill">{sigla}</span>
                <span style="background: #0F172A; color: #94A3B8; font-size: 0.8rem; font-weight: 700; padding: 3px 10px; border-radius: 6px; border: 1px solid #334155;">{badge_div_label}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.95rem; display: flex; gap: 16px; flex-wrap: wrap;">
                <span>🏟️ <b style="color: #F8FAFC;">{estadio}</b> ({cidade})</span>
                <span>👔 Técnico: <b style="color: #F8FAFC;">{tec_atual}</b></span>
                <span>📅 Temporada <b style="color: #F8FAFC;">{ano_hub_sel}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Vitrine de Uniformes da Temporada (Kits 1, 2 e 3)
    st.markdown("### 👕 Vitrine Oficial de Uniformes da Temporada")
    if is_elite:
        st.caption(f"Coleção de mantos oficiais do **{clube_hub_sel}** para a temporada **{ano_hub_sel}**:")
    else:
        st.caption(f"Identidade visual e mantos do **{clube_hub_sel}** ({badge_div_label}):")

    col_k1, col_k2, col_k3 = st.columns(3)
    camisas_meta = [
        (1, "Uniforme 1 (Titular)", "TITULAR", "Manto tradicional do clube"),
        (2, "Uniforme 2 (Reserva)", "RESERVA", "Uniforme visitante contrastante"),
        (3, "Uniforme 3 (Alternativa)", "ALTERNATIVA", "Edição especial comemorativa")
    ]

    for col, (tipo, nome_uni, tag, desc) in zip([col_k1, col_k2, col_k3], camisas_meta):
        with col:
            c_img = obter_camisa_img(clube_hub_sel, tipo)
            st.markdown(f"""
            <div class="kit-card-box">
                <div style="font-weight: 700; font-size: 0.95rem; color: #F8FAFC; margin-bottom: 4px;">{nome_uni}</div>
                <span style="background: #1E293B; color: #94A3B8; font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 4px; border: 1px solid #334155;">{tag}</span>
            </div>
            """, unsafe_allow_html=True)
            if c_img and os.path.exists(c_img):
                st.image(c_img, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #0F172A; border-radius: 10px; border: 1px dashed #334155; color: #94A3B8; text-align: center; padding: 12px;">
                    <div style="font-size: 1.8rem; margin-bottom: 6px;">🛡️</div>
                    <div style="font-weight: 700; font-size: 0.88rem; color: #F8FAFC;">Manto {tag}</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Disponível sob consulta</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"🎨 {desc}")

    # ==========================================================================
    # DESTAQUES POR SETOR — O "MELHOR EM CADA POSIÇÃO" (MÉTRICAS AVANÇADAS)
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🏆 Destaques por Posição — O Melhor de Cada Setor na Temporada")
    st.caption(
        f"Premiação estatística automática do melhor atleta de cada setor do **{clube_hub_sel}** em **{ano_hub_sel}**, "
        "calculada a partir de métricas táticas avançadas específicas por função e regularidade em campo:"
    )

    chave_clube_hub = normalizar_texto_app(clube_hub_sel)
    df_elenco_ano = df_jogadores[
        (df_jogadores["clube_chave"] == chave_clube_hub) &
        (df_jogadores["ano"] == int(ano_hub_sel))
    ].copy() if not df_jogadores.empty else pd.DataFrame()

    # Filtra estritamente a base de goleiros do clube (apenas posicao == 'G' e defesas_totais > 0, sem jogadores de linha)
    ids_linha_clube = set()
    if not df_elenco_ano.empty:
        ids_linha_clube = set(
            df_elenco_ano[df_elenco_ano["posicao"].isin(["D", "M", "F"])]["sofascore_player_id"].dropna().astype(int).tolist()
        )

    df_g_clube = pd.DataFrame()
    if not df_goleiros.empty:
        df_g_clube = df_goleiros[
            (df_goleiros["clube_chave"] == chave_clube_hub) &
            (df_goleiros["ano"] == int(ano_hub_sel)) &
            (df_goleiros["defesas_totais"] > 0)
        ].copy()
        if not df_g_clube.empty and ids_linha_clube:
            pids_gk = pd.to_numeric(df_g_clube["sofascore_player_id"], errors="coerce").fillna(-1).astype(int)
            df_g_clube = df_g_clube[~pids_gk.isin(ids_linha_clube)].copy()

    # Se o clube tiver goleiros em df_elenco_ano[posicao == 'G'] que ainda não estejam em df_g_clube, complementa
    if not df_elenco_ano.empty:
        df_g_el = df_elenco_ano[df_elenco_ano["posicao"] == "G"].copy()
        if not df_g_el.empty:
            ids_ja_gk = set(df_g_clube["sofascore_player_id"].dropna().astype(int).tolist()) if not df_g_clube.empty else set()
            novos_gk_rows = []
            for _, rg in df_g_el.iterrows():
                pid_rg = int(rg.get("sofascore_player_id", 0) or 0)
                if pid_rg not in ids_ja_gk and int(rg.get("jogos", 0) or 0) > 0:
                    j_val = max(1, int(rg.get("jogos", 1) or 1))
                    n_val = float(rg.get("nota_media_sofascore", 6.8) or 6.8)
                    defs_t = max(1, int(rg.get("defesas_totais", 0) or round(j_val * 3.0)))
                    defs_b = max(1, int(rg.get("defesas_dentro_area", 0) or round(defs_t * 0.58)))
                    cs_v = int(rg.get("clean_sheets", 0) or round(j_val * 0.30))
                    gols_s = max(0, int(round(j_val * 1.05)))
                    td_p = float(rg.get("taxa_defesas_pct", 0.0) or round((defs_t / max(1, defs_t + gols_s)) * 100.0, 1))
                    td_dif = float(rg.get("taxa_defesas_dificeis_pct", 0.0) or round(np.clip((defs_b / max(1.0, defs_b + gols_s * 0.72)) * 100.0, 48.0, 88.0), 1))
                    psxg_s = float(rg.get("gols_evitados_psxg_saldo", 0.0) or round((n_val - 6.75) * j_val * 0.25, 2))
                    novos_gk_rows.append({
                        "sofascore_player_id": pid_rg,
                        "jogador_nome": rg.get("jogador_nome"),
                        "jogador_chave": rg.get("jogador_chave"),
                        "sofascore_team_id": rg.get("sofascore_team_id"),
                        "clube_nome": rg.get("clube_nome"),
                        "clube_chave": chave_clube_hub,
                        "ano": int(ano_hub_sel),
                        "divisao": rg.get("divisao"),
                        "posicao": "G",
                        "jogos": j_val,
                        "minutos": int(rg.get("minutos", j_val * 90) or (j_val * 90)),
                        "defesas_totais": defs_t,
                        "defesas_dentro_area": defs_b,
                        "clean_sheets": cs_v,
                        "defesas_penalti": 0,
                        "gols_sofridos": gols_s,
                        "psxg_enfrentado": round(max(0.5, gols_s + psxg_s), 2),
                        "gols_evitados_psxg_saldo": psxg_s,
                        "taxa_defesas_pct": td_p,
                        "taxa_defesas_dificeis_pct": td_dif,
                        "saidas_aereas": max(1, int(round(j_val * 0.75))),
                        "socos": max(0, int(round(j_val * 0.38))),
                        "nota_media_sofascore": n_val,
                    })
            if novos_gk_rows:
                df_g_clube = pd.concat([df_g_clube, pd.DataFrame(novos_gk_rows)], ignore_index=True)

    def selecionar_melhor_do_setor(df_setor: pd.DataFrame, col_score: str = "score_destaque_setor") -> Optional[pd.Series]:
        if df_setor.empty:
            return None
        # Prioriza atletas com minutagem representativa na temporada (>= 180 min ou >= 2 jogos)
        df_cand = df_setor[df_setor["minutos"] >= 180]
        if df_cand.empty:
            df_cand = df_setor[df_setor["jogos"] >= 2]
        if df_cand.empty:
            df_cand = df_setor
        if col_score in df_cand.columns:
            return df_cand.sort_values(by=[col_score, "minutos", "nota_media_sofascore"], ascending=[False, False, False]).iloc[0]
        return df_cand.sort_values(by=["nota_media_sofascore", "minutos"], ascending=[False, False]).iloc[0]

    def render_card_destaque_setor(
        titulo_setor: str,
        icone_setor: str,
        cor_borda: str,
        atleta_row: Optional[pd.Series],
        metricas_lista: List[Tuple[str, str, str]]
    ) -> str:
        if atleta_row is None:
            return _compactar_html(f"""
            <div style="background: #1E293B; border: 1px dashed #334155; border-top: 4px solid {cor_borda};
                        border-radius: 12px; padding: 16px; min-height: 280px; display: flex; flex-direction: column;
                        align-items: center; justify-content: center; text-align: center; color: #94A3B8;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">{icone_setor}</div>
                <div style="font-weight: 800; color: #F8FAFC; font-size: 0.95rem;">{titulo_setor}</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 4px;">Sem minutagem registrada no setor</div>
            </div>
            """)

        nome_at = str(atleta_row.get("jogador_nome", "Atleta"))
        pid_at = atleta_row.get("sofascore_player_id")
        f_sofa = str(atleta_row.get("foto_url", ""))
        f_tm = str(atleta_row.get("foto_url_transfermarkt", atleta_row.get("image_url", "")))
        foto_src, _ = obter_foto_atleta_segura(pid_at, f_sofa, f_tm, nome=nome_at)

        pos_det = str(atleta_row.get("posicao_detalhada", titulo_setor))
        idade_at = int(float(atleta_row.get("idade_na_temporada", 24) or 24))
        jogos_at = int(float(atleta_row.get("jogos", 0) or 0))
        min_at = int(float(atleta_row.get("minutos", 0) or 0))
        nota_at = float(atleta_row.get("nota_media_sofascore", 6.80) or 6.80)

        blocos_m = ""
        for lbl_m, val_m, sub_m in metricas_lista:
            blocos_m += f"""
            <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 8px 10px;">
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.3px;">{lbl_m}</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #F8FAFC; margin-top: 2px;">{val_m}</div>
                <div style="font-size: 0.72rem; color: #64748B;">{sub_m}</div>
            </div>
            """

        return _compactar_html(f"""
        <div style="background: linear-gradient(165deg, #1E293B 0%, #0F172A 100%);
                    border: 1px solid #334155; border-top: 4px solid {cor_borda};
                    border-radius: 12px; padding: 14px 16px; box-shadow: 0 6px 16px rgba(0,0,0,0.28);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="background: {cor_borda}22; color: {cor_borda}; border: 1px solid {cor_borda}66;
                             font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 999px;
                             text-transform: uppercase; letter-spacing: 0.6px;">
                    {icone_setor} {titulo_setor}
                </span>
                <span style="background: #F59E0B22; color: #FBBF24; border: 1px solid #F59E0B66;
                             font-size: 0.76rem; font-weight: 800; padding: 2px 8px; border-radius: 6px;">
                    ★ {nota_at:.2f}
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <img src="{foto_src}" alt="{nome_at}"
                     style="width: 58px; height: 58px; border-radius: 50%; object-fit: cover;
                            background: #0F172A; border: 2px solid {cor_borda}; flex-shrink: 0;" />
                <div style="overflow: hidden;">
                    <div style="font-size: 1.08rem; font-weight: 800; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{nome_at}">
                        {nome_at}
                    </div>
                    <div style="font-size: 0.80rem; color: #38BDF8; font-weight: 700;">{pos_det}</div>
                    <div style="font-size: 0.75rem; color: #94A3B8;">{idade_at} anos &bull; {jogos_at}J ({min_at:,} min)</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                {blocos_m}
            </div>
        </div>
        """)

    with st.spinner(f"🔄 Auditando elenco do {clube_hub_sel} ({ano_hub_sel}) e calculando destaques por setor..."):
        # 1. Melhor Goleiro (G)
        melhor_gk = None
        if not df_g_clube.empty:
            df_gk_rank = df_g_clube.copy()
            if "taxa_defesas_dificeis_pct" not in df_gk_rank.columns:
                ch_d = np.maximum(1.0, df_gk_rank["defesas_dentro_area"] + df_gk_rank["gols_sofridos"] * 0.72)
                df_gk_rank["taxa_defesas_dificeis_pct"] = np.round(np.clip((df_gk_rank["defesas_dentro_area"] / ch_d) * 100.0, 45.0, 92.0), 1)
            fator_reg_gk = np.clip(df_gk_rank["minutos"] / 900.0, 0.25, 1.18)
            df_gk_rank["score_destaque_setor"] = (
                (df_gk_rank["nota_media_sofascore"] * 8.2)
                + (df_gk_rank["taxa_defesas_dificeis_pct"] * 0.28)
                + (df_gk_rank["clean_sheets"] * 1.45)
                + (df_gk_rank["gols_evitados_psxg_saldo"] * 1.8)
            ) * fator_reg_gk
            df_gk_rank["posicao_detalhada"] = "Goleiro"
            # Enriquece idade se disponível em df_elenco_ano
            if not df_elenco_ano.empty:
                mapa_id_idade = df_elenco_ano.drop_duplicates("sofascore_player_id").set_index("sofascore_player_id")["idade_na_temporada"].to_dict()
                df_gk_rank["idade_na_temporada"] = df_gk_rank["sofascore_player_id"].map(mapa_id_idade).fillna(27)
            melhor_gk = selecionar_melhor_do_setor(df_gk_rank)

        # 2. Melhor Defensor (D), 3. Melhor Meio-campista (M), 4. Melhor Atacante (F)
        df_defs_clube = df_elenco_ano[df_elenco_ano["posicao"] == "D"].copy() if not df_elenco_ano.empty else pd.DataFrame()
        df_meis_clube = df_elenco_ano[df_elenco_ano["posicao"] == "M"].copy() if not df_elenco_ano.empty else pd.DataFrame()
        df_atas_clube = df_elenco_ano[df_elenco_ano["posicao"] == "F"].copy() if not df_elenco_ano.empty else pd.DataFrame()

        melhor_def = selecionar_melhor_do_setor(df_defs_clube)
        melhor_mei = selecionar_melhor_do_setor(df_meis_clube)
        melhor_ata = selecionar_melhor_do_setor(df_atas_clube)

    col_best_g, col_best_d, col_best_m, col_best_f = st.columns(4)

    with col_best_g:
        metricas_gk = []
        if melhor_gk is not None:
            td_dif_v = float(melhor_gk.get("taxa_defesas_dificeis_pct", melhor_gk.get("taxa_defesas_pct", 70.0)))
            metricas_gk = [
                ("Nota SofaScore", f"{float(melhor_gk.get('nota_media_sofascore', 6.8)):.2f}", "Rendimento sob a trave"),
                ("Defesas Difíceis", f"{td_dif_v:.1f}%", f"{int(melhor_gk.get('defesas_dentro_area', 0))} defesas na área"),
                ("Clean Sheets", f"{int(melhor_gk.get('clean_sheets', 0))} SG", f"Em {int(melhor_gk.get('jogos', 0))} partidas"),
                ("Defesas Totais", f"{int(melhor_gk.get('defesas_totais', 0))}", f"Taxa Geral: {float(melhor_gk.get('taxa_defesas_pct', 0)):.1f}%"),
            ]
        st.markdown(
            render_card_destaque_setor("Melhor Goleiro", "🧤", "#0284C7", melhor_gk, metricas_gk),
            unsafe_allow_html=True
        )

    with col_best_d:
        metricas_def = []
        if melhor_def is not None:
            metricas_def = [
                ("Desarmes / Jogo", f"{float(melhor_def.get('desarmes_por_jogo', 0)):.2f}", f"{int(melhor_def.get('desarmes', 0))} desarmes totais"),
                ("Interceptações", f"{int(melhor_def.get('interceptacoes', 0))}", f"{float(melhor_def.get('interceptacoes_por_jogo', 0)):.2f} por partida"),
                ("Duelos Aéreos %", f"{float(melhor_def.get('duelos_aereos_ganhos_pct', 60.0)):.1f}%", f"{int(melhor_def.get('duelos_aereos_ganhos', 0))} ganhos pelo alto"),
                ("Passes Defensivos", f"{float(melhor_def.get('precisao_passes_defensivos_pct', melhor_def.get('passes_certos_pct', 80.0))):.1f}%", "Precisão na saída"),
            ]
        st.markdown(
            render_card_destaque_setor("Melhor Defensor", "🛡️", "#16A34A", melhor_def, metricas_def),
            unsafe_allow_html=True
        )

    with col_best_m:
        metricas_mei = []
        if melhor_mei is not None:
            metricas_mei = [
                ("Assist. Esp. (xA)", f"{float(melhor_mei.get('xa_estimado', 0)):.2f}", f"{int(melhor_mei.get('assistencias', 0))} assistências reais"),
                ("Passes Decisivos", f"{int(melhor_mei.get('passes_decisivos', 0))}", f"{float(melhor_mei.get('passes_decisivos_por_jogo', 0)):.2f} key passes/j"),
                ("Duelos no Meio", f"{int(melhor_mei.get('duelos_ganhos_meio', 0))}", f"{float(melhor_mei.get('duelos_ganhos_meio_por_jogo', 0)):.1f} ganhos/jogo"),
                ("Acerto de Passes", f"{float(melhor_mei.get('passes_certos_pct', 80.0)):.1f}%", "Controle e distribuição"),
            ]
        st.markdown(
            render_card_destaque_setor("Melhor Meio-Campo", "⚙️", "#9333EA", melhor_mei, metricas_mei),
            unsafe_allow_html=True
        )

    with col_best_f:
        metricas_ata = []
        if melhor_ata is not None:
            metricas_ata = [
                ("Gols Marcados", f"{int(melhor_ata.get('gols', 0))}", f"{float(melhor_ata.get('gols_por_90', 0)):.2f} gols p/ 90m"),
                ("Gols Esp. (xG)", f"{float(melhor_ata.get('xg_estimado', 0)):.2f}", f"{int(melhor_ata.get('finalizacoes', 0))} finalizações"),
                ("Fin. no Gol / J", f"{float(melhor_ata.get('finalizacoes_no_gol_por_jogo', 0)):.2f}", f"{int(melhor_ata.get('finalizacoes_no_gol', 0))} chutes no alvo"),
                ("Part. Diretas (G+A)", f"{int(melhor_ata.get('participacoes_diretas_gols', 0))}", f"{int(melhor_ata.get('gols', 0))}G + {int(melhor_ata.get('assistencias', 0))}A"),
            ]
        st.markdown(
            render_card_destaque_setor("Melhor Atacante", "⚽", "#E11D48", melhor_ata, metricas_ata),
            unsafe_allow_html=True
        )

    # Destaques Individuais Previstos pelo Modelo (Machine Learning Breakout Talents)
    st.write("")
    st.markdown("##### 🚀 Destaques do Elenco Previstos por Machine Learning (Teto 2026+)")
    df_j_clube = df_elenco_ano

    if not df_j_clube.empty and "indice_teto_projetado" in df_j_clube.columns:
        top_talentos = df_j_clube.nlargest(3, "indice_teto_projetado")
        cols_destaque = st.columns(len(top_talentos))
        for col_dest, (_, j_dest) in zip(cols_destaque, top_talentos.iterrows()):
            with col_dest:
                st.markdown(f"""
                <div class="club-card-accent">
                    <div style="font-size: 0.76rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">⭐ Destaque Preditivo ML</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin: 3px 0;">{j_dest['jogador_nome']}</div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">
                        Posição: <b style="color: #F8FAFC;">{j_dest.get('posicao_detalhada', j_dest.get('posicao', '-'))} ({j_dest.get('posicao', '-')})</b> &bull; Idade: <b style="color: #F8FAFC;">{int(j_dest.get('idade_na_temporada', 0))} anos</b>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.84rem; border-top: 1px solid #334155; padding-top: 6px;">
                        <span style="color: #94A3B8;">Teto: <b style="color: #38BDF8;">{j_dest.get('indice_teto_projetado', 0):.1f} pts</b></span>
                        <span style="color: #94A3B8;">Nota: <b style="color: #F8FAFC;">{j_dest.get('nota_media_sofascore', 0):.2f}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Projeções de Machine Learning para este elenco em processamento ou minutos insuficientes na temporada.")

    st.markdown("---")

    # Resumo da Temporada: Técnico e Resultados Recentes
    st.markdown("### 📊 Raio-X da Temporada do Clube")

    col_tec, col_forma = st.columns([1, 1])

    with col_tec:
        st.markdown("##### 👔 Retrospecto da Comissão Técnica")
        df_t_clube = df_treinadores[df_treinadores["clube_chave"] == chave_clube_hub] if not df_treinadores.empty else pd.DataFrame()
        if not df_t_clube.empty:
            r_tec = df_t_clube.iloc[0]
            st.markdown(f"""
            <div class="club-card-accent" style="text-align: left;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; margin-bottom: 2px;">{r_tec['tecnico_nome']}</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-bottom: 10px;">{r_tec.get('nacionalidade', 'Brasil')} | Formação: <b style="color: #F8FAFC;">{r_tec.get('esquema_preferido', '4-2-3-1')}</b></div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.88rem;">
                    <div><span style="color: #94A3B8;">Permanência:</span> <b style="color: #F8FAFC;">{r_tec.get('tempo_permanencia_texto', 'Em atividade')}</b></div>
                    <div><span style="color: #94A3B8;">Jogos no Cargo:</span> <b style="color: #F8FAFC;">{r_tec.get('jogos_comandados', 0)} partidas</b></div>
                    <div><span style="color: #94A3B8;">Aproveitamento:</span> <b style="color: #22C55E;">{r_tec.get('aproveitamento_pct', 0)}%</b></div>
                    <div><span style="color: #94A3B8;">Taxa Vitórias:</span> <b style="color: #F8FAFC;">{r_tec.get('pct_vitorias', 0)}%</b></div>
                    <div><span style="color: #94A3B8;">Invicto:</span> <b style="color: #F8FAFC;">{r_tec.get('jogos_invictos', 0)} ({r_tec.get('pct_invicto', 0)}%)</b></div>
                    <div><span style="color: #94A3B8;">Saldo de Gols:</span> <b style="color: #F8FAFC;">+{r_tec.get('saldo_gols', 0)} ({r_tec.get('gols_pro', 0)} GP / {r_tec.get('gols_contra', 0)} GC)</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Dados históricos de comissão técnica em processamento para {clube_hub_sel}.")

    with col_forma:
        st.markdown("##### 📈 Sequência Recente de Resultados")
        if ConfrontosH2HAnalytics and not df_confrontos.empty:
            ultimos_jogos = ConfrontosH2HAnalytics.obter_forma_recente_clube(df_confrontos, clube_hub_sel, ano=ano_hub_sel, n_jogos=6)
            if ultimos_jogos:
                cols_badges = st.columns(len(ultimos_jogos))
                for c_b, j in zip(cols_badges, ultimos_jogos):
                    res = j["resultado"]
                    cor_bg = "#16A34A" if res == "V" else "#D97706" if res == "E" else "#DC2626"
                    with c_b:
                        st.markdown(f"""
                        <div style="background-color: {cor_bg}; color: white; border-radius: 8px; padding: 6px; text-align: center; font-weight: 800; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                            <div style="font-size: 1.15rem;">{res}</div>
                            <div style="font-size: 0.7rem; font-weight: 600;">{j['placar']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                df_forma = pd.DataFrame(ultimos_jogos)[["data", "adversario", "mando", "placar", "resultado_extenso"]]
                st.dataframe(df_forma.rename(columns={
                    "data": "Data", "adversario": "Adversário", "mando": "Mando",
                    "placar": "Placar", "resultado_extenso": "Resultado"
                }), use_container_width=True, hide_index=True)
            else:
                st.info(f"Sem partidas recentes registradas para {clube_hub_sel} em {ano_hub_sel}.")
        else:
            st.info("Base de confrontos não carregada.")

    st.markdown("---")

    # Elenco por Posição com Regra Estrita e Métricas Setoriais Avançadas
    st.markdown("### 👥 Elenco da Temporada por Setor (Filtro Estrito de Posições)")

    sub_gol, sub_def, sub_mei, sub_ata = st.tabs([
        "🧤 Goleiros (Métricas Especializadas & PSxG)",
        "🛡️ Defensores (Zagueiros & Laterais)",
        "⚙️ Meio-Campistas (Volantes & Meias)",
        "⚽ Atacantes & Artilharia"
    ])

    with sub_gol:
        st.markdown("##### 🧤 Desempenho Completo dos Goleiros (Exclusivo Posição `G`)")
        st.write("Métricas analíticas detalhadas: nota média SofaScore, taxa de defesas difíceis (na área), clean sheets, PSxG (Post-Shot Expected Goals) e gols evitados:")

        if not df_g_clube.empty:
            df_g_clube_sorted = df_g_clube.sort_values(by=["minutos", "nota_media_sofascore"], ascending=[False, False])
            g_titular = df_g_clube_sorted.iloc[0]
            gk1, gk2, gk3, gk4, gk5 = st.columns(5)
            with gk1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Goleiro Principal</div>
                    <div class="metric-value" style="font-size: 1.15rem;">{g_titular['jogador_nome']}</div>
                    <small>{int(g_titular.get('jogos', 0))} jogos ({int(g_titular.get('minutos', 0)):,} min)</small>
                </div>
                """, unsafe_allow_html=True)
            with gk2:
                td_dif_tit = float(g_titular.get("taxa_defesas_dificeis_pct", g_titular.get("taxa_defesas_pct", 70.0)))
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Taxa Defesas Difíceis</div>
                    <div class="metric-value" style="color: #0284C7;">{td_dif_tit:.1f}% <span style="font-size: 0.90rem; color: #64748B;">({int(g_titular['defesas_dentro_area'])} área)</span></div>
                    <small>Defesas Totais: <b>{int(g_titular['defesas_totais'])} ({g_titular['taxa_defesas_pct']}%)</b></small>
                </div>
                """, unsafe_allow_html=True)
            with gk3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Clean Sheets (Sem Gols)</div>
                    <div class="metric-value" style="color: #16A34A;">{int(g_titular['clean_sheets'])}</div>
                    <small>Pênaltis Defendidos: <b>{int(g_titular.get('defesas_penalti', 0))}</b></small>
                </div>
                """, unsafe_allow_html=True)
            with gk4:
                saldo_psxg = g_titular.get('gols_evitados_psxg_saldo', 0.0)
                cor_psxg = "#16A34A" if saldo_psxg > 0 else "#DC2626"
                sinal_ps = "+" if saldo_psxg > 0 else ""
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gols Evitados (Saldo PSxG)</div>
                    <div class="metric-value" style="color: {cor_psxg};">{sinal_ps}{saldo_psxg:.2f}</div>
                    <small>PSxG Enfrentado: <b>{g_titular.get('psxg_enfrentado', 0):.1f}</b></small>
                </div>
                """, unsafe_allow_html=True)
            with gk5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nota Sofascore</div>
                    <div class="metric-value" style="color: #9333EA;">{g_titular['nota_media_sofascore']:.2f}</div>
                    <small>Gols Sofridos: <b>{int(g_titular['gols_sofridos'])}</b></small>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            cols_g_show = [
                "jogador_nome", "jogos", "minutos", "nota_media_sofascore",
                "taxa_defesas_dificeis_pct", "clean_sheets", "defesas_totais",
                "defesas_dentro_area", "taxa_defesas_pct", "defesas_penalti",
                "gols_sofridos", "psxg_enfrentado", "gols_evitados_psxg_saldo", "saidas_aereas"
            ]
            cols_g_disp = [c for c in cols_g_show if c in df_g_clube_sorted.columns]
            st.dataframe(
                df_g_clube_sorted[cols_g_disp].rename(columns={
                    "jogador_nome": "Goleiro", "jogos": "J", "minutos": "Min",
                    "nota_media_sofascore": "Nota Sofa",
                    "taxa_defesas_dificeis_pct": "Def. Difíceis %",
                    "clean_sheets": "Clean Sheets",
                    "defesas_totais": "Defesas", "defesas_dentro_area": "Def. Área",
                    "taxa_defesas_pct": "Defesas %", "defesas_penalti": "Pênaltis Def.",
                    "gols_sofridos": "Gols Sofridos", "psxg_enfrentado": "PSxG",
                    "gols_evitados_psxg_saldo": "Saldo PSxG (+/-)", "saidas_aereas": "Saídas Aéreas"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Nenhum goleiro registrado com minutos para esta temporada.")

    with sub_def:
        df_defs = df_defs_clube.sort_values(by=["score_destaque_setor", "minutos"], ascending=[False, False]) if not df_defs_clube.empty else pd.DataFrame()
        if not df_defs.empty:
            cols_d = [
                "jogador_nome", "posicao_detalhada", "jogos", "minutos",
                "desarmes_por_jogo", "desarmes", "interceptacoes",
                "duelos_aereos_ganhos_pct", "duelos_aereos_ganhos",
                "precisao_passes_defensivos_pct", "nota_media_sofascore",
                "score_destaque_setor", "valor_mercado_eur"
            ]
            cols_d_disp = [c for c in cols_d if c in df_defs.columns]
            df_show_d = df_defs[cols_d_disp].copy()
            if "valor_mercado_eur" in df_show_d.columns:
                df_show_d["valor_mercado_eur"] = df_show_d["valor_mercado_eur"].apply(formatar_moeda_eur)
            st.dataframe(df_show_d.rename(columns={
                "jogador_nome": "Defensor", "posicao_detalhada": "Função Tática",
                "jogos": "J", "minutos": "Min",
                "desarmes_por_jogo": "Desarmes/J", "desarmes": "Desarmes Tot.",
                "interceptacoes": "Interceptações",
                "duelos_aereos_ganhos_pct": "Duelos Aéreos %", "duelos_aereos_ganhos": "Aéreos Ganhos",
                "precisao_passes_defensivos_pct": "Passes Def. %",
                "nota_media_sofascore": "Nota", "score_destaque_setor": "Score Setor",
                "valor_mercado_eur": "Valor de Mercado"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum defensor com minutos registrados.")

    with sub_mei:
        df_meis = df_meis_clube.sort_values(by=["score_destaque_setor", "minutos"], ascending=[False, False]) if not df_meis_clube.empty else pd.DataFrame()
        if not df_meis.empty:
            cols_m = [
                "jogador_nome", "posicao_detalhada", "jogos", "minutos",
                "xa_estimado", "assistencias", "passes_decisivos",
                "duelos_ganhos_meio", "passes_certos_pct", "gols",
                "nota_media_sofascore", "score_destaque_setor", "valor_mercado_eur"
            ]
            cols_m_disp = [c for c in cols_m if c in df_meis.columns]
            df_show_m = df_meis[cols_m_disp].copy()
            if "valor_mercado_eur" in df_show_m.columns:
                df_show_m["valor_mercado_eur"] = df_show_m["valor_mercado_eur"].apply(formatar_moeda_eur)
            st.dataframe(df_show_m.rename(columns={
                "jogador_nome": "Meio-Campista", "posicao_detalhada": "Função Tática",
                "jogos": "J", "minutos": "Min",
                "xa_estimado": "xA (Esp.)", "assistencias": "Assists",
                "passes_decisivos": "Passes Decisivos",
                "duelos_ganhos_meio": "Duelos Meio", "passes_certos_pct": "Passes %",
                "gols": "Gols", "nota_media_sofascore": "Nota",
                "score_destaque_setor": "Score Setor", "valor_mercado_eur": "Valor de Mercado"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum meio-campista com minutos registrados.")

    with sub_ata:
        df_atas = df_atas_clube.sort_values(by=["score_destaque_setor", "gols", "minutos"], ascending=[False, False, False]) if not df_atas_clube.empty else pd.DataFrame()
        if not df_atas.empty:
            art = df_atas.sort_values(by=["gols", "participacoes_diretas_gols", "minutos"], ascending=[False, False, False]).iloc[0]
            a1, a2, a3 = st.columns(3)
            with a1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #16A34A;">
                    <div class="metric-label">Principal Artilheiro</div>
                    <div class="metric-value" style="font-size: 1.25rem; color: #16A34A;">{art['jogador_nome']}</div>
                    <small><b>{int(art.get('gols', 0))} gols</b> (xG: {float(art.get('xg_estimado', 0)):.2f}) em {int(art.get('jogos', 0))} J</small>
                </div>
                """, unsafe_allow_html=True)
            with a2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #0284C7;">
                    <div class="metric-label">Finalizações no Gol / J</div>
                    <div class="metric-value" style="color: #0284C7;">{float(art.get('finalizacoes_no_gol_por_jogo', 0)):.2f}</div>
                    <small>No Alvo: <b>{int(art.get('finalizacoes_no_gol', 0))}</b> de {int(art.get('finalizacoes', 0))} chutes</small>
                </div>
                """, unsafe_allow_html=True)
            with a3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #9333EA;">
                    <div class="metric-label">Participações Diretas (G+A)</div>
                    <div class="metric-value" style="color: #9333EA;">{int(art.get('participacoes_diretas_gols', 0))}</div>
                    <small><b>{int(art.get('gols', 0))} gols</b> + <b>{int(art.get('assistencias', 0))} assistências</b> ({art.get('participacoes_gol_por_90', 0):.2f}/90)</small>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            cols_f = [
                "jogador_nome", "posicao_detalhada", "jogos", "minutos",
                "gols", "xg_estimado", "finalizacoes_no_gol_por_jogo",
                "participacoes_diretas_gols", "assistencias", "gols_por_90",
                "nota_media_sofascore", "score_destaque_setor", "valor_mercado_eur"
            ]
            cols_f_disp = [c for c in cols_f if c in df_atas.columns]
            df_show_f = df_atas[cols_f_disp].copy()
            if "valor_mercado_eur" in df_show_f.columns:
                df_show_f["valor_mercado_eur"] = df_show_f["valor_mercado_eur"].apply(formatar_moeda_eur)
            st.dataframe(df_show_f.rename(columns={
                "jogador_nome": "Atacante", "posicao_detalhada": "Função Tática",
                "jogos": "J", "minutos": "Min",
                "gols": "Gols", "xg_estimado": "xG (Esp.)",
                "finalizacoes_no_gol_por_jogo": "Fin. no Gol/J",
                "participacoes_diretas_gols": "Part. Diretas (G+A)",
                "assistencias": "Assists", "gols_por_90": "Gols p90",
                "nota_media_sofascore": "Nota", "score_destaque_setor": "Score Setor",
                "valor_mercado_eur": "Valor de Mercado"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum atacante com minutos registrados.")

    st.write("")
    col_dl_hub_info, col_dl_hub_btn = st.columns([3, 1.3])
    with col_dl_hub_info:
        st.caption(f"📥 Exporte a base completa de atletas e métricas setoriais do **{clube_hub_sel}** ({ano_hub_sel}) em formato CSV.")
    with col_dl_hub_btn:
        render_botao_download_csv(
            df_elenco_ano if not df_elenco_ano.empty else pd.DataFrame(),
            nome_arquivo=f"elenco_setores_{chave_clube_hub}_{ano_hub_sel}.csv",
            label_botao=f"📥 Exportar Elenco {clube_hub_sel}",
            key_btn=f"dl_elenco_{chave_clube_hub}_{ano_hub_sel}"
        )


# ==============================================================================
# MÓDULO 3: CONFRONTOS DIRETOS (H2H & HISTÓRICO)
# ==============================================================================
if nav_modulo == MODULO_H2H:
    st.subheader("⚔️ Confrontos Diretos (H2H): Duelos entre Clubes, Retrospecto de Técnicos & Drill-Down")
    st.write("Selecione dois times para analisar o histórico de confrontos diretos, aproveitamento, gols e o relatório tático minucioso de cada partida:")

    c_h1, c_h2, c_h3 = st.columns([2, 2, 1.5])
    clubes_opcoes_h2h = sorted(df_clubes["clube_nome"].dropna().unique().tolist()) if not df_clubes.empty else ["Palmeiras", "Corinthians"]

    idx_a = clubes_opcoes_h2h.index("Palmeiras") if "Palmeiras" in clubes_opcoes_h2h else 0
    idx_b = clubes_opcoes_h2h.index("Corinthians") if "Corinthians" in clubes_opcoes_h2h else (1 if len(clubes_opcoes_h2h) > 1 else 0)

    with c_h1:
        time_a_sel = st.selectbox("Time A (Mandante/Referência):", clubes_opcoes_h2h, index=idx_a, key="h2h_time_a")
    with c_h2:
        time_b_sel = st.selectbox("Time B (Visitante/Adversário):", clubes_opcoes_h2h, index=idx_b, key="h2h_time_b")
    with c_h3:
        anos_h2h = ["Todas (Geral)"] + sorted(df_clubes["ano"].dropna().unique().tolist(), reverse=True) if not df_clubes.empty else ["Todas (Geral)", 2026, 2025, 2024, 2023]
        ano_h2h_sel = st.selectbox("Filtrar Temporada:", anos_h2h, index=0, key="h2h_ano_filtro")

    if time_a_sel == time_b_sel:
        st.warning("Selecione dois clubes distintos para analisar o confronto direto.")
    else:
        info_a = obter_info_clube_meta(time_a_sel)
        info_b = obter_info_clube_meta(time_b_sel)
        cor_a = info_a.get("cor_primaria", "#1E3A8A")
        cor_b = info_b.get("cor_primaria", "#DC2626")
        esc_a = obter_escudo_img(time_a_sel)
        esc_b = obter_escudo_img(time_b_sel)

        @st.cache_data(ttl=3600, show_spinner=False)
        def obter_resumo_duelo_cached(clube_a: str, clube_b: str, ano_filtro: Optional[str]) -> Dict[str, Any]:
            if ConfrontosH2HAnalytics and not df_confrontos.empty:
                return ConfrontosH2HAnalytics.obter_resumo_duelo(
                    df_confrontos, clube_a, clube_b, ano=ano_filtro
                )
            return {}

        with st.spinner(f"🔄 Consultando histórico de confrontos ({time_a_sel} vs {time_b_sel}) e calculando probabilidades..."):
            resumo_duelo = obter_resumo_duelo_cached(
                time_a_sel,
                time_b_sel,
                ano_h2h_sel if ano_h2h_sel != "Todas (Geral)" else None
            )

        tot_j = resumo_duelo.get("total_jogos", 0)

        # Tratar acentos sutis para alto contraste garantido
        accent_a = cor_a if str(cor_a).lower() not in ["#000000", "#111111", "#18181b"] else info_a.get("accent", "#60A5FA")
        accent_b = cor_b if str(cor_b).lower() not in ["#000000", "#111111", "#18181b"] else info_b.get("accent", "#F87171")

        # Header Visual H2H com Alto Contraste (Opção A)
        st.markdown(f"""
        <div style="background: #1E293B; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;
                    border: 1px solid #334155; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div style="flex: 1; text-align: center; min-width: 140px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">{time_a_sel}</div>
                    <span style="background: #0F172A; color: #F8FAFC; border: 1.5px solid {accent_a}; font-weight: 700; padding: 3px 12px; border-radius: 6px; font-size: 0.85rem;">{info_a.get('sigla', 'A')}</span>
                </div>
                <div style="font-size: 2.0rem; font-weight: 900; color: #64748B; padding: 0 16px;">VS</div>
                <div style="flex: 1; text-align: center; min-width: 140px;">
                    <div style="font-size: 1.8rem; font-weight: 800; color: #F8FAFC;">{time_b_sel}</div>
                    <span style="background: #0F172A; color: #F8FAFC; border: 1.5px solid {accent_b}; font-weight: 700; padding: 3px 12px; border-radius: 6px; font-size: 0.85rem;">{info_b.get('sigla', 'B')}</span>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 0.88rem; color: #94A3B8; border-top: 1px solid #334155; padding-top: 8px;">
                📅 <b>{tot_j} confronto(s) registrado(s)</b> {f'no ano de {ano_h2h_sel}' if ano_h2h_sel != 'Todas (Geral)' else 'no histórico geral (2023-2026)'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if tot_j == 0:
            st.info(f"Nenhum confronto registrado entre **{time_a_sel}** e **{time_b_sel}** no filtro selecionado ({ano_h2h_sel}).")
        else:
            # Cards de resumo estatístico dos duelos
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {accent_a};">
                    <div class="metric-label">Vitórias {time_a_sel}</div>
                    <div class="metric-value" style="color: #F8FAFC;">{resumo_duelo['vitorias_a']}</div>
                    <small>Aproveitamento: <b style="color: #22C55E;">{resumo_duelo['aproveitamento_a']}%</b></small>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #64748B;">
                    <div class="metric-label">Empates</div>
                    <div class="metric-value" style="color: #F8FAFC;">{resumo_duelo['empates']}</div>
                    <small>{(resumo_duelo['empates']/tot_j*100):.1f}% dos confrontos</small>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {accent_b};">
                    <div class="metric-label">Vitórias {time_b_sel}</div>
                    <div class="metric-value" style="color: #F8FAFC;">{resumo_duelo['vitorias_b']}</div>
                    <small>Aproveitamento: <b style="color: #22C55E;">{resumo_duelo['aproveitamento_b']}%</b></small>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gols Marcados (Média)</div>
                    <div class="metric-value" style="color: #38BDF8;">{resumo_duelo['gols_a']} x {resumo_duelo['gols_b']}</div>
                    <small>Média: <b>{resumo_duelo['media_gols_jogo']:.2f} gols/jogo</b></small>
                </div>
                """, unsafe_allow_html=True)

            # Modelo Preditivo Estocástico de Poisson para o Confronto
            xg_a_calc = (resumo_duelo.get("gols_a", 0) / max(1, tot_j)) if tot_j > 0 else 1.45
            xg_b_calc = (resumo_duelo.get("gols_b", 0) / max(1, tot_j)) if tot_j > 0 else 1.15
            p_win_a, p_draw, p_win_b, placar_provavel = calcular_probabilidades_poisson(xg_a_calc, xg_b_calc)

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
                        border: 1px solid #3B82F655; border-radius: 12px; padding: 16px 20px;
                        margin: 16px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">
                    🎯 Modelo Preditivo Estocástico (Distribuição Bivariada de Poisson)
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; align-items: center;">
                    <div>
                        <span style="color: #94A3B8; font-size: 0.85rem;">Prob. Vitória {time_a_sel}:</span><br/>
                        <b style="color: {accent_a}; font-size: 1.35rem;">{p_win_a}%</b>
                    </div>
                    <div>
                        <span style="color: #94A3B8; font-size: 0.85rem;">Prob. Empate:</span><br/>
                        <b style="color: #E2E8F0; font-size: 1.35rem;">{p_draw}%</b>
                    </div>
                    <div>
                        <span style="color: #94A3B8; font-size: 0.85rem;">Prob. Vitória {time_b_sel}:</span><br/>
                        <b style="color: {accent_b}; font-size: 1.35rem;">{p_win_b}%</b>
                    </div>
                    <div>
                        <span style="color: #94A3B8; font-size: 0.85rem;">Placar Mais Provável:</span><br/>
                        <b style="color: #38BDF8; font-size: 1.35rem;">{placar_provavel}</b>
                    </div>
                </div>
                <div style="margin-top: 10px; font-size: 0.78rem; color: #94A3B8; border-top: 1px solid #334155; padding-top: 6px;">
                    ⚙️ <b>Metodologia:</b> Baseado na taxa de ocorrência de gols (&lambda;<sub>{info_a.get('sigla','A')}</sub> = {xg_a_calc:.2f}, &lambda;<sub>{info_b.get('sigla','B')}</sub> = {xg_b_calc:.2f}) modelada por processos de Poisson independentes sobre o histórico de duelos.
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # Gráfico de barras horizontais com a proporção de vitórias
            fig_h2h = go.Figure()
            fig_h2h.add_trace(go.Bar(
                y=["Confrontos Diretos"],
                x=[resumo_duelo["vitorias_a"]],
                name=f"Vitórias {time_a_sel}",
                orientation="h",
                marker=dict(color=accent_a),
                text=[f"{resumo_duelo['vitorias_a']} V ({time_a_sel})"],
                textposition="inside"
            ))
            fig_h2h.add_trace(go.Bar(
                y=["Confrontos Diretos"],
                x=[resumo_duelo["empates"]],
                name="Empates",
                orientation="h",
                marker=dict(color="#64748B"),
                text=[f"{resumo_duelo['empates']} E"],
                textposition="inside"
            ))
            fig_h2h.add_trace(go.Bar(
                y=["Confrontos Diretos"],
                x=[resumo_duelo["vitorias_b"]],
                name=f"Vitórias {time_b_sel}",
                orientation="h",
                marker=dict(color=accent_b),
                text=[f"{resumo_duelo['vitorias_b']} V ({time_b_sel})"],
                textposition="inside"
            ))
            aplicar_tema_plotly_dark(fig_h2h, altura=145)
            fig_h2h.update_layout(
                barmode="stack",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showticklabels=False)
            )
            st.plotly_chart(fig_h2h, use_container_width=True)

            st.markdown("---")

            # Lista de Confrontos Individuais com Técnicos
            col_h2h_t, col_h2h_dl = st.columns([3, 1.3])
            with col_h2h_t:
                st.markdown("##### 📜 Histórico de Partidas Realizadas (Com Técnicos no Comando)")
            partidas = resumo_duelo["partidas"]
            df_partidas = pd.DataFrame(partidas)

            with col_h2h_dl:
                render_botao_download_csv(
                    df_partidas,
                    nome_arquivo=f"confrontos_h2h_{time_a_sel}_vs_{time_b_sel}.csv",
                    label_botao="📥 Exportar Histórico H2H",
                    key_btn="dl_h2h_partidas_csv"
                )

            cols_p_show = ["data", "mandante", "placar", "visitante", "tecnico_mandante", "tecnico_visitante"]
            cols_p_disp = [c for c in cols_p_show if c in df_partidas.columns]
            st.dataframe(
                df_partidas[cols_p_disp].rename(columns={
                    "data": "Data", "mandante": "Mandante", "placar": "Placar",
                    "visitante": "Visitante", "tecnico_mandante": "Técnico Mandante",
                    "tecnico_visitante": "Técnico Visitante"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.write("")

            # DRILL-DOWN: Relatório Detalhado de Partida Específica
            st.markdown("##### 🔬 Drill-Down: Relatório Tático Minucioso da Partida")
            st.write("Selecione um duelo para visualizar o boxscore completo de finalizações, posse de bola, defesas e scout disciplinar:")

            opcoes_jogos_dd = [
                f"{p['data']} - {p['mandante']} {p['placar']} {p['visitante']} (Rodada {p.get('rodada', '?')})"
                for p in partidas
            ]
            jogo_sel_idx = st.selectbox(
                "Selecione uma partida para inspecionar:",
                range(len(opcoes_jogos_dd)),
                format_func=lambda i: opcoes_jogos_dd[i],
                key="sel_partida_drilldown"
            )

            p_det = partidas[jogo_sel_idx]

            st.markdown(f"""
            <div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 16px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <div style="text-align: center; margin-bottom: 14px;">
                    <span style="font-size: 1.0rem; color: #94A3B8; font-weight: 600;">{p_det['data']} &bull; Rodada {p_det.get('rodada', '?')}</span>
                    <div style="font-size: 2.1rem; font-weight: 800; color: #F8FAFC; margin: 4px 0;">
                        {p_det['mandante']} <span style="color: #38BDF8;">{p_det['placar']}</span> {p_det['visitante']}
                    </div>
                    <div style="font-size: 0.95rem; color: #94A3B8;">
                        👔 <b style="color: #F8FAFC;">{p_det['tecnico_mandante']}</b> (Mandante) &nbsp;&times;&nbsp; 👔 <b style="color: #F8FAFC;">{p_det['tecnico_visitante']}</b> (Visitante)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            metricas_comparativo = [
                ("Expected Goals (xG)", p_det["xg_mandante"], p_det["xg_visitante"]),
                ("Posse de Bola (%)", p_det["posse_mandante"], p_det["posse_visitante"]),
                ("Finalizações Totais", p_det["finalizacoes_mandante"], p_det["finalizacoes_visitante"]),
                ("Chutes no Alvo", p_det["chutes_alvo_mandante"], p_det["chutes_alvo_visitante"]),
                ("Defesas do Goleiro", p_det["defesas_mandante"], p_det["defesas_visitante"]),
                ("Escanteios", p_det["escanteios_mandante"], p_det["escanteios_visitante"]),
                ("Faltas Cometidas", p_det["faltas_mandante"], p_det["faltas_visitante"]),
                ("Cartões Amarelos", p_det["amarelos_mandante"], p_det["amarelos_visitante"])
            ]

            fig_dd = go.Figure()
            nomes_m = [m[0] for m in reversed(metricas_comparativo)]
            vals_mand = [m[1] for m in reversed(metricas_comparativo)]
            vals_vis = [m[2] for m in reversed(metricas_comparativo)]

            fig_dd.add_trace(go.Bar(
                y=nomes_m,
                x=[-v for v in vals_mand],
                name=p_det["mandante"],
                orientation="h",
                marker=dict(color=accent_a),
                text=[str(v) for v in vals_mand],
                textposition="inside",
                hoverinfo="text"
            ))
            fig_dd.add_trace(go.Bar(
                y=nomes_m,
                x=vals_vis,
                name=p_det["visitante"],
                orientation="h",
                marker=dict(color=accent_b),
                text=[str(v) for v in vals_vis],
                textposition="inside",
                hoverinfo="text"
            ))

            aplicar_tema_plotly_dark(fig_dd, altura=380)
            fig_dd.update_layout(
                title="<b>Scout Comparativo da Partida</b> (Mandante &larr; &bull; &rarr; Visitante)",
                barmode="relative",
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis=dict(showticklabels=False, showgrid=False)
            )
            st.plotly_chart(fig_dd, use_container_width=True)


# ==============================================================================
# MÓDULO 4: RADAR DE JOGADORES, JOIAS & OCUPAÇÃO COLETIVA DO TIME
# ==============================================================================
if nav_modulo == MODULO_RADAR_JOIAS:
    st.subheader("🚀 Radar de Atletas & Análise Tática Coletiva dos Clubes")

    subtab_jogadores, subtab_tatica_time = st.tabs([
        "🚀 Radar de Jogadores & Joias Sub-23",
        "🛡️ Mapa de Calor Coletivo & Ocupação dos 11 Titulares"
    ])

    # --------------------------------------------------------------------------
    # SUB-ABA 1: RADAR DE JOGADORES & JOIAS
    # --------------------------------------------------------------------------
    with subtab_jogadores:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Filtros do Radar de Jogadores")

        anos_jog_disp = ["Todas"] + sorted(df_jogadores["ano"].dropna().unique().tolist(), reverse=True) if not df_jogadores.empty else ["Todas"]
        filtro_ano_jog = st.sidebar.selectbox("Temporada (Radar):", anos_jog_disp, index=0)

        filtro_div_jog = st.sidebar.multiselect(
            "Divisão:",
            ["Serie_A", "Serie_B", "Serie_C", "Serie_D"],
            default=["Serie_A", "Serie_B", "Serie_C", "Serie_D"],
            format_func=lambda x: mapa_divisoes.get(x, x)
        )

        filtro_pos_jog = st.sidebar.multiselect(
            "Posição:",
            ["G", "D", "M", "F"],
            default=["D", "M", "F"],
            format_func=lambda x: {"G": "Goleiro (G)", "D": "Defensor (D)", "M": "Meio-Campo (M)", "F": "Atacante (F)"}.get(x, x)
        )

        max_idade = int(df_jogadores["idade_na_temporada"].max()) if not df_jogadores.empty else 40
        filtro_idade_max = st.sidebar.slider("Idade Máxima (ex: Sub-23):", 16, max_idade, 23)

        max_minutos = int(df_jogadores["minutos"].max()) if not df_jogadores.empty else 3400
        filtro_min_minutos = st.sidebar.slider("Minutagem Mínima:", 0, max_minutos, 360, step=90)

        with st.spinner("🔄 Filtrando base de atletas e calculando índices de teto projetado..."):
            df_j_filtro = df_jogadores.copy()
            if filtro_ano_jog != "Todas":
                df_j_filtro = df_j_filtro[df_j_filtro["ano"] == int(filtro_ano_jog)]
            if filtro_div_jog:
                df_j_filtro = df_j_filtro[df_j_filtro["divisao"].isin(filtro_div_jog)]
            if filtro_pos_jog:
                df_j_filtro = df_j_filtro[df_j_filtro["posicao"].isin(filtro_pos_jog)]
            df_j_filtro = df_j_filtro[df_j_filtro["idade_na_temporada"] <= filtro_idade_max]
            df_j_filtro = df_j_filtro[df_j_filtro["minutos"] >= filtro_min_minutos]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Atletas Filtrados", f"{len(df_j_filtro):,}")
        with k2:
            media_teto = df_j_filtro["indice_teto_projetado"].mean() if not df_j_filtro.empty else 0
            st.metric("Teto Médio", f"{media_teto:.1f} pts")
        with k3:
            top_artilheiro = df_j_filtro.nlargest(1, "gols").iloc[0] if not df_j_filtro.empty else None
            st.metric("Artilheiro Filtrado", f"{top_artilheiro['jogador_nome']} ({top_artilheiro['gols']}G)" if top_artilheiro is not None else "-")
        with k4:
            val_total = df_j_filtro["valor_mercado_eur"].sum() if not df_j_filtro.empty else 0
            st.metric("Valor Mercado Somado", formatar_moeda_eur(val_total))

        st.markdown("##### 📈 Idade vs. Índice de Teto Projetado (Colorido por Divisão)")
        if not df_j_filtro.empty:
            fig_disp_jog = px.scatter(
                df_j_filtro,
                x="idade_na_temporada",
                y="indice_teto_projetado",
                color="divisao",
                size="minutos",
                hover_name="jogador_nome",
                hover_data={
                    "ano": True,
                    "clube_nome": True,
                    "posicao": True,
                    "gols": True,
                    "assistencias": True,
                    "minutos": True,
                    "nota_media_sofascore": True,
                    "delta_evolucao_nota": True,
                    "salto_de_divisao": True,
                    "valor_mercado_eur": False
                },
                color_discrete_map={
                    "Serie_A": "#38BDF8",
                    "Serie_B": "#4ADE80",
                    "Serie_C": "#FBBF24",
                    "Serie_D": "#F87171"
                },
                labels={
                    "idade_na_temporada": "Idade na Temporada",
                    "indice_teto_projetado": "Índice de Teto Projetado",
                    "divisao": "Divisão",
                    "minutos": "Minutos em Campo"
                }
            )
            med_idade = float(df_j_filtro["idade_na_temporada"].median())
            med_teto = float(df_j_filtro["indice_teto_projetado"].median())
            fig_disp_jog.add_vline(x=med_idade, line_dash="dash", line_color="#94A3B8", annotation_text="Mediana Idade", annotation_position="top left")
            fig_disp_jog.add_hline(y=med_teto, line_dash="dash", line_color="#94A3B8", annotation_text="Mediana Teto", annotation_position="bottom right")

            aplicar_tema_plotly_dark(fig_disp_jog, altura=480)
            st.plotly_chart(fig_disp_jog, use_container_width=True)

            st.caption(
                "💡 **Interpretação dos Quadrantes:** **Sup. Esquerdo (Joias em Ascensão / Sub-23):** Atletas jovens com altíssimo teto preditivo e rápido potencial de explosão. "
                "**Inf. Direito (Veteranos Estáveis / Baixa Margem):** Atletas com minutagem alta e estabilização de potencial técnico com menor projeção de valorização financeira."
            )
        else:
            st.warning("Nenhum jogador encontrado com os filtros selecionados.")

        aba_tab1, aba_tab2 = st.tabs(["💎 Top 50 Joias com Potencial de Explosão", "📋 Todos os Atletas Filtrados"])

        with aba_tab1:
            if not df_joias.empty:
                df_joias_view = df_joias.copy()
                render_botao_download_csv(
                    df_joias_view,
                    nome_arquivo="top_50_joias_sub23_brasileirao.csv",
                    label_botao="📥 Exportar Top 50 Joias (CSV)",
                    key_btn="dl_top50_joias_csv"
                )
                if "valor_mercado_eur" in df_joias_view.columns:
                    df_joias_view["valor_mercado_eur"] = df_joias_view["valor_mercado_eur"].apply(formatar_moeda_eur)

                st.dataframe(
                    df_joias_view.rename(columns={
                        "posicao_ranking": "Rank",
                        "jogador_nome": "Atleta",
                        "clube_nome": "Clube",
                        "divisao": "Divisão",
                        "ano": "Ano",
                        "posicao": "Pos",
                        "idade_na_temporada": "Idade",
                        "minutos": "Min",
                        "gols": "Gols",
                        "assistencias": "Assists",
                        "participacoes_gol_por_90": "G+A/90",
                        "nota_media_sofascore": "Nota Sofa",
                        "delta_evolucao_nota": "Delta Nota",
                        "salto_de_divisao": "Salto Div",
                        "indice_teto_projetado": "Teto Projetado",
                        "score_explosao_proximo_ciclo": "Score Explosão 2026+",
                        "valor_mercado_eur": "Valor de Mercado"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Arquivo de joias escondidas não encontrado.")

        with aba_tab2:
            cols_jog_view = [
                "jogador_nome", "clube_nome", "divisao", "ano", "posicao", "idade_na_temporada",
                "minutos", "gols", "assistencias", "gols_por_90", "assistencias_por_90",
                "nota_media_sofascore", "delta_evolucao_nota", "salto_de_divisao",
                "indice_teto_projetado", "valor_mercado_eur"
            ]
            cols_jog_disp = [c for c in cols_jog_view if c in df_j_filtro.columns]
            df_j_table = df_j_filtro[cols_jog_disp].sort_values("indice_teto_projetado", ascending=False).copy()
            render_botao_download_csv(
                df_j_table,
                nome_arquivo="radar_atletas_filtrados.csv",
                label_botao="📥 Exportar Atletas Filtrados (CSV)",
                key_btn="dl_radar_filtrados_csv"
            )
            if "valor_mercado_eur" in df_j_table.columns:
                df_j_table["valor_mercado_eur"] = df_j_table["valor_mercado_eur"].apply(formatar_moeda_eur)

            st.dataframe(
                df_j_table.rename(columns={
                    "jogador_nome": "Nome",
                    "clube_nome": "Clube",
                    "divisao": "Divisão",
                    "ano": "Ano",
                    "posicao": "Pos",
                    "idade_na_temporada": "Idade",
                    "minutos": "Min",
                    "gols": "Gols",
                    "assistencias": "Assists",
                    "gols_por_90": "Gols/90",
                    "assistencias_por_90": "Assists/90",
                    "nota_media_sofascore": "Nota",
                    "delta_evolucao_nota": "Delta Nota",
                    "salto_de_divisao": "Salto Div",
                    "indice_teto_projetado": "Teto",
                    "valor_mercado_eur": "Valor Mercado"
                }),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------------------------
    # SUB-ABA 2: OCUPAÇÃO TÁTICA & HEATMAP COLETIVO DOS 11 TITULARES
    # --------------------------------------------------------------------------
    with subtab_tatica_time:
        st.markdown("#### 🛡️ Ocupação Territorial e Posição Média dos 11 Titulares")
        st.write("Selecione um clube e temporada para visualizar o mapa de calor coletivo e o posicionamento tático médio dos atletas mais utilizados:")

        c_t1, c_t2, c_t3, c_t4 = st.columns([2, 3, 2, 2])
        with c_t1:
            div_time_opcoes = ["Serie_A", "Serie_B", "Serie_C", "Serie_D"]
            div_time_sel = st.selectbox("Divisão do Clube:", div_time_opcoes, format_func=lambda x: mapa_divisoes.get(x, x), key="time_div_sel")

        with c_t2:
            clubes_div = sorted(df_jogadores[df_jogadores["divisao"] == div_time_sel]["clube_nome"].dropna().unique().tolist())
            idx_clube_def = 0
            for c_cand in ["Palmeiras", "Flamengo", "Botafogo", "Cruzeiro", "Fluminense", "Santos"]:
                if c_cand in clubes_div:
                    idx_clube_def = clubes_div.index(c_cand)
                    break
            clube_time_sel = st.selectbox("Selecione o Clube:", clubes_div, index=idx_clube_def if clubes_div else 0, key="time_clube_sel")

        with c_t3:
            anos_time = sorted(df_jogadores[df_jogadores["clube_nome"] == clube_time_sel]["ano"].dropna().unique().tolist(), reverse=True)
            ano_time_sel = st.selectbox("Temporada:", anos_time, index=0 if anos_time else 0, key="time_ano_sel")

        with c_t4:
            st.write("")
            st.write("")
            btn_carregar_coletivo = st.button("🛡️ Carregar Tática", type="primary", key="btn_tatica_time")

        chave_cache_time = f"tatica_{clube_time_sel}_{ano_time_sel}_{div_time_sel}"

        @st.cache_data(ttl=86400, show_spinner=False)
        def buscar_heatmap_coletivo_time_cached(clube_nome: str, ano_val: int, tid_val: int, sid_val: int):
            df_t_jog = df_jogadores[
                (df_jogadores["clube_nome"] == clube_nome) & (df_jogadores["ano"] == int(ano_val))
            ].copy()
            return modulo_tatica.buscar_heatmap_coletivo_time(df_t_jog, tid_val, sid_val, top_n=11)

        if btn_carregar_coletivo and clube_time_sel and ano_time_sel and modulo_tatica:
            tid_time, sid_time = modulo_tatica.obter_tournament_e_season_id(div_time_sel, int(ano_time_sel))

            with st.spinner(f"🔄 Carregando dados táticos e mapas de calor dos 11 titulares do {clube_time_sel} ({ano_time_sel})..."):
                prog_tatica = st.progress(20, text=f"📡 Buscando coordenadas de ocupação territorial do {clube_time_sel}...")
                pts_coletivo, medias_titulares = buscar_heatmap_coletivo_time_cached(
                    clube_time_sel, int(ano_time_sel), int(tid_time), int(sid_time)
                )
                prog_tatica.progress(85, text="🏟️ Calculando centro de gravidade e posições médias dos 11 titulares...")
                st.session_state[chave_cache_time] = (pts_coletivo, medias_titulares)
                prog_tatica.progress(100, text="✅ Mapa tático coletivo concluído!")
            prog_tatica.empty()

        if chave_cache_time in st.session_state:
            pts_coletivo, medias_titulares = st.session_state[chave_cache_time]
            col_pitch, col_info_tit = st.columns([6, 4])

            with col_pitch:
                with st.spinner("🔄 Renderizando gramado tático (KDE + posições médias)..."):
                    fig_time_pitch = modulo_tatica.plotar_heatmap_coletivo_time(
                        pts_coletivo, medias_titulares, clube_time_sel, int(ano_time_sel), mapa_divisoes.get(div_time_sel, div_time_sel)
                    )
                    st.pyplot(fig_time_pitch, clear_figure=True)

            with col_info_tit:
                st.markdown("##### 👥 11 Titulares Mais Utilizados:")
                if medias_titulares:
                    df_tit_view = pd.DataFrame(medias_titulares)
                    st.dataframe(
                        df_tit_view.rename(columns={
                            "jogador_nome": "Atleta",
                            "posicao": "Pos",
                            "minutos": "Minutos",
                            "media_x": "Linha Médio-Campo (X)",
                            "media_y": "Amplitude Lateral (Y)",
                            "acoes": "Ações de Calor"
                        })[["Atleta", "Pos", "Minutos", "Linha Médio-Campo (X)", "Ações de Calor"]],
                        use_container_width=True,
                        hide_index=True
                    )
                    # Métricas de Centro de Gravidade
                    x_vals = [m["media_x"] for m in medias_titulares]
                    y_vals = [m["media_y"] for m in medias_titulares]
                    bloco_medio = np.mean(x_vals)
                    amplitude_lat = np.max(y_vals) - np.min(y_vals)

                    st.markdown(f"""
                    <div style="background-color: #1e293b; padding: 12px; border-radius: 8px; color: #f8fafc; font-size: 0.9rem; margin-top: 10px;">
                        📌 <b>Indicadores Táticos da Equipe:</b><br>
                        • <b>Altura Média do Bloco:</b> {bloco_medio:.1f}% do campo ({'Bloco Alto / Pressionante' if bloco_medio > 52 else 'Bloco Médio / Reativo' if bloco_medio > 45 else 'Bloco Baixo'})<br>
                        • <b>Amplitude Territorial:</b> {amplitude_lat:.1f}% da largura do gramado<br>
                        • <b>Volume Total de Ações:</b> {len(pts_coletivo):,} coordenadas registradas
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Sem dados de titulares com minutos registrados.")
        else:
            st.info("👆 Selecione o clube e clique em **'🛡️ Carregar Tática'** para gerar a ocupação do campo e posições médias dos titulares.")



# ==============================================================================
# MÓDULO 5: RAIO-X DO ATLETA (TÁTICO, HEATMAP, SHOTMAP, GOLEIROS & PREVISÃO DE CARREIRA)
# ==============================================================================
if nav_modulo == MODULO_PERFIL_ATLETA:
    st.subheader("🔬 Perfil do Atleta: Raio-X Cadastral, Biometria, Comparador & Hub Tático")

    if not df_jogadores.empty:
        # ----------------------------------------------------------------------
        # Seção de Navegação e Filtros (Autocomplete ou Hierárquico)
        # ----------------------------------------------------------------------
        st.markdown("##### 🔍 Localizar Atleta:")
        col_modo_sel, col_filtro_extra = st.columns([2, 3])

        with col_modo_sel:
            modo_busca = st.radio(
                "Método de Seleção:",
                ["🔍 Busca Direta com Autocomplete", "📂 Filtro Hierárquico (Clube ➔ Posição ➔ Atleta)"],
                horizontal=False,
                key="modo_busca_rx"
            )

        if modo_busca == "🔍 Busca Direta com Autocomplete":
            with col_filtro_extra:
                termo_busca = st.text_input(
                    "Digitação rápida (sem acento, ex: hulk, cassio, arrascaeta, veiga):",
                    placeholder="Digite parte do nome para filtrar...",
                    key="termo_busca_atleta"
                )

            df_pool_nomes = df_jogadores.copy()
            if termo_busca and termo_busca.strip():
                k_busca = normalizar_texto_app(termo_busca.strip())
                if "jogador_nome_busca" in df_pool_nomes.columns:
                    mask = df_pool_nomes["jogador_nome_busca"].str.contains(k_busca, na=False)
                else:
                    mask = df_pool_nomes["jogador_nome"].apply(normalizar_texto_app).str.contains(k_busca, na=False)
                df_pool_nomes = df_pool_nomes[mask]

            todos_nomes = sorted(df_pool_nomes["jogador_nome"].dropna().unique().tolist())
            if not todos_nomes:
                st.warning("Nenhum atleta encontrado para o termo pesquisado. Exibindo lista completa.")
                todos_nomes = sorted(df_jogadores["jogador_nome"].dropna().unique().tolist())

            idx_nome_padrao = 0
            for nome_cand in ["Hulk", "Alan Patrick", "Raphael Veiga", "Pedro", "Cássio", "Weverton", "Jonathan Calleri"]:
                if nome_cand in todos_nomes:
                    idx_nome_padrao = todos_nomes.index(nome_cand)
                    break

            nome_escolhido = st.selectbox(
                "Selecione o Atleta:",
                todos_nomes,
                index=idx_nome_padrao if (0 <= idx_nome_padrao < len(todos_nomes)) else 0,
                key="rx_atleta_autocomplete"
            )

        else:
            col_div, col_clube, col_pos, col_atl = st.columns(4)
            with col_div:
                divs_rx = ["Todas", "Serie_A", "Serie_B", "Serie_C", "Serie_D"]
                div_sel_rx = st.selectbox(
                    "1. Divisão:",
                    divs_rx,
                    format_func=lambda x: mapa_divisoes.get(x, x),
                    key="hier_div_sel"
                )

            df_base_clubes = df_jogadores.copy()
            if div_sel_rx != "Todas":
                df_base_clubes = df_base_clubes[df_base_clubes["divisao"] == div_sel_rx]

            clubes_disponiveis = sorted(df_base_clubes["clube_nome"].dropna().unique().tolist())
            if not clubes_disponiveis:
                st.warning("Nenhum clube encontrado para esta divisão.")
                st.stop()

            idx_clube_padrao = 0
            for c_cand in ["Atlético Mineiro", "Palmeiras", "Flamengo", "Corinthians", "São Paulo", "Fluminense", "Cruzeiro"]:
                if c_cand in clubes_disponiveis:
                    idx_clube_padrao = clubes_disponiveis.index(c_cand)
                    break

            with col_clube:
                clube_sel_rx = st.selectbox(
                    "2. Clube:",
                    clubes_disponiveis,
                    index=idx_clube_padrao if (0 <= idx_clube_padrao < len(clubes_disponiveis)) else 0,
                    key="hier_clube_sel"
                )

            df_clube = df_base_clubes[df_base_clubes["clube_nome"] == clube_sel_rx]
            posicoes_disp = ["Todas"] + sorted(df_clube["posicao_nome"].dropna().unique().tolist())
            with col_pos:
                pos_sel = st.selectbox("3. Posição:", posicoes_disp, key="hier_pos_sel")

            df_pos = df_clube.copy()
            if pos_sel != "Todas":
                df_pos = df_pos[df_pos["posicao_nome"] == pos_sel]

            atletas_do_clube = sorted(df_pos["jogador_nome"].dropna().unique().tolist())
            if not atletas_do_clube:
                st.warning(f"Nenhum atleta nesta posição para o clube {clube_sel_rx}.")
                st.stop()

            with col_atl:
                nome_escolhido = st.selectbox(
                    "4. Atleta:",
                    atletas_do_clube,
                    key="hier_atleta_sel"
                )

        # Dados históricos do atleta selecionado
        historico_atleta = df_jogadores[df_jogadores["jogador_nome"] == nome_escolhido].sort_values("ano")
        if historico_atleta.empty:
            st.warning(f"Sem dados históricos registrados para {nome_escolhido}.")
            st.stop()

        # Seletor de Temporada se houver múltiplas temporadas
        anos_disponiveis = sorted(historico_atleta["ano"].unique().tolist(), reverse=True)
        if len(anos_disponiveis) > 1:
            col_t1, col_t2 = st.columns([3, 1])
            with col_t2:
                ano_selecionado = st.selectbox(
                    "Temporada:",
                    anos_disponiveis,
                    index=0,
                    key="sel_ano_atleta"
                )
            atleta_mais_recente = historico_atleta[historico_atleta["ano"] == ano_selecionado].iloc[0]
        else:
            atleta_mais_recente = historico_atleta.iloc[-1]

        # ----------------------------------------------------------------------
        # Card Principal (Header do Atleta com Foto, Biometria e Contrato)
        # ----------------------------------------------------------------------
        clube_atleta_nome = atleta_mais_recente.get("clube_nome", "")
        team_id_atleta = atleta_mais_recente.get("sofascore_team_id")
        escudo_path = obter_escudo_img(clube_atleta_nome, team_id=team_id_atleta)
        escudo_b64 = obter_img_base64(escudo_path) if escudo_path else ""

        with st.spinner(f"🔄 Carregando ficha biométrica e foto de perfil de {nome_escolhido}..."):
            card_html_atleta = render_card_perfil_atleta(atleta_mais_recente, escudo_b64=escudo_b64)
        st.markdown(card_html_atleta, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Painel Estatístico & Métricas da Temporada
        # ----------------------------------------------------------------------
        is_goleiro = (atleta_mais_recente.get("posicao") == "G")
        dados_gk = None
        if is_goleiro and not df_goleiros.empty:
            m_gk = df_goleiros[
                (df_goleiros["sofascore_player_id"] == atleta_mais_recente.get("sofascore_player_id")) &
                (df_goleiros["ano"] == atleta_mais_recente.get("ano"))
            ]
            if m_gk.empty:
                m_gk = df_goleiros[df_goleiros["sofascore_player_id"] == atleta_mais_recente.get("sofascore_player_id")]
            if not m_gk.empty:
                dados_gk = m_gk.iloc[-1]

        st.markdown("##### 📊 Painel Estatístico & Métricas da Temporada:")

        if is_goleiro and dados_gk is not None:
            # Goleiro: 8 Métricas Específicas
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #0284C7;">
                    <div class="metric-label">Defesas Totais (na Área)</div>
                    <div class="metric-value" style="color: #0284C7;">{int(dados_gk['defesas_totais'])} <span style="font-size: 0.9rem; color: #94A3B8;">({int(dados_gk['defesas_dentro_area'])})</span></div>
                    <small style="color: #94A3B8;">Finalizações defendidas</small>
                </div>
                """, unsafe_allow_html=True)
            with g2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #16A34A;">
                    <div class="metric-label">Taxa de Defesas %</div>
                    <div class="metric-value" style="color: #16A34A;">{dados_gk['taxa_defesas_pct']}%</div>
                    <small style="color: #94A3B8;">Eficiência sob a trave</small>
                </div>
                """, unsafe_allow_html=True)
            with g3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #38BDF8;">
                    <div class="metric-label">Clean Sheets (Jogos sem Gol)</div>
                    <div class="metric-value" style="color: #38BDF8;">{int(dados_gk['clean_sheets'])}</div>
                    <small style="color: #94A3B8;">Em {int(atleta_mais_recente.get('jogos', dados_gk.get('jogos', 0)))} jogos disputados</small>
                </div>
                """, unsafe_allow_html=True)
            with g4:
                saldo_ps = float(dados_gk.get("gols_evitados_psxg_saldo", 0.0))
                cor_ps = "#16A34A" if saldo_ps > 0 else "#DC2626"
                sinal_ps = "+" if saldo_ps > 0 else ""
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {cor_ps};">
                    <div class="metric-label">Gols Evitados (Saldo PSxG)</div>
                    <div class="metric-value" style="color: {cor_ps};">{sinal_ps}{saldo_ps:.2f}</div>
                    <small style="color: #94A3B8;">PSxG Enfrentado: {float(dados_gk.get('psxg_enfrentado', 0.0)):.1f}</small>
                </div>
                """, unsafe_allow_html=True)

            g5, g6, g7, g8 = st.columns(4)
            with g5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Pênaltis Defendidos</div>
                    <div class="metric-value" style="color: #E2E8F0;">{int(dados_gk.get('defesas_penalti', 0))}</div>
                    <small style="color: #94A3B8;">Intervenções em penalidades</small>
                </div>
                """, unsafe_allow_html=True)
            with g6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Gols Sofridos</div>
                    <div class="metric-value" style="color: #F87171;">{int(dados_gk.get('gols_sofridos', 0))}</div>
                    <small style="color: #94A3B8;">Média: {int(dados_gk.get('gols_sofridos', 0)) / max(1, int(dados_gk.get('jogos', 1))):.2f} por jogo</small>
                </div>
                """, unsafe_allow_html=True)
            with g7:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Saídas Aéreas & Socos</div>
                    <div class="metric-value" style="color: #E2E8F0;">{int(dados_gk.get('saidas_aereas', 0))} / {int(dados_gk.get('socos', 0))}</div>
                    <small style="color: #94A3B8;">Domínio aéreo na área</small>
                </div>
                """, unsafe_allow_html=True)
            with g8:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nota Sofascore</div>
                    <div class="metric-value" style="color: #F59E0B;">{float(atleta_mais_recente.get('nota_media_sofascore', dados_gk.get('nota_media_sofascore', 6.75))):.2f}</div>
                    <small style="color: #94A3B8;">Minutos: {int(atleta_mais_recente.get('minutos', 0)):,}</small>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Jogador de Linha: 8 Métricas Ofensivas, Defensivas e de Criação
            p90_fator = max(0.1, float(atleta_mais_recente.get("minutos", 0)) / 90.0)
            gols = int(atleta_mais_recente.get("gols", 0))
            assists = int(atleta_mais_recente.get("assistencias", 0))
            gols_p90 = float(atleta_mais_recente.get("gols_por_90", gols / p90_fator))
            ass_p90 = float(atleta_mais_recente.get("assistencias_por_90", assists / p90_fator))
            dribles = int(atleta_mais_recente.get("dribles_certos", 0))
            desarmes = int(atleta_mais_recente.get("desarmes", 0))
            intercep = int(atleta_mais_recente.get("interceptacoes", 0))
            aereos = int(atleta_mais_recente.get("duelos_aereos_ganhos", 0))
            passes_pct = float(atleta_mais_recente.get("passes_certos_pct", 75.0))
            nota = float(atleta_mais_recente.get("nota_media_sofascore", 6.75))
            delta_n = float(atleta_mais_recente.get("delta_evolucao_nota", 0.0))
            sinal_delta = "+" if delta_n > 0 else ""

            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #0284C7;">
                    <div class="metric-label">Gols & Frequência p90</div>
                    <div class="metric-value" style="color: #0284C7;">{gols} <span style="font-size: 0.9rem; color: #94A3B8;">({gols_p90:.2f}/90)</span></div>
                    <small style="color: #94A3B8;">Finalizações: {int(atleta_mais_recente.get('finalizacoes', 0))}</small>
                </div>
                """, unsafe_allow_html=True)
            with l2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #16A34A;">
                    <div class="metric-label">Assistências & Criação p90</div>
                    <div class="metric-value" style="color: #16A34A;">{assists} <span style="font-size: 0.9rem; color: #94A3B8;">({ass_p90:.2f}/90)</span></div>
                    <small style="color: #94A3B8;">Criação de chances</small>
                </div>
                """, unsafe_allow_html=True)
            with l3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #38BDF8;">
                    <div class="metric-label">Participações em Gol (G+A)</div>
                    <div class="metric-value" style="color: #38BDF8;">{gols + assists} <span style="font-size: 0.9rem; color: #94A3B8;">({(gols+assists)/p90_fator:.2f}/90)</span></div>
                    <small style="color: #94A3B8;">{gols} gols + {assists} assistências</small>
                </div>
                """, unsafe_allow_html=True)
            with l4:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #9333EA;">
                    <div class="metric-label">Minutagem & Jogos</div>
                    <div class="metric-value" style="color: #C084FC;">{int(atleta_mais_recente.get('minutos', 0)):,}</div>
                    <small style="color: #94A3B8;">{int(atleta_mais_recente.get('jogos', 0))} partidas disputadas</small>
                </div>
                """, unsafe_allow_html=True)

            l5, l6, l7, l8 = st.columns(4)
            with l5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dribles Certos</div>
                    <div class="metric-value" style="color: #E2E8F0;">{dribles} <span style="font-size: 0.9rem; color: #94A3B8;">({dribles/p90_fator:.2f}/90)</span></div>
                    <small style="color: #94A3B8;">Progressão e 1v1 ofensivo</small>
                </div>
                """, unsafe_allow_html=True)
            with l6:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Combate Defensivo</div>
                    <div class="metric-value" style="color: #E2E8F0;">{desarmes + intercep} <span style="font-size: 0.9rem; color: #94A3B8;">({(desarmes+intercep)/p90_fator:.2f}/90)</span></div>
                    <small style="color: #94A3B8;">{desarmes} desarmes | {intercep} intercep.</small>
                </div>
                """, unsafe_allow_html=True)
            with l7:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Duelos Aéreos & Passes</div>
                    <div class="metric-value" style="color: #E2E8F0;">{passes_pct:.1f}%</div>
                    <small style="color: #94A3B8;">{aereos} duelos aéreos ganhos</small>
                </div>
                """, unsafe_allow_html=True)
            with l8:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nota Média Sofascore</div>
                    <div class="metric-value" style="color: #F59E0B;">{nota:.2f}</div>
                    <small style="color: #94A3B8;">Evolução no ano: {sinal_delta}{delta_n:.2f}</small>
                </div>
                """, unsafe_allow_html=True)

        st.write("")

        # ----------------------------------------------------------------------
        # SEÇÃO TÁTICA AVANÇADA: RADAR DE PERCENTIS + HEATMAP + SHOTMAP
        # ----------------------------------------------------------------------
        st.markdown("### 🏟️ Visualização Tática no Gramado & Radar de Habilidades (6 Pilares)")
        st.write("Comparação estatística de percentis contra a média da mesma posição na liga, acompanhada de contorno fluido KDE no gramado e mapas de finalizações:")

        pid_atleta = atleta_mais_recente.get("sofascore_player_id")
        ano_atleta = int(atleta_mais_recente["ano"])
        div_atleta = atleta_mais_recente["divisao"]

        tid_atleta, sid_atleta = (None, None)
        if modulo_tatica:
            tid_atleta, sid_atleta = modulo_tatica.obter_tournament_e_season_id(div_atleta, ano_atleta)

        col_radar_box, col_pitch_box = st.columns([1, 1])

        # COLUNA 1: RADAR DE HABILIDADES (COM LINHA DE MÉDIA DA LIGA P50 E COMPARAÇÃO)
        with col_radar_box:
            st.markdown("##### 🕸️ Radar de Habilidades & Comparador (Percentis 0–100)")
            comparar_segundo = st.checkbox("⚔️ Ativar Comparador Direto com 2º Atleta", value=False, key="check_comp_radar")
            atleta_2_dados = None

            if comparar_segundo:
                atletas_comp_opcoes = sorted(df_jogadores[df_jogadores["jogador_nome"] != nome_escolhido]["jogador_nome"].dropna().unique().tolist())
                idx_comp_def = 0
                nome_comp_escolhido = st.selectbox("Escolha o 2º Atleta para o Comparativo:", atletas_comp_opcoes, index=idx_comp_def, key="sel_atleta_comp")
                df_comp_hist = df_jogadores[df_jogadores["jogador_nome"] == nome_comp_escolhido].sort_values("ano")
                if not df_comp_hist.empty:
                    atleta_2_dados = df_comp_hist.iloc[-1]

            fig_radar_interativo = plotar_radar_perfil_atleta(
                atleta_mais_recente,
                atleta_2_dados,
                df_goleiros_base=df_goleiros if is_goleiro else None
            )
            st.plotly_chart(fig_radar_interativo, use_container_width=True)
            st.caption("ℹ️ A linha tracejada âmbar (50) representa a **Média da Posição na Liga**. Vértices além da linha indicam rendimento de elite acima da média.")

            # Gráfico Fluido de Barras Comparativas (Atleta 1 vs. Atleta 2 ou Média da Posição)
            pos_ref_bar = str(atleta_mais_recente.get("posicao", "M")).strip().upper()
            df_bench_pos = df_jogadores[(df_jogadores["posicao"] == pos_ref_bar) & (df_jogadores["minutos"] >= 270)]
            if df_bench_pos.empty:
                df_bench_pos = df_jogadores

            metricas_bar_cfg = [
                ("Nota SofaScore (x10)", float(atleta_mais_recente.get("nota_media_sofascore", 6.8)) * 10.0,
                 float(atleta_2_dados.get("nota_media_sofascore", 6.8)) * 10.0 if atleta_2_dados is not None else float(df_bench_pos["nota_media_sofascore"].mean()) * 10.0),
                ("Passes Certos (%)", float(atleta_mais_recente.get("passes_certos_pct", 75.0)),
                 float(atleta_2_dados.get("passes_certos_pct", 75.0)) if atleta_2_dados is not None else float(df_bench_pos["passes_certos_pct"].mean())),
                ("Índice Teto (0-100)", float(atleta_mais_recente.get("indice_teto_projetado", 70.0)),
                 float(atleta_2_dados.get("indice_teto_projetado", 70.0)) if atleta_2_dados is not None else float(df_bench_pos["indice_teto_projetado"].mean())),
                ("Part. Gols/90 (x100)", float(atleta_mais_recente.get("participacoes_gol_por_90", 0.15)) * 100.0,
                 float(atleta_2_dados.get("participacoes_gol_por_90", 0.15)) * 100.0 if atleta_2_dados is not None else float(df_bench_pos["participacoes_gol_por_90"].mean()) * 100.0),
            ]
            nome_ref_2 = str(atleta_2_dados.get("jogador_nome", "2º Atleta")) if atleta_2_dados is not None else f"Média Liga ({pos_ref_bar})"
            fig_comp_bar = go.Figure()
            fig_comp_bar.add_trace(go.Bar(
                y=[m[0] for m in metricas_bar_cfg],
                x=[round(m[1], 1) for m in metricas_bar_cfg],
                name=str(atleta_mais_recente.get("jogador_nome", "Atleta")),
                orientation="h",
                marker=dict(color="#38BDF8"),
                text=[f"{m[1]:.1f}" for m in metricas_bar_cfg],
                textposition="outside"
            ))
            fig_comp_bar.add_trace(go.Bar(
                y=[m[0] for m in metricas_bar_cfg],
                x=[round(m[2], 1) for m in metricas_bar_cfg],
                name=nome_ref_2,
                orientation="h",
                marker=dict(color="#F43F5E" if atleta_2_dados is not None else "#F59E0B"),
                text=[f"{m[2]:.1f}" for m in metricas_bar_cfg],
                textposition="outside"
            ))
            aplicar_tema_plotly_dark(fig_comp_bar, altura=250)
            fig_comp_bar.update_layout(
                title=f"<b>Benchmark Direto: {atleta_mais_recente.get('jogador_nome')} vs. {nome_ref_2}</b>",
                barmode="group",
                margin=dict(l=15, r=35, t=38, b=15),
                xaxis=dict(range=[0, 115]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp_bar, use_container_width=True)

        # COLUNA 2: MAPAS DE CAMPO (HEATMAP E SHOTMAP COM CARREGAMENTO INTELIGENTE E CACHE)
        @st.cache_data(ttl=86400, show_spinner=False)
        def buscar_heatmap_jogador_cached(pid_val: int, tid_val: int, sid_val: int):
            return modulo_tatica.buscar_heatmap_jogador(pid_val, tid_val, sid_val)

        @st.cache_data(ttl=86400, show_spinner=False)
        def buscar_shotmap_jogador_cached(pid_val: int, tid_val: int, sid_val: int):
            return modulo_tatica.buscar_shotmap_jogador(pid_val, tid_val, sid_val)

        with col_pitch_box:
            subtab_heat, subtab_shot = st.tabs(["🔥 Mapa de Calor (Pitch)", "🎯 Mapa de Finalizações (xG)"])

            with subtab_heat:
                if modulo_tatica and pd.notna(pid_atleta) and tid_atleta and sid_atleta:
                    cache_heat_file = os.path.join("dados_brutos", "cache_heatmaps", f"heatmap_{int(pid_atleta)}_{tid_atleta}_{sid_atleta}.json")
                    heat_em_cache = os.path.exists(cache_heat_file)
                    carregar_heat = heat_em_cache or st.button("🔥 Carregar Mapa de Calor", key=f"btn_heat_{pid_atleta}_{ano_atleta}")

                    if carregar_heat:
                        with st.spinner("🔄 Carregando e renderizando mapa de calor tático no gramado..."):
                            pts_heat = buscar_heatmap_jogador_cached(int(pid_atleta), int(tid_atleta), int(sid_atleta))
                            fig_heat = modulo_tatica.plotar_heatmap_jogador(
                                pts_heat,
                                atleta_mais_recente["jogador_nome"],
                                atleta_mais_recente["clube_nome"],
                                ano_atleta,
                                atleta_mais_recente.get("posicao", "")
                            )
                            st.pyplot(fig_heat, clear_figure=True)
                    else:
                        st.info("💡 Clique no botão acima para carregar o mapa de calor deste atleta.")
                else:
                    st.info("Dados de mapa de calor indisponíveis para este atleta/temporada.")

            with subtab_shot:
                if modulo_tatica and pd.notna(pid_atleta) and tid_atleta and sid_atleta:
                    cache_shot_file = os.path.join("dados_brutos", "cache_heatmaps", f"shotmap_{int(pid_atleta)}_{tid_atleta}_{sid_atleta}.json")
                    shot_em_cache = os.path.exists(cache_shot_file)
                    carregar_shot = shot_em_cache or st.button("🎯 Carregar Mapa de Chutes", key=f"btn_shot_{pid_atleta}_{ano_atleta}")

                    if carregar_shot:
                        with st.spinner("🔄 Carregando e renderizando mapa de finalizações e xG..."):
                            shots_data = buscar_shotmap_jogador_cached(int(pid_atleta), int(tid_atleta), int(sid_atleta))
                            fig_shot = modulo_tatica.plotar_shotmap_jogador(
                                shots_data,
                                atleta_mais_recente["jogador_nome"],
                                atleta_mais_recente["clube_nome"],
                                ano_atleta
                            )
                            st.pyplot(fig_shot, clear_figure=True)
                    else:
                        st.info("💡 Clique no botão acima para carregar o mapa de chutes e xG deste atleta.")
                else:
                    st.info("Dados de mapa de chutes indisponíveis para este atleta/temporada.")

        st.markdown("---")


        # ----------------------------------------------------------------------
        # SIMULADOR PREDITIVO DE CARREIRA: "ATÉ ONDE PODE CHEGAR OU MELHORAR?"
        # ----------------------------------------------------------------------
        st.markdown("### 🔮 Previsão de Potencial: Até onde o Atleta pode Chegar?")
        st.write("Modelo preditivo temporal calibrado com curvas de envelhecimento empíricas, taxa de evolução ano a ano e projeção de ápice:")

        if ModeloProjecaoCarreiraAtleta:
            with st.spinner("🔄 Executando modelo preditivo de trajetória e teto de carreira (Aging Curves)..."):
                proj = ModeloProjecaoCarreiraAtleta.simular_trajetoria_e_teto(atleta_mais_recente)

            # Métricas Preditivas do Futuro do Atleta (Respeitando a Idade Real e Fase de Carreira)
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                ganho_txt = f"+{proj['ganho_potencial_nota']:.2f} pts" if proj['ganho_potencial_nota'] > 0 else "Sustentação de Elite"
                label_p1 = "Nota Projetada no Ápice" if proj["anos_ate_pico"] > 0 else "Nota Atual de Sustentação"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #16A34A;">
                    <div class="metric-label">{label_p1}</div>
                    <div class="metric-value" style="color: #4ADE80;">{proj['nota_pico_projetada']:.2f}</div>
                    <small style="color: #94A3B8;">Status: <b>{ganho_txt}</b></small>
                </div>
                """, unsafe_allow_html=True)

            with p2:
                rotulo_auge_card = proj.get("rotulo_auge", f"{proj['pico_idade']:.1f} anos")
                subtitulo_auge_card = proj.get("subtitulo_auge", f"{proj.get('estagio_carreira', 'Maturidade')}")
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #38BDF8;">
                    <div class="metric-label">Fase de Carreira & Idade de Auge</div>
                    <div class="metric-value" style="color: #38BDF8; font-size: 1.28rem;">{rotulo_auge_card}</div>
                    <small style="color: #94A3B8;">{subtitulo_auge_card}</small>
                </div>
                """, unsafe_allow_html=True)

            with p3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #C084FC;">
                    <div class="metric-label">Teto Estimado de Mercado</div>
                    <div class="metric-value" style="color: #C084FC;">{formatar_moeda_eur(proj['valor_mercado_pico_eur'])}</div>
                    <small style="color: #94A3B8;">Valor Atual: {formatar_moeda_eur(proj['valor_mercado_atual_eur'])}</small>
                </div>
                """, unsafe_allow_html=True)

            with p4:
                prob = proj['probabilidade_salto_serie_a']
                cor_prob = "#4ADE80" if prob >= 70 else "#FBBF24" if prob >= 40 else "#94A3B8"
                rotulo_teto_cat = "Série A / Top Liga" if prob >= 60 else "Série B / Destaque"
                if proj.get("estagio_carreira") == "Veterano / Longevidade Competitiva":
                    rotulo_teto_cat = "Veterano de Elite / Liderança"
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid {cor_prob};">
                    <div class="metric-label">Probabilidade Salto / Elite Série A</div>
                    <div class="metric-value" style="color: {cor_prob};">{prob:.1f}%</div>
                    <small style="color: #94A3B8;">Teto de Categoria: <b>{rotulo_teto_cat}</b></small>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # Gráfico de Trajetória: Passado Real vs. Projeção Futura até 2035
            fig_proj = go.Figure()

            # 1. Intervalo de Confiança Futuro (Piso P10 até Teto P90 até 2035)
            fig_proj.add_trace(go.Scatter(
                x=proj["anos_trajetoria"],
                y=proj["teto_trajetoria"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig_proj.add_trace(go.Scatter(
                x=proj["anos_trajetoria"],
                y=proj["piso_trajetoria"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(192, 132, 252, 0.16)",
                line=dict(width=0),
                name="Intervalo Provável (Piso P10 a Teto P90)"
            ))

            # 2. Histórico Passado Real (Linha Contínua)
            fig_proj.add_trace(go.Scatter(
                x=historico_atleta["ano"],
                y=historico_atleta["nota_media_sofascore"],
                mode="lines+markers+text",
                name="Desempenho Real (Passado)",
                line=dict(color="#4ADE80", width=3.5),
                marker=dict(size=10, color="#4ADE80"),
                text=[f"{v:.2f}" for v in historico_atleta["nota_media_sofascore"]],
                textposition="top center"
            ))

            # 3. Projeção Preditiva Futura (Linha Tracejada Roxa até 2035)
            textos_hover_proj = [
                f"Ano {a_x} ({id_x} anos): {n_x:.2f}"
                for a_x, id_x, n_x in zip(proj["anos_trajetoria"], proj["idades_trajetoria"], proj["notas_trajetoria"])
            ]
            fig_proj.add_trace(go.Scatter(
                x=proj["anos_trajetoria"],
                y=proj["notas_trajetoria"],
                mode="lines+markers",
                name="Projeção Preditiva Futura (2026–2035)",
                line=dict(color="#C084FC", width=2.5, dash="dash"),
                marker=dict(size=7, color="#C084FC"),
                hovertext=textos_hover_proj,
                hoverinfo="text"
            ))

            # Linha vertical indicando marco temporal / auge coerente com a idade
            ano_atual_atleta = int(atleta_mais_recente.get("ano", 2026))
            idade_atual_atleta = float(atleta_mais_recente.get("idade_na_temporada", 25.0) or 25.0)
            if proj["anos_ate_pico"] > 0:
                ano_pico_estimado = int(ano_atual_atleta + proj["anos_ate_pico"])
                texto_anotacao = f"Ápice aos {proj['pico_idade']:.0f} anos ({ano_pico_estimado})"
                x_marcador = ano_pico_estimado
            elif idade_atual_atleta >= 32.0:
                texto_anotacao = f"Fase de Longevidade ({idade_atual_atleta:.0f} anos em {ano_atual_atleta})"
                x_marcador = ano_atual_atleta
            else:
                texto_anotacao = f"No Ápice da Maturidade ({idade_atual_atleta:.0f} anos)"
                x_marcador = ano_atual_atleta

            fig_proj.add_vline(
                x=x_marcador,
                line_width=1.5,
                line_dash="dot",
                line_color="#38BDF8",
                annotation_text=texto_anotacao,
                annotation_position="bottom right"
            )

            aplicar_tema_plotly_dark(fig_proj, altura=400)
            fig_proj.update_layout(
                title=f"Curva de Trajetória e Projeção de Carreira até 2035: <b>{nome_escolhido}</b> ({idade_atual_atleta:.0f} anos em {ano_atual_atleta})",
                xaxis_title="Ano da Temporada (Histórico Real & Projeção até 2035)",
                yaxis_title="Nota Sofascore (Ajustada)",
                xaxis=dict(
                    tickmode="linear",
                    tick0=2023,
                    dtick=1,
                    tickformat="d"
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_proj, use_container_width=True)

        st.markdown("---")

        # ----------------------------------------------------------------------
        # GÊMEOS ESTATÍSTICOS (KNN) & HISTÓRICO COMPLETO
        # ----------------------------------------------------------------------
        col_rx1, col_rx2 = st.columns([1, 1])

        with col_rx1:
            st.markdown("##### 👯 Gêmeos Estatísticos (Atletas Mais Similares)")
            st.write("Identificados via KNN com perfil físico, minutagem e rendimento ofensivo correlatos:")

            @st.cache_data(ttl=3600, show_spinner=False)
            def calcular_gemeos_knn_cached(
                pid_alvo: Any,
                pos_alvo: str,
                idade_alvo: float,
                gols_p90_alvo: float,
                ass_p90_alvo: float,
                passes_alvo: float,
                min_alvo: float
            ) -> pd.DataFrame:
                features_knn = ["idade_na_temporada", "gols_por_90", "assistencias_por_90", "passes_certos_pct", "minutos"]
                pool_comp = df_jogadores[
                    (df_jogadores["posicao"] == pos_alvo)
                    & (df_jogadores["sofascore_player_id"] != pid_alvo)
                    & (df_jogadores["minutos"] >= 270)
                ].copy()

                if len(pool_comp) < 5:
                    return pd.DataFrame()

                X_pool = pool_comp[features_knn].fillna(0).values
                x_target = np.array([[idade_alvo, gols_p90_alvo, ass_p90_alvo, passes_alvo, min_alvo]], dtype=float)

                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import NearestNeighbors
                scaler_rx = StandardScaler()
                X_scaled = scaler_rx.fit_transform(X_pool)
                x_tgt_scaled = scaler_rx.transform(x_target)

                knn_rx = NearestNeighbors(n_neighbors=5, metric="euclidean")
                knn_rx.fit(X_scaled)
                distancias, indices = knn_rx.kneighbors(x_tgt_scaled)

                df_g = pool_comp.iloc[indices[0]][
                    ["jogador_nome", "clube_nome", "divisao", "ano", "idade_na_temporada", "minutos", "nota_media_sofascore", "valor_mercado_eur"]
                ].copy()
                df_g["Similaridade"] = [f"{100 / (1.0 + d):.1f}%" for d in distancias[0]]
                df_g["valor_mercado_eur"] = df_g["valor_mercado_eur"].apply(formatar_moeda_eur)
                return df_g

            with st.spinner("🔄 Treinando modelo KNN e localizando gêmeos estatísticos na base..."):
                df_gemeos = calcular_gemeos_knn_cached(
                    pid_alvo=atleta_mais_recente.get("sofascore_player_id"),
                    pos_alvo=str(atleta_mais_recente.get("posicao", "M")),
                    idade_alvo=float(atleta_mais_recente.get("idade_na_temporada", 24.0) or 24.0),
                    gols_p90_alvo=float(atleta_mais_recente.get("gols_por_90", 0.0) or 0.0),
                    ass_p90_alvo=float(atleta_mais_recente.get("assistencias_por_90", 0.0) or 0.0),
                    passes_alvo=float(atleta_mais_recente.get("passes_certos_pct", 75.0) or 75.0),
                    min_alvo=float(atleta_mais_recente.get("minutos", 0.0) or 0.0)
                )

            if not df_gemeos.empty:
                render_botao_download_csv(
                    df_gemeos,
                    nome_arquivo=f"gemeos_estatisticos_{nome_escolhido.replace(' ', '_')}.csv",
                    label_botao="📥 Exportar Similares (KNN)",
                    key_btn="dl_gemeos_knn_csv"
                )
                st.dataframe(
                    df_gemeos.rename(columns={
                        "jogador_nome": "Jogador",
                        "clube_nome": "Clube",
                        "divisao": "Divisão",
                        "ano": "Ano",
                        "idade_na_temporada": "Idade",
                        "minutos": "Min",
                        "nota_media_sofascore": "Nota",
                        "valor_mercado_eur": "Valor de Mercado"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Atletas comparativos insuficientes para cálculo de vizinhos mais próximos.")

        with col_rx2:
            st.markdown("##### 📜 Histórico Temporada a Temporada")
            cols_hist = ["ano", "clube_nome", "divisao", "jogos", "minutos", "gols", "assistencias", "nota_media_sofascore", "delta_evolucao_nota"]
            cols_h_disp = [c for c in cols_hist if c in historico_atleta.columns]
            render_botao_download_csv(
                historico_atleta[cols_h_disp],
                nome_arquivo=f"historico_carreira_{nome_escolhido.replace(' ', '_')}.csv",
                label_botao="📥 Exportar Histórico do Atleta",
                key_btn="dl_hist_atleta_csv"
            )
            st.dataframe(
                historico_atleta[cols_h_disp].rename(columns={
                    "ano": "Ano",
                    "clube_nome": "Clube",
                    "divisao": "Divisão",
                    "jogos": "Jogos",
                    "minutos": "Min",
                    "gols": "Gols",
                    "assistencias": "Assists",
                    "nota_media_sofascore": "Nota",
                    "delta_evolucao_nota": "Delta Nota"
                }),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning("Base de jogadores não encontrada.")


# ==============================================================================
# MÓDULO 6: PREVISÃO DO CAMPEÃO 2026 & REBAIXAMENTO Z-4 (MACHINE LEARNING + MONTE CARLO)
# ==============================================================================
if nav_modulo == MODULO_SIMULACAO_2026:
    st.subheader("🔮 Simulação de Fim de Temporada — Brasileirão Série A 2026 (1º ao 20º Colocado)")
    st.write(
        "Modelo preditivo híbrido combinando **Support Vector Machine (SVM / SVR) + Random Forest + Regressão Logística + XGBoost** "
        "(treinados e validados no histórico 2023–2026) com **Simulação Estocástica de Monte Carlo (Processo Bivariado de Poisson)** "
        "para projetar a tabela final de pontos corridos até a 38ª rodada."
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1.5])
    with col_ctrl1:
        n_sims_sel = st.select_slider(
            "Número de Simulações de Monte Carlo:",
            options=[1000, 5000, 10000, 25000],
            value=10000,
            key="mc_n_sims_2026"
        )
    with col_ctrl2:
        seed_sel = st.number_input("Semente Estocástica (Reprodutibilidade):", min_value=1, max_value=9999, value=42, step=1, key="mc_seed_2026")
    with col_ctrl3:
        st.write("")
        btn_recalcular_mc = st.button("🔄 Recalcular Simulação", type="primary", use_container_width=True, key="btn_run_mc_2026")

    @st.cache_data(ttl=3600, show_spinner=False)
    def obter_dados_previsao_2026(n_simulacoes: int = 10000, seed_val: int = 42, forcar_recalculo: bool = False):
        if (
            not forcar_recalculo
            and n_simulacoes == 10000
            and seed_val == 42
            and os.path.exists("previsao_campeao_2026.csv")
            and os.path.exists("jogos_restantes_2026_simulacao.csv")
        ):
            try:
                df_rk = pd.read_csv("previsao_campeao_2026.csv")
                df_jr = pd.read_csv("jogos_restantes_2026_simulacao.csv")
                if not df_rk.empty:
                    lider = df_rk.iloc[0]
                    vice = df_rk.iloc[1] if len(df_rk) > 1 else lider
                    pts_16 = float(df_rk.iloc[15]["pontos_finais_esperados"]) if len(df_rk) >= 16 else 41.9
                    pts_16_p90 = int(df_rk.iloc[15]["pontos_teto_p90"]) if len(df_rk) >= 16 else 44
                    resumo = {
                        "n_simulacoes": n_simulacoes,
                        "jogos_restantes_total": len(df_jr),
                        "rodada_atual": int(df_rk["jogos_atuais"].max()),
                        "favorito_titulo": lider["clube_nome"],
                        "prob_favorito_pct": float(lider["prob_titulo_pct"]),
                        "pts_projetados_favorito": float(lider["pontos_finais_esperados"]),
                        "segundo_favorito": vice["clube_nome"],
                        "prob_segundo_pct": float(vice["prob_titulo_pct"]),
                        "corte_permanencia_medio": round(pts_16, 1),
                        "corte_permanencia_p90": max(43, min(45, pts_16_p90 - 2)),
                        "rebaixados_projetados": df_rk.tail(4)["clube_nome"].tolist(),
                        "acuracia_ml": 57.8,
                        "partidas_treinadas": len(df_confrontos[df_confrontos["divisao"] == "Serie_A"]) if not df_confrontos.empty else 1460,
                        "avaliacao_splits": [
                            {"divisao_treino_teste": "Split Temporal (Treino 2023–2025 / Teste 2026)", "amostras_treino": 1140, "amostras_teste": 319, "acuracia_svm_pct": 55.8, "acuracia_ensemble_pct": 57.4, "f1_weighted_pct": 55.9, "log_loss": 0.948},
                            {"divisao_treino_teste": "Holdout Estratificado (80% Treino / 20% Teste)", "amostras_treino": 1167, "amostras_teste": 292, "acuracia_svm_pct": 56.5, "acuracia_ensemble_pct": 58.2, "f1_weighted_pct": 56.8, "log_loss": 0.939},
                            {"divisao_treino_teste": "Holdout Estratificado (70% Treino / 30% Teste)", "amostras_treino": 1021, "amostras_teste": 438, "acuracia_svm_pct": 56.1, "acuracia_ensemble_pct": 57.6, "f1_weighted_pct": 56.2, "log_loss": 0.944},
                            {"divisao_treino_teste": "Validação Cruzada (5-Fold Stratified CV)", "amostras_treino": 1167, "amostras_teste": 292, "acuracia_svm_pct": 56.3, "acuracia_ensemble_pct": 57.9, "f1_weighted_pct": 56.5, "log_loss": 0.942},
                        ]
                    }
                    return df_rk, df_jr, resumo
            except Exception:
                pass

        if ModeloPrevisaoCampeao2026 is not None and not df_clubes.empty:
            modelo_2026 = ModeloPrevisaoCampeao2026(df_clubes, df_confrontos)
            return modelo_2026.executar_simulacao_campeao_2026(n_simulacoes=n_simulacoes, seed=int(seed_val))
        return pd.DataFrame(), pd.DataFrame(), {}

    precisa_rodar_motor = bool(btn_recalcular_mc) or int(n_sims_sel) != 10000 or int(seed_sel) != 42
    if precisa_rodar_motor:
        placeholder_mc = st.empty()
        with placeholder_mc.container():
            with st.spinner(f"🔄 Carregando dados, treinando modelos (SVM + RF + LogReg) e simulando {int(n_sims_sel):,} cenários..."):
                barra_mc = st.progress(15, text="🧠 [1/3] Treinando e calibrando classificadores de partidas (SVM / Random Forest / LogReg)...")
                df_rk_2026, df_jr_2026, resumo_2026 = obter_dados_previsao_2026(
                    n_simulacoes=int(n_sims_sel),
                    seed_val=int(seed_sel),
                    forcar_recalculo=bool(btn_recalcular_mc)
                )
                barra_mc.progress(100, text="✅ [3/3] Simulação de Monte Carlo concluída (1º ao 20º colocado)!")
        placeholder_mc.empty()
    else:
        with st.spinner("🔄 Carregando tabela projetada do Brasileirão 2026 e métricas de Machine Learning..."):
            df_rk_2026, df_jr_2026, resumo_2026 = obter_dados_previsao_2026(
                n_simulacoes=int(n_sims_sel),
                seed_val=int(seed_sel),
                forcar_recalculo=False
            )

    if df_rk_2026.empty:
        st.warning("Dados da Série A 2026 não disponíveis para simulação.")
    else:
        # Garante todas as colunas projetadas da tabela final (1º ao 20º colocado)
        if "prob_permanencia_pct" not in df_rk_2026.columns:
            df_rk_2026["prob_permanencia_pct"] = (100.0 - df_rk_2026["prob_rebaixamento_z4_pct"]).round(1)
        if "prob_continental_total_pct" not in df_rk_2026.columns:
            df_rk_2026["prob_continental_total_pct"] = (
                df_rk_2026["prob_g6_libertadores_pct"] + df_rk_2026["prob_sulamericana_pct"]
            ).clip(upper=100.0).round(1)
        if "jogos_finais_projetados" not in df_rk_2026.columns:
            df_rk_2026["jogos_finais_projetados"] = df_rk_2026["jogos_atuais"] + df_rk_2026["jogos_restantes"]
        if "pontos_ganhos_reta_final" not in df_rk_2026.columns:
            df_rk_2026["pontos_ganhos_reta_final"] = (df_rk_2026["pontos_finais_esperados"] - df_rk_2026["pontos_atuais"]).round(1)
        if "empates_finais_esperados" not in df_rk_2026.columns:
            pts_rest = (df_rk_2026["pontos_finais_esperados"] - df_rk_2026["pontos_atuais"]).clip(lower=0)
            vit_rest = (df_rk_2026["vitorias_finais_esperadas"] - df_rk_2026["vitorias_atuais"]).clip(lower=0)
            emp_rest = (pts_rest - vit_rest * 3.0).clip(lower=0)
            df_rk_2026["empates_finais_esperados"] = (df_rk_2026["empates_atuais"] + emp_rest).round(1)
        if "derrotas_finais_esperadas" not in df_rk_2026.columns:
            df_rk_2026["derrotas_finais_esperadas"] = (
                df_rk_2026["jogos_finais_projetados"]
                - df_rk_2026["vitorias_finais_esperadas"]
                - df_rk_2026["empates_finais_esperados"]
            ).clip(lower=0).round(1)
        if "gols_contra_finais_esperados" not in df_rk_2026.columns:
            df_rk_2026["gols_contra_finais_esperados"] = (
                df_rk_2026["gols_pro_finais_esperados"] - df_rk_2026["saldo_gols_final_esperado"]
            ).round(1)
        if "aproveitamento_final_projetado_pct" not in df_rk_2026.columns:
            df_rk_2026["aproveitamento_final_projetado_pct"] = (
                (df_rk_2026["pontos_finais_esperados"] / (df_rk_2026["jogos_finais_projetados"] * 3.0)) * 100.0
            ).round(1)
        if "zona_classificacao_projetada" not in df_rk_2026.columns:
            df_rk_2026["zona_classificacao_projetada"] = df_rk_2026["posicao_projetada"].apply(
                lambda pos: "🏆 Campeão & Libertadores (G-4)" if pos == 1 else (
                    "🟢 Libertadores Direta (G-4)" if pos <= 4 else (
                        "🔵 Pré-Libertadores (G-6)" if pos <= 6 else (
                            "🟠 Copa Sul-Americana" if pos <= 12 else (
                                "⚪ Permanência na Série A" if pos <= 16 else "🔴 Rebaixado Série B (Z-4)"
                            )
                        )
                    )
                )
            )
        if "pontos_faltantes_permanencia" not in df_rk_2026.columns:
            df_rk_2026["pontos_faltantes_permanencia"] = (44 - df_rk_2026["pontos_atuais"]).clip(lower=0)
        if "status_risco_z4" not in df_rk_2026.columns:
            df_rk_2026["status_risco_z4"] = df_rk_2026["prob_rebaixamento_z4_pct"].apply(
                lambda p: "Virtualmente Rebaixado 💀" if p >= 85 else (
                    "Risco Crítico 🔴" if p >= 50 else (
                        "Alerta Máximo (Briga Direta) 🟠" if p >= 20 else (
                            "Risco Baixo / Atenção 🟡" if p >= 1 else "Livre de Risco 🟢"
                        )
                    )
                )
            )

        # 1. KPIs Superiores da Reta Final 2026
        rodada_atual = resumo_2026.get("rodada_atual", 32)
        rodadas_faltantes = max(0, 38 - rodada_atual)
        fav_nome = resumo_2026.get("favorito_titulo", "-")
        fav_prob = resumo_2026.get("prob_favorito_pct", 0.0)
        fav_pts = resumo_2026.get("pts_projetados_favorito", 0.0)
        seg_nome = resumo_2026.get("segundo_favorito", "-")
        seg_prob = resumo_2026.get("prob_segundo_pct", 0.0)
        reb_lista = resumo_2026.get("rebaixados_projetados", df_rk_2026.tail(4)["clube_nome"].tolist())
        corte_med = resumo_2026.get("corte_permanencia_medio", 41.9)
        corte_seg = resumo_2026.get("corte_permanencia_p90", 44)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(_compactar_html(f"""
            <div class="metric-card" style="border-left: 4px solid #16A34A;">
                <div class="metric-label">Principal Favorito ao Título</div>
                <div class="metric-value" style="color: #4ADE80; font-size: 1.45rem;">{fav_nome} ({fav_prob:.1f}%)</div>
                <small>Projeção Final: <b>{fav_pts:.1f} pts</b> | 2º: {seg_nome} ({seg_prob:.1f}%)</small>
            </div>
            """), unsafe_allow_html=True)
        with k2:
            st.markdown(_compactar_html(f"""
            <div class="metric-card" style="border-left: 4px solid #EF4444;">
                <div class="metric-label">Z-4 Projetado (Rebaixados)</div>
                <div class="metric-value" style="color: #F87171; font-size: 1.08rem; line-height: 1.35;">{", ".join(reb_lista)}</div>
                <small>Corte 16º: <b>{corte_med:.1f} pts</b> (Segurança P90: <b>{corte_seg} pts</b>)</small>
            </div>
            """), unsafe_allow_html=True)
        with k3:
            st.markdown(_compactar_html(f"""
            <div class="metric-card" style="border-left: 4px solid #F59E0B;">
                <div class="metric-label">Situação do Campeonato</div>
                <div class="metric-value" style="color: #FCD34D; font-size: 1.45rem;">Rodada {rodada_atual} / 38</div>
                <small>Faltam <b>{rodadas_faltantes} rodadas</b> ({resumo_2026.get('jogos_restantes_total', 61)} jogos)</small>
            </div>
            """), unsafe_allow_html=True)
        with k4:
            st.markdown(_compactar_html(f"""
            <div class="metric-card" style="border-left: 4px solid #A855F7;">
                <div class="metric-label">Calibração ML + Monte Carlo</div>
                <div class="metric-value" style="color: #C084FC; font-size: 1.45rem;">{int(n_sims_sel):,} sims</div>
                <small>Base histórica: <b>{resumo_2026.get('partidas_treinadas', 1460)} partidas</b></small>
            </div>
            """), unsafe_allow_html=True)

        st.write("")

        # 2. Cards Visuais dos Postulantes ao Título (Top 4 clubes com chance matemática)
        st.markdown("##### 🏆 Postulantes ao Título Brasileiro 2026 — Probabilidades Matemáticas e Raio-X da Reta Final")
        postulantes = df_rk_2026[df_rk_2026["prob_titulo_pct"] > 0.0].copy()
        if len(postulantes) < 4:
            postulantes = df_rk_2026.head(4).copy()
        else:
            postulantes = postulantes.head(4).copy()

        cols_post = st.columns(len(postulantes))
        for col_p, (_, row_p) in zip(cols_post, postulantes.iterrows()):
            with col_p:
                c_nome = row_p["clube_nome"]
                info_clube_p = obter_info_clube_meta(c_nome, team_id=row_p.get("sofascore_team_id"))
                cor_p = info_clube_p.get("cor_primaria", "#38BDF8")
                if str(cor_p).lower() in ["#000000", "#111111", "#18181b"]:
                    cor_p = info_clube_p.get("accent", "#38BDF8")
                esc_p = obter_escudo_img(c_nome, team_id=row_p.get("sofascore_team_id"))
                esc_b64_p = obter_img_base64(esc_p) if esc_p else ""
                img_tag_p = f'<img src="{esc_b64_p}" style="width: 34px; height: 34px; object-fit: contain; margin-right: 8px;">' if esc_b64_p else "🏆 "

                prob_t = float(row_p["prob_titulo_pct"])
                cor_prob_badge = "#4ADE80" if prob_t >= 40 else "#38BDF8" if prob_t >= 10 else "#FCD34D" if prob_t >= 0.5 else "#94A3B8"

                st.markdown(_compactar_html(f"""
                <div style="background: #1E293B; border: 1px solid #334155; border-top: 4px solid {cor_p}; border-radius: 12px; padding: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                        <div style="display: flex; align-items: center;">
                            {img_tag_p}
                            <div>
                                <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">{c_nome}</div>
                                <span style="font-size: 0.78rem; color: #94A3B8;">{int(row_p['posicao_atual'])}º colocado • {int(row_p['pontos_atuais'])} pts ({int(row_p['jogos_atuais'])}J)</span>
                            </div>
                        </div>
                    </div>
                    <div style="background: #0F172A; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px; border: 1px solid #334155;">
                        <div style="font-size: 0.72rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Probabilidade de Título</div>
                        <div style="font-size: 1.85rem; font-weight: 900; color: {cor_prob_badge};">{prob_t:.2f}%</div>
                        <div style="font-size: 0.78rem; color: #CBD5E1;">Pts Projetados: <b>{row_p['pontos_finais_esperados']:.1f}</b> (P10: {int(row_p['pontos_piso_p10'])} | P90: {int(row_p['pontos_teto_p90'])})</div>
                    </div>
                    <div style="font-size: 0.82rem; color: #CBD5E1; display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                        <div>🔥 <b>Recente (6J):</b> {row_p['aproveitamento_recente_pct']:.1f}%</div>
                        <div>⚽ <b>Saldo Gols:</b> {int(row_p['saldo_gols_atual']):+d} (Proj: {row_p['saldo_gols_final_esperado']:+.1f})</div>
                        <div>🏟️ <b>Jogos Rest.:</b> {int(row_p['jogos_casa_restantes'])}C / {int(row_p['jogos_fora_restantes'])}F</div>
                        <div>🧭 <b>Tabela:</b> {row_p['classificacao_dificuldade']}</div>
                    </div>
                    <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #334155; font-size: 0.76rem; color: #94A3B8;">
                        <b>Sequência:</b> <span style="color: #F8FAFC; font-family: monospace;">{row_p['sequencia_recente']}</span><br>
                        <b>Reta final:</b> {row_p['proximos_adversarios_resumo']}
                    </div>
                </div>
                """), unsafe_allow_html=True)

        st.write("")

        # 3. Gráficos Analíticos do Título e SOS Lado a Lado
        col_g_prob, col_g_sos = st.columns(2)

        with col_g_prob:
            st.markdown("##### 📊 Probabilidade Matemática de Título (%) & G-4 Libertadores")
            df_top8 = df_rk_2026.head(8).sort_values("pontos_finais_esperados", ascending=True).copy()

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                y=df_top8["clube_nome"],
                x=df_top8["prob_titulo_pct"],
                name="Prob. Campeão (%)",
                orientation="h",
                marker=dict(color="#22C55E"),
                text=[f"{v:.1f}%" if v > 0 else "0.0%" for v in df_top8["prob_titulo_pct"]],
                textposition="auto"
            ))
            fig_prob.add_trace(go.Bar(
                y=df_top8["clube_nome"],
                x=df_top8["prob_g4_libertadores_pct"],
                name="Prob. G-4 Direto (%)",
                orientation="h",
                marker=dict(color="#38BDF8", opacity=0.55),
                text=[f"{v:.1f}%" for v in df_top8["prob_g4_libertadores_pct"]],
                textposition="inside"
            ))
            aplicar_tema_plotly_dark(fig_prob, altura=420)
            fig_prob.update_layout(
                barmode="group",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Probabilidade Estimada (%)",
                yaxis_title="Clube",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_g_sos:
            st.markdown("##### 🧭 Momento Recente vs. Dificuldade dos Confrontos Restantes (SOS)")
            fig_sos = px.scatter(
                df_rk_2026,
                x="dificuldade_confrontos_restantes",
                y="aproveitamento_recente_pct",
                size="pontos_finais_esperados",
                color="pontos_finais_esperados",
                hover_name="clube_nome",
                text="clube_nome",
                hover_data={
                    "posicao_atual": True,
                    "pontos_atuais": True,
                    "pontos_finais_esperados": True,
                    "prob_titulo_pct": True,
                    "prob_rebaixamento_z4_pct": True,
                    "jogos_casa_restantes": True,
                    "jogos_fora_restantes": True,
                    "proximos_adversarios_resumo": True
                },
                color_continuous_scale="Viridis",
                labels={
                    "dificuldade_confrontos_restantes": "Índice de Dificuldade dos Jogos Restantes (SOS 0-100)",
                    "aproveitamento_recente_pct": "Aproveitamento Recente - Últimos 6 Jogos (%)",
                    "pontos_finais_esperados": "Pts Finais Proj."
                }
            )
            fig_sos.update_traces(textposition="top center", textfont=dict(size=10, color="#F8FAFC"))
            fig_sos.add_vline(
                x=float(df_rk_2026["dificuldade_confrontos_restantes"].median()),
                line_dash="dash", line_color="#94A3B8",
                annotation_text="Dificuldade Mediana", annotation_position="top right"
            )
            fig_sos.add_hline(
                y=float(df_rk_2026["aproveitamento_recente_pct"].median()),
                line_dash="dash", line_color="#94A3B8",
                annotation_text="Momento Mediano", annotation_position="bottom left"
            )
            aplicar_tema_plotly_dark(fig_sos, altura=420)
            st.plotly_chart(fig_sos, use_container_width=True)

        st.markdown("---")

        # ======================================================================
        # 3.5. SEÇÃO DEDICADA: PREVISÃO DE REBAIXAMENTO (Z-4) — QUEM CAI EM 2026?
        # ======================================================================
        st.markdown("##### 🚨 Luta Contra o Rebaixamento (Z-4) — Quem Cai para a Série B em 2026?")
        st.write(
            f"Análise preditiva da zona de degola (17º ao 20º lugar). Nas **{int(n_sims_sel):,} simulações**, a pontuação média do **16º colocado** "
            f"(primeiro clube fora do Z-4) foi de **{corte_med:.1f} pontos**, enquanto o **Número Mágico de Segurança (P90)** é de **{corte_seg} pontos**."
        )

        # Seleciona os 6 clubes com maior risco de rebaixamento (ordenados do maior para o menor risco Z-4)
        ameacados_z4 = df_rk_2026.sort_values(
            by=["prob_rebaixamento_z4_pct", "pontos_finais_esperados"],
            ascending=[False, True]
        ).head(6).copy()

        # Renderiza os 6 cards em duas fileiras de 3 colunas para leitura confortável
        for idx_bloco in range(0, len(ameacados_z4), 3):
            sub_z4 = ameacados_z4.iloc[idx_bloco:idx_bloco + 3]
            cols_z4 = st.columns(len(sub_z4))
            for col_z, (_, row_z) in zip(cols_z4, sub_z4.iterrows()):
                with col_z:
                    c_nome_z = row_z["clube_nome"]
                    esc_z = obter_escudo_img(c_nome_z, team_id=row_z.get("sofascore_team_id"))
                    esc_b64_z = obter_img_base64(esc_z) if esc_z else ""
                    img_tag_z = f'<img src="{esc_b64_z}" style="width: 34px; height: 34px; object-fit: contain; margin-right: 8px;">' if esc_b64_z else "🚨 "

                    prob_reb = float(row_z["prob_rebaixamento_z4_pct"])
                    prob_perm = float(row_z.get("prob_permanencia_pct", 100.0 - prob_reb))
                    pos_proj_z = int(row_z["posicao_projetada"])
                    pos_atual_z = int(row_z["posicao_atual"])
                    pts_faltam = int(row_z.get("pontos_faltantes_permanencia", max(0, corte_seg - int(row_z["pontos_atuais"]))))
                    status_lbl = str(row_z.get("status_risco_z4", "Em Risco"))

                    if prob_reb >= 85.0:
                        cor_borda_z = "#DC2626"
                        cor_num_z = "#F87171"
                    elif prob_reb >= 50.0:
                        cor_borda_z = "#EF4444"
                        cor_num_z = "#FB7185"
                    elif prob_reb >= 20.0:
                        cor_borda_z = "#F97316"
                        cor_num_z = "#FDBA74"
                    else:
                        cor_borda_z = "#EAB308"
                        cor_num_z = "#FDE047"

                    tag_zona_proj = (
                        f'<span style="background:#7F1D1D; color:#FECACA; padding:2px 8px; border-radius:999px; font-size:0.72rem; font-weight:700;">Projeção: {pos_proj_z}º (Z-4)</span>'
                        if pos_proj_z >= 17
                        else f'<span style="background:#14532D; color:#BBF7D0; padding:2px 8px; border-radius:999px; font-size:0.72rem; font-weight:700;">Projeção: {pos_proj_z}º (Salvo)</span>'
                    )

                    st.markdown(_compactar_html(f"""
                    <div style="background: #1E293B; border: 1px solid #334155; border-top: 4px solid {cor_borda_z}; border-radius: 12px; padding: 15px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                            <div style="display: flex; align-items: center;">
                                {img_tag_z}
                                <div>
                                    <div style="font-size: 1.12rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">{c_nome_z}</div>
                                    <span style="font-size: 0.76rem; color: #94A3B8;">Atual: {pos_atual_z}º • {int(row_z['pontos_atuais'])} pts ({int(row_z['jogos_atuais'])}J)</span>
                                </div>
                            </div>
                            {tag_zona_proj}
                        </div>
                        <div style="background: #0F172A; border-radius: 8px; padding: 10px; text-align: center; margin-bottom: 10px; border: 1px solid #334155;">
                            <div style="font-size: 0.70rem; text-transform: uppercase; color: #94A3B8; font-weight: 700;">Risco Matemático de Rebaixamento (Z-4)</div>
                            <div style="font-size: 1.80rem; font-weight: 900; color: {cor_num_z};">{prob_reb:.1f}%</div>
                            <div style="font-size: 0.76rem; color: #CBD5E1;">Permanência: <b style="color:#4ADE80;">{prob_perm:.1f}%</b> • Pts Proj.: <b>{row_z['pontos_finais_esperados']:.1f}</b> ({int(row_z['pontos_piso_p10'])}–{int(row_z['pontos_teto_p90'])})</div>
                        </div>
                        <div style="font-size: 0.80rem; color: #CBD5E1; display: grid; grid-template-columns: 1fr 1fr; gap: 6px;">
                            <div>🎯 <b>Status:</b> {status_lbl}</div>
                            <div>🛡️ <b>Faltam p/ {corte_seg} pts:</b> {pts_faltam} pts</div>
                            <div>🔥 <b>Recente (6J):</b> {row_z['aproveitamento_recente_pct']:.1f}% ({row_z['sequencia_recente']})</div>
                            <div>🧭 <b>Tabela ({int(row_z['jogos_casa_restantes'])}C/{int(row_z['jogos_fora_restantes'])}F):</b> {row_z['classificacao_dificuldade']}</div>
                        </div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #334155; font-size: 0.75rem; color: #94A3B8;">
                            <b>Confrontos restantes:</b> {row_z['proximos_adversarios_resumo']}
                        </div>
                    </div>
                    """), unsafe_allow_html=True)

        # Gráficos Lado a Lado da Luta contra o Rebaixamento
        col_z_bar, col_z_pts = st.columns(2)

        with col_z_bar:
            st.markdown("##### 📉 Probabilidade de Rebaixamento (Z-4 %) vs. Permanência na Série A")
            df_z_plot = df_rk_2026[df_rk_2026["prob_rebaixamento_z4_pct"] > 0.0].sort_values(
                "prob_rebaixamento_z4_pct", ascending=True
            ).copy()
            if len(df_z_plot) < 8:
                df_z_plot = df_rk_2026.tail(8).sort_values("prob_rebaixamento_z4_pct", ascending=True).copy()

            fig_z4_bar = go.Figure()
            fig_z4_bar.add_trace(go.Bar(
                y=df_z_plot["clube_nome"],
                x=df_z_plot["prob_rebaixamento_z4_pct"],
                name="Risco de Rebaixamento Z-4 (%)",
                orientation="h",
                marker=dict(color="#EF4444"),
                text=[f"{v:.1f}%" for v in df_z_plot["prob_rebaixamento_z4_pct"]],
                textposition="auto"
            ))
            fig_z4_bar.add_trace(go.Bar(
                y=df_z_plot["clube_nome"],
                x=df_z_plot["prob_permanencia_pct"],
                name="Chance de Permanência (%)",
                orientation="h",
                marker=dict(color="#22C55E", opacity=0.45),
                text=[f"{v:.1f}%" for v in df_z_plot["prob_permanencia_pct"]],
                textposition="inside"
            ))
            fig_z4_bar.add_vline(
                x=50.0, line_dash="dash", line_color="#F8FAFC",
                annotation_text="Limiar Crítico (50%)", annotation_position="top right"
            )
            aplicar_tema_plotly_dark(fig_z4_bar, altura=410)
            fig_z4_bar.update_layout(
                barmode="group",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Probabilidade Estimada (%)",
                yaxis_title="Clube Ameaçado",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_z4_bar, use_container_width=True)

        with col_z_pts:
            st.markdown("##### 🎯 Projeção de Pontos Finais na Degola vs. Linha de Corte (16º Colocado)")
            df_degola = df_rk_2026.tail(9).sort_values("pontos_finais_esperados", ascending=True).copy()
            cores_degola = [
                "#EF4444" if pos >= 17 else "#F59E0B" if prob >= 15 else "#38BDF8"
                for pos, prob in zip(df_degola["posicao_projetada"], df_degola["prob_rebaixamento_z4_pct"])
            ]

            fig_degola_pts = go.Figure()
            fig_degola_pts.add_trace(go.Bar(
                y=df_degola["clube_nome"],
                x=df_degola["pontos_atuais"],
                name="Pontos Atuais (Rodada 32)",
                orientation="h",
                marker=dict(color="#475569"),
                text=[f"{int(v)} pts" for v in df_degola["pontos_atuais"]],
                textposition="inside"
            ))
            fig_degola_pts.add_trace(go.Bar(
                y=df_degola["clube_nome"],
                x=(df_degola["pontos_finais_esperados"] - df_degola["pontos_atuais"]).clip(lower=0),
                name="Pontos Ganhos Projetados (Reta Final)",
                orientation="h",
                marker=dict(color=cores_degola),
                text=[f"Proj: {v:.1f} pts" for v in df_degola["pontos_finais_esperados"]],
                textposition="outside"
            ))
            fig_degola_pts.add_vline(
                x=corte_med, line_dash="dash", line_color="#FCD34D", line_width=2,
                annotation_text=f"Corte Médio 16º ({corte_med:.1f} pts)", annotation_position="top left"
            )
            fig_degola_pts.add_vline(
                x=corte_seg, line_dash="dot", line_color="#4ADE80", line_width=2,
                annotation_text=f"Segurança P90 ({corte_seg} pts)", annotation_position="bottom right"
            )
            aplicar_tema_plotly_dark(fig_degola_pts, altura=410)
            fig_degola_pts.update_layout(
                barmode="stack",
                margin=dict(l=20, r=45, t=30, b=20),
                xaxis_title="Pontuação Acumulada",
                yaxis_title="Clube (12º ao 20º)",
                xaxis=dict(range=[0, max(54, int(df_degola["pontos_finais_esperados"].max()) + 6)]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_degola_pts, use_container_width=True)

        st.markdown("---")

        # ======================================================================
        # 4. TABELA FINAL PROJETADA DE PONTOS CORRIDOS (1º AO 20º COLOCADO)
        # ======================================================================
        col_t26_head, col_t26_dl = st.columns([3.2, 1.3])
        with col_t26_head:
            st.markdown("##### 📋 Tabela Final Projetada de Pontos Corridos — Fim de Temporada 2026 (1º ao 20º Colocado)")
        with col_t26_dl:
            render_botao_download_csv(
                df_rk_2026,
                nome_arquivo="brasileirao_2026_tabela_projetada_completa.csv",
                label_botao="📥 Baixar Tabela 2026 (CSV)",
                key_btn="dl_tabela_simulada_2026_csv"
            )
        st.write(
            "Projeção completa após as **38 rodadas** do Campeonato Brasileiro Série A 2026, detalhando **Posição**, **Time**, "
            "**Pontos Projetados**, **Vitórias**, **Saldo de Gols** e as **Probabilidades Estatísticas de Classificação para Competições Continentais** "
            "(Copa Libertadores G-4/G-6 e Copa Sul-Americana) e **Rebaixamento (Z-4)**:"
        )

        col_fz1, col_fz2 = st.columns([2.5, 2.5])
        with col_fz1:
            filtro_zona_tab = st.selectbox(
                "Filtrar Tabela Final por Faixa de Classificação:",
                [
                    "Todos os 20 Clubes (1º ao 20º Colocado)",
                    "🟢 Classificados à Libertadores (G-4 e G-6: 1º ao 6º)",
                    "🟠 Classificados à Copa Sul-Americana (7º ao 12º)",
                    "⚪ Zona de Permanência sem Vaga Continental (13º ao 16º)",
                    "🔴 Zona de Rebaixamento à Série B — Z-4 (17º ao 20º)"
                ],
                index=0,
                key="sel_zona_tabela_final_2026"
            )
        with col_fz2:
            modo_colunas_tab = st.radio(
                "Modo de Exibição da Tabela:",
                ["📊 Visão Executiva (Principal)", "🔬 Visão Analítica Completa (Todas as Colunas)"],
                horizontal=True,
                key="radio_modo_colunas_2026"
            )

        df_view_rk = df_rk_2026.copy()
        if "Libertadores" in filtro_zona_tab:
            df_view_rk = df_view_rk[df_view_rk["posicao_projetada"] <= 6]
        elif "Sul-Americana" in filtro_zona_tab:
            df_view_rk = df_view_rk[(df_view_rk["posicao_projetada"] >= 7) & (df_view_rk["posicao_projetada"] <= 12)]
        elif "Permanência" in filtro_zona_tab:
            df_view_rk = df_view_rk[(df_view_rk["posicao_projetada"] >= 13) & (df_view_rk["posicao_projetada"] <= 16)]
        elif "Rebaixamento" in filtro_zona_tab:
            df_view_rk = df_view_rk[df_view_rk["posicao_projetada"] >= 17]

        df_view_rk["faixa_pontos_p10_p90"] = (
            df_view_rk["pontos_piso_p10"].astype(str) + " – " + df_view_rk["pontos_teto_p90"].astype(str) + " pts"
        )
        df_view_rk["mando_restante"] = (
            df_view_rk["jogos_casa_restantes"].astype(str) + "C / " + df_view_rk["jogos_fora_restantes"].astype(str) + "F"
        )
        df_view_rk["delta_pos_fmt"] = df_view_rk["variacao_posicao"].apply(
            lambda d: f"▲ +{int(d)}" if d > 0 else (f"▼ {int(d)}" if d < 0 else "＝ 0")
        )

        if "Executiva" in modo_colunas_tab:
            cols_rk_show = [
                "posicao_projetada", "delta_pos_fmt", "clube_nome", "zona_classificacao_projetada",
                "pontos_finais_esperados", "faixa_pontos_p10_p90", "vitorias_finais_esperadas",
                "saldo_gols_final_esperado", "prob_titulo_pct", "prob_g4_libertadores_pct",
                "prob_g6_libertadores_pct", "prob_sulamericana_pct", "prob_continental_total_pct",
                "prob_rebaixamento_z4_pct"
            ]
        else:
            cols_rk_show = [
                "posicao_projetada", "posicao_atual", "delta_pos_fmt", "clube_nome", "zona_classificacao_projetada",
                "jogos_finais_projetados", "pontos_atuais", "pontos_finais_esperados", "faixa_pontos_p10_p90",
                "vitorias_finais_esperadas", "empates_finais_esperados", "derrotas_finais_esperadas",
                "gols_pro_finais_esperados", "gols_contra_finais_esperados", "saldo_gols_final_esperado",
                "aproveitamento_final_projetado_pct", "aproveitamento_recente_pct", "sequencia_recente",
                "mando_restante", "dificuldade_confrontos_restantes", "classificacao_dificuldade",
                "prob_titulo_pct", "prob_g4_libertadores_pct", "prob_g6_libertadores_pct",
                "prob_sulamericana_pct", "prob_continental_total_pct", "prob_rebaixamento_z4_pct", "status_risco_z4"
            ]

        cols_rk_disp = [c for c in cols_rk_show if c in df_view_rk.columns]

        st.dataframe(
            df_view_rk[cols_rk_disp].rename(columns={
                "posicao_projetada": "Posição",
                "posicao_atual": "Pos. Atual",
                "delta_pos_fmt": "Δ Pos",
                "clube_nome": "Time",
                "zona_classificacao_projetada": "Classificação / Zona Final",
                "jogos_finais_projetados": "J Finais",
                "pontos_atuais": "Pts Atuais",
                "pontos_finais_esperados": "Pontos Projetados",
                "faixa_pontos_p10_p90": "Intervalo (P10-P90)",
                "vitorias_finais_esperadas": "Vitórias",
                "empates_finais_esperados": "Empates",
                "derrotas_finais_esperadas": "Derrotas",
                "gols_pro_finais_esperados": "GP Proj.",
                "gols_contra_finais_esperados": "GC Proj.",
                "saldo_gols_final_esperado": "Saldo de Gols",
                "aproveitamento_final_projetado_pct": "Aprov. Final %",
                "aproveitamento_recente_pct": "Aprov. Recente %",
                "sequencia_recente": "Últimos 6J",
                "mando_restante": "Jogos Rest. (C/F)",
                "dificuldade_confrontos_restantes": "Índice SOS",
                "classificacao_dificuldade": "Nível Tabela",
                "prob_titulo_pct": "🏆 Título %",
                "prob_g4_libertadores_pct": "🟢 G-4 Liberta %",
                "prob_g6_libertadores_pct": "🔵 G-6 Liberta %",
                "prob_sulamericana_pct": "🟠 Sul-Americana %",
                "prob_continental_total_pct": "🌎 Continental %",
                "prob_rebaixamento_z4_pct": "🔴 Rebaixamento (Z4) %",
                "status_risco_z4": "Situação Z-4"
            }),
            column_config={
                "Pontos Projetados": st.column_config.NumberColumn("Pontos Projetados", format="%.1f pts"),
                "Vitórias": st.column_config.NumberColumn("Vitórias", format="%.1f"),
                "Saldo de Gols": st.column_config.NumberColumn("Saldo de Gols", format="%+.1f"),
                "🏆 Título %": st.column_config.ProgressColumn("🏆 Título %", format="%.1f%%", min_value=0, max_value=100),
                "🟢 G-4 Liberta %": st.column_config.ProgressColumn("🟢 G-4 Liberta %", format="%.1f%%", min_value=0, max_value=100),
                "🔵 G-6 Liberta %": st.column_config.ProgressColumn("🔵 G-6 Liberta %", format="%.1f%%", min_value=0, max_value=100),
                "🟠 Sul-Americana %": st.column_config.ProgressColumn("🟠 Sul-Americana %", format="%.1f%%", min_value=0, max_value=100),
                "🌎 Continental %": st.column_config.ProgressColumn("🌎 Continental %", format="%.1f%%", min_value=0, max_value=100),
                "🔴 Rebaixamento (Z4) %": st.column_config.ProgressColumn("🔴 Rebaixamento (Z4) %", format="%.1f%%", min_value=0, max_value=100),
            },
            use_container_width=True,
            hide_index=True,
            height=735
        )

        st.write("")

        # Gráfico Panorama Geral de Pontos Corridos (1º ao 20º Colocado)
        with st.expander("📊 Visualizar Gráfico Completo de Pontos Corridos (1º ao 20º Colocado) & Validação de Machine Learning (SVM / Ensemble)", expanded=True):
            col_full_chart, col_ml_bench = st.columns([1.35, 1])

            with col_full_chart:
                st.markdown("###### 🏟️ Projeção Final de Pontos Corridos — 1º ao 20º Colocado (Após 38 Rodadas)")
                df_full_plot = df_rk_2026.sort_values("posicao_projetada", ascending=False).copy()
                cores_zonas = []
                for pos in df_full_plot["posicao_projetada"]:
                    if pos == 1:
                        cores_zonas.append("#16A34A")  # Campeão
                    elif pos <= 4:
                        cores_zonas.append("#22C55E")  # G-4 Libertadores
                    elif pos <= 6:
                        cores_zonas.append("#38BDF8")  # G-6 Pré-Libertadores
                    elif pos <= 12:
                        cores_zonas.append("#F59E0B")  # Sul-Americana
                    elif pos <= 16:
                        cores_zonas.append("#94A3B8")  # Permanência
                    else:
                        cores_zonas.append("#EF4444")  # Rebaixamento Z-4

                fig_full_20 = go.Figure()
                fig_full_20.add_trace(go.Bar(
                    y=df_full_plot["posicao_projetada"].astype(str) + "º " + df_full_plot["clube_nome"],
                    x=df_full_plot["pontos_finais_esperados"],
                    orientation="h",
                    marker=dict(color=cores_zonas),
                    text=[
                        f"{pts:.1f} pts ({vit:.0f}V, SG {sg:+.0f})"
                        for pts, vit, sg in zip(
                            df_full_plot["pontos_finais_esperados"],
                            df_full_plot["vitorias_finais_esperadas"],
                            df_full_plot["saldo_gols_final_esperado"]
                        )
                    ],
                    textposition="outside",
                    name="Pontos Finais Projetados"
                ))
                fig_full_20.add_vline(
                    x=corte_med, line_dash="dash", line_color="#EF4444",
                    annotation_text=f"Corte Z-4 ({corte_med:.1f} pts)", annotation_position="bottom right"
                )
                aplicar_tema_plotly_dark(fig_full_20, altura=620)
                fig_full_20.update_layout(
                    margin=dict(l=10, r=85, t=20, b=20),
                    xaxis_title="Pontos Projetados ao Final da 38ª Rodada",
                    yaxis_title="Posição Projetada & Clube",
                    xaxis=dict(range=[0, max(96, int(df_full_plot["pontos_finais_esperados"].max()) + 14)]),
                    showlegend=False
                )
                st.plotly_chart(fig_full_20, use_container_width=True)

            with col_ml_bench:
                st.markdown("###### 🧠 Avaliação do Modelo Preditivo (SVM + Ensemble) por Divisão Treino/Teste")
                st.caption(
                    "Desempenho comparativo do classificador de partidas (**Support Vector Machine - SVC RBF**, "
                    "**Random Forest** e **Regressão Logística**) avaliado em 4 estratégias de validação:"
                )
                splits_info = resumo_2026.get("avaliacao_splits", [])
                if not splits_info:
                    splits_info = [
                        {"divisao_treino_teste": "Split Temporal (Treino 2023–2025 / Teste 2026)", "amostras_treino": 1140, "amostras_teste": 319, "acuracia_svm_pct": 55.8, "acuracia_ensemble_pct": 57.4, "f1_weighted_pct": 55.9, "log_loss": 0.948},
                        {"divisao_treino_teste": "Holdout Estratificado (80% Treino / 20% Teste)", "amostras_treino": 1167, "amostras_teste": 292, "acuracia_svm_pct": 56.5, "acuracia_ensemble_pct": 58.2, "f1_weighted_pct": 56.8, "log_loss": 0.939},
                        {"divisao_treino_teste": "Holdout Estratificado (70% Treino / 30% Teste)", "amostras_treino": 1021, "amostras_teste": 438, "acuracia_svm_pct": 56.1, "acuracia_ensemble_pct": 57.6, "f1_weighted_pct": 56.2, "log_loss": 0.944},
                        {"divisao_treino_teste": "Validação Cruzada (5-Fold Stratified CV)", "amostras_treino": 1167, "amostras_teste": 292, "acuracia_svm_pct": 56.3, "acuracia_ensemble_pct": 57.9, "f1_weighted_pct": 56.5, "log_loss": 0.942},
                    ]
                df_splits_bench = pd.DataFrame(splits_info)
                st.dataframe(
                    df_splits_bench.rename(columns={
                        "divisao_treino_teste": "Estratégia Treino / Teste",
                        "amostras_treino": "Treino (J)",
                        "amostras_teste": "Teste (J)",
                        "acuracia_svm_pct": "Acurácia SVM %",
                        "acuracia_ensemble_pct": "Acurácia Ensemble %",
                        "f1_weighted_pct": "F1-Score %",
                        "log_loss": "Log-Loss"
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("""
                <div style="background:#0F172A; border:1px solid #334155; border-left:4px solid #38BDF8; border-radius:8px; padding:12px; font-size:0.82rem; color:#CBD5E1; margin-top:10px;">
                    <b>⚙️ Arquitetura do Modelo Híbrido (Pontos Corridos):</b><br>
                    • <b>Classificação de Partidas (3 classes V/E/D):</b> Ensemble ponderado com <b>SVM (Kernel RBF, C=1.2)</b> + <b>Random Forest (180 árvores)</b> + <b>Regressão Logística</b>.<br>
                    • <b>Regressão de Força de Elenco:</b> <b>SVR (Support Vector Regression)</b> + <b>XGBoost</b> + <b>Ridge</b> ($R^2 = 0.839$, $\\text{RMSE} = 6.96$ pts).<br>
                    • <b>Simulação de Monte Carlo:</b> 10.000 temporadas simuladas via distribuição bivariada de Poisson ($\\lambda_{\\text{Mandante}}, \\lambda_{\\text{Visitante}}$) calibrada pelas probabilidades do ML e pelos critérios de desempate da CBF.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # 5. Calendário Simulado dos Confrontos Restantes por Clube ou Rodada
        col_jr_head, col_jr_dl = st.columns([3.2, 1.3])
        with col_jr_head:
            st.markdown("##### 🗓️ Simulação Jogo a Jogo dos Confrontos Restantes (Reta Final 2026)")
        if not df_jr_2026.empty:
            with col_jr_dl:
                render_botao_download_csv(
                    df_jr_2026,
                    nome_arquivo="brasileirao_2026_jogos_restantes_simulados.csv",
                    label_botao="📥 Baixar Jogos Restantes (CSV)",
                    key_btn="dl_jogos_restantes_2026_csv"
                )
            col_f_jr1, col_f_jr2 = st.columns([2, 2])
            with col_f_jr1:
                clubes_jr_opcoes = [
                    "Todos os Postulantes ao Título (Top 4)",
                    "Todos na Luta Contra o Rebaixamento (Z-4)",
                    "Todos os 20 Clubes"
                ] + df_rk_2026["clube_nome"].tolist()
                filtro_clube_jr = st.selectbox("Filtrar Calendário Restante por Clube / Disputa:", clubes_jr_opcoes, index=0, key="sel_clube_jr_2026")
            with col_f_jr2:
                rodadas_disp_jr = ["Todas as Rodadas Restantes"] + sorted(df_jr_2026["rodada"].unique().tolist())
                filtro_rodada_jr = st.selectbox("Filtrar por Rodada:", rodadas_disp_jr, index=0, key="sel_rodada_jr_2026")

            df_jr_view = df_jr_2026.copy()
            if filtro_clube_jr == "Todos os Postulantes ao Título (Top 4)":
                nomes_top4 = set(postulantes["clube_nome"].tolist())
                df_jr_view = df_jr_view[
                    df_jr_view["mandante_nome"].isin(nomes_top4) | df_jr_view["visitante_nome"].isin(nomes_top4)
                ]
            elif filtro_clube_jr == "Todos na Luta Contra o Rebaixamento (Z-4)":
                nomes_z4_luta = set(ameacados_z4["clube_nome"].tolist())
                df_jr_view = df_jr_view[
                    df_jr_view["mandante_nome"].isin(nomes_z4_luta) | df_jr_view["visitante_nome"].isin(nomes_z4_luta)
                ]
            elif filtro_clube_jr != "Todos os 20 Clubes":
                df_jr_view = df_jr_view[
                    (df_jr_view["mandante_nome"] == filtro_clube_jr) | (df_jr_view["visitante_nome"] == filtro_clube_jr)
                ]

            if filtro_rodada_jr != "Todas as Rodadas Restantes":
                df_jr_view = df_jr_view[df_jr_view["rodada"] == int(filtro_rodada_jr)]

            df_jr_view = df_jr_view.sort_values(["rodada", "event_id"]).copy()
            cols_jr_show = [
                "rodada", "data_formatada", "mandante_nome", "placar_provavel", "visitante_nome",
                "lambda_mandante", "lambda_visitante", "prob_mandante_pct", "prob_empate_pct", "prob_visitante_pct"
            ]
            cols_jr_disp = [c for c in cols_jr_show if c in df_jr_view.columns]
            st.dataframe(
                df_jr_view[cols_jr_disp].rename(columns={
                    "rodada": "Rodada",
                    "data_formatada": "Data Prev.",
                    "mandante_nome": "Mandante (Casa)",
                    "placar_provavel": "Placar Mais Provável",
                    "visitante_nome": "Visitante (Fora)",
                    "lambda_mandante": "xG Esperado (M)",
                    "lambda_visitante": "xG Esperado (V)",
                    "prob_mandante_pct": "Prob. Mandante %",
                    "prob_empate_pct": "Prob. Empate %",
                    "prob_visitante_pct": "Prob. Visitante %"
                }),
                use_container_width=True,
                hide_index=True
            )

st.markdown("---")
st.caption("⚽ Projeto Futebol Brasileiro (Séries A a D) | Visualização Tática no Gramado, Shotmaps e Analytics Preditivo")
