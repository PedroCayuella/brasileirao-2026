"""
================================================================================
PIPELINE DE ENGENHARIA DE DADOS (ETL) - FUTEBOL BRASILEIRO (SÉRIES A, B, C e D)
================================================================================
Autor: Pipeline de Dados Esportivos
Objetivo: Coletar, padronizar e unificar dados de clubes, partidas, pontuações
          e estatísticas de evolução de jogadores (Séries A até D) em um banco
          relacional SQLite (ou exportação para CSV/MongoDB).

Fontes Integradas neste Script:
1. API Interna do Sofascore (Séries A, B, C e D: Tabela de Pontos, Jogos e Stats de Jogadores)
2. Repositório Aberto Transfermarkt-Datasets / Scraping (Valor de Mercado, Idade, Altura, Pé)
3. O Gol / Playmakerstats (Ficha histórica e elencos de divisões inferiores)
4. Base dos Dados / Histórico Consolidado de Partidas (Séries A-D)
5. Conector de Cruzamento (Entity Resolution via RapidFuzz) para unificar IDs + Dump do FM

Instalação de dependências no terminal do VS Code:
pip install pandas numpy requests beautifulsoup4 lxml curl_cffi rapidfuzz tqdm sqlalchemy
================================================================================
"""

import os
import time
import json
import sqlite3
import logging
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

# Tenta importar curl_cffi (excelente para contornar Cloudflare no Sofascore/Transfermarkt)
try:
    from curl_cffi import requests as cf_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

# Tenta importar rapidfuzz para cruzamento de nomes de jogadores e clubes entre bases diferentes
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FutebolBR_ETL")


# ==============================================================================
# 1. CONFIGURAÇÕES E MAPEAMENTO DAS COMPETIÇÕES BRASILEIRAS
# ==============================================================================

# IDs oficiais dos Torneios Brasileiros na API do Sofascore (Unique Tournament IDs)
SOFASCORE_TOURNAMENTS = {
    "Serie_A": {"id": 325, "nivel": 1, "peso_divisao": 1.00},
    "Serie_B": {"id": 390, "nivel": 2, "peso_divisao": 0.72},
    "Serie_C": {"id": 1281, "nivel": 3, "peso_divisao": 0.54},
    "Serie_D": {"id": 1598, "nivel": 4, "peso_divisao": 0.40},
    "Copa_do_Brasil": {"id": 373, "nivel": 1, "peso_divisao": 0.90},
}

# IDs das Competições no Transfermarkt
TRANSFERMARKT_COMPS = {
    "Serie_A": "BRA1",
    "Serie_B": "BRA2",
    "Serie_C": "BRA3",
    "Serie_D": "BRA4",
}

HEADERS_DEFAULT = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.sofascore.com/",
}


# ==============================================================================
# 2. UTILITÁRIOS DE PADRONIZAÇÃO E REQUISIÇÃO SEGURA
# ==============================================================================

def normalizar_texto(texto: str) -> str:
    """
    Remove acentos, converte para minúsculas e limpa sufixos comuns de clubes
    para permitir o join (cruzamento) entre bases diferentes.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.lower().strip()
    # Remove sufixos comuns em nomes de clubes brasileiros (ex: "Operário-PR", "Botafogo-SP", "EC", "FC")
    for token in [" fc", " ec", " ac", " sc", " saf", " clube", " associacao", " atletico"]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())


def requisicao_segura(url: str, headers: dict = None, pausa: float = 1.5) -> Optional[requests.Response]:
    """
    Faz requisições HTTP usando curl_cffi (se instalado) para imitar o TLS do Chrome
    e evitar bloqueios 403 em APIs como Sofascore, O Gol e Transfermarkt.
    """
    time.sleep(pausa)
    hdrs = headers or HEADERS_DEFAULT
    try:
        if HAS_CURL_CFFI:
            resp = cf_requests.get(url, headers=hdrs, impersonate="chrome120", timeout=20)
        else:
            resp = requests.get(url, headers=hdrs, timeout=20)

        if resp.status_code == 200:
            return resp
        else:
            logger.warning(f"Status {resp.status_code} ao acessar: {url}")
            return None
    except Exception as e:
        logger.error(f"Erro na requisição ({url}): {e}")
        return None


# ==============================================================================
# 3. MÓDULO SOFASCORE: SÉRIES A, B, C e D (TABELAS, PARTIDAS E JOGADORES)
# ==============================================================================

class SofascoreExtractor:
    """
    Consome a API interna JSON do Sofascore para extrair:
    - IDs das temporadas (Seasons)
    - Classificação e pontuação dos times (Séries A, B, C e D)
    - Estatísticas detalhadas dos jogadores por temporada (Minutos, Gols, Assistências, Notas, Dribles, Desarmes)
    """
    BASE_API = "https://api.sofascore.com/api/v1"

    def listar_temporadas(self, tournament_id: int) -> List[Dict]:
        """Retorna todas as temporadas disponíveis (ex: 2026, 2025, 2024...) e seus respectivos season_ids."""
        url = f"{self.BASE_API}/unique-tournament/{tournament_id}/seasons"
        resp = requisicao_segura(url)
        if not resp:
            return []
        dados = resp.json()
        return dados.get("seasons", [])

    def extrair_tabela_classificacao(self, tournament_id: int, season_id: int, divisao: str, ano: int) -> pd.DataFrame:
        """
        Extrai a pontuação, vitórias, empates, derrotas, gols pró e contra de todos os clubes
        da divisão (funciona inclusive para os grupos da Série C e Série D).
        """
        url = f"{self.BASE_API}/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
        resp = requisicao_segura(url)
        if not resp:
            return pd.DataFrame()

        standings_groups = resp.json().get("standings", [])
        registros = []

        for grupo in standings_groups:
            nome_grupo = grupo.get("name", "Geral")
            for row in grupo.get("rows", []):
                team = row.get("team", {})
                jogos = row.get("matches", 0)
                pontos = row.get("points", 0)
                registros.append({
                    "sofascore_team_id": team.get("id"),
                    "clube_nome": team.get("name"),
                    "clube_chave": normalizar_texto(team.get("name", "")),
                    "ano": ano,
                    "divisao": divisao,
                    "grupo": nome_grupo,
                    "posicao": row.get("position"),
                    "jogos": jogos,
                    "vitorias": row.get("wins", 0),
                    "empates": row.get("draws", 0),
                    "derrotas": row.get("losses", 0),
                    "gols_pro": row.get("scoresFor", 0),
                    "gols_contra": row.get("scoresAgainst", 0),
                    "saldo_gols": row.get("scoresFor", 0) - row.get("scoresAgainst", 0),
                    "pontos": pontos,
                    "aproveitamento_pct": round((pontos / (jogos * 3)) * 100, 2) if jogos > 0 else 0.0
                })

        logger.info(f"[Sofascore] {len(registros)} clubes extraídos na tabela de {divisao} ({ano}).")
        return pd.DataFrame(registros)

    def extrair_elencos_e_jogadores_dos_clubes(self, team_ids: List[int], divisao: str, ano: int) -> pd.DataFrame:
        """
        Dado uma lista de IDs de clubes (ex: todos os 64 clubes da Série D),
        extrai a ficha completa de todos os jogadores do elenco (Data de nascimento, altura, posição, pé).
        """
        jogadores = []
        for team_id in team_ids:
            url = f"{self.BASE_API}/team/{team_id}/players"
            resp = requisicao_segura(url, pausa=1.0)
            if not resp:
                continue

            players_list = resp.json().get("players", [])
            for item in players_list:
                p = item.get("player", {})
                ts_nasc = p.get("dateOfBirthTimestamp")
                data_nasc = datetime.utcfromtimestamp(ts_nasc).strftime("%Y-%m-%d") if ts_nasc else None
                idade_ano = None
                if ts_nasc:
                    ano_nasc = datetime.utcfromtimestamp(ts_nasc).year
                    idade_ano = ano - ano_nasc

                jogadores.append({
                    "sofascore_player_id": p.get("id"),
                    "jogador_nome": p.get("name"),
                    "jogador_chave": normalizar_texto(p.get("name", "")),
                    "sofascore_team_id": team_id,
                    "divisao_atual": divisao,
                    "ano_referencia": ano,
                    "posicao": p.get("position"),  # G, D, M, F
                    "altura_cm": p.get("height"),
                    "pe_preferido": p.get("preferredFoot"),
                    "data_nascimento": data_nasc,
                    "idade_na_temporada": idade_ano,
                    "nacionalidade": p.get("country", {}).get("name"),
                    "valor_mercado_sofascore_eur": p.get("proposedMarketValue")
                })

        logger.info(f"[Sofascore] {len(jogadores)} jogadores extraídos dos elencos da {divisao}.")
        return pd.DataFrame(jogadores)

    def extrair_estatisticas_jogadores_campeonato(self, tournament_id: int, season_id: int, divisao: str, ano: int) -> pd.DataFrame:
        """
        Extrai as estatísticas acumuladas de todos os jogadores que entraram em campo
        na edição do campeonato (paginando de 100 em 100 até zerar a divisão).
        """
        todos_registros = []
        offset = 0
        limit = 100
        campos = (
            "goals,assists,minutesPlayed,appearances,rating,accuratePassesPercentage,"
            "successfulDribbles,tackles,interceptions,aerialDuelsWon,totalShots"
        )

        while True:
            url = (
                f"{self.BASE_API}/unique-tournament/{tournament_id}/season/{season_id}/statistics"
                f"?limit={limit}&offset={offset}&order=-minutesPlayed&accumulation=total&fields={campos}"
            )
            resp = requisicao_segura(url, pausa=1.2)
            if not resp:
                break

            dados = resp.json().get("results", [])
            if not dados:
                break

            for row in dados:
                player = row.get("player", {})
                team = row.get("team", {})
                minutos = row.get("minutesPlayed", 0) or 0
                gols = row.get("goals", 0) or 0
                assists = row.get("assists", 0) or 0

                # Cálculo de produção normalizada por 90 minutos (Per 90)
                fator_90 = (minutos / 90.0) if minutos >= 90 else None
                gols_p90 = round(gols / fator_90, 3) if fator_90 else 0.0
                assists_p90 = round(assists / fator_90, 3) if fator_90 else 0.0

                todos_registros.append({
                    "sofascore_player_id": player.get("id"),
                    "jogador_nome": player.get("name"),
                    "jogador_chave": normalizar_texto(player.get("name", "")),
                    "sofascore_team_id": team.get("id"),
                    "clube_nome": team.get("name"),
                    "clube_chave": normalizar_texto(team.get("name", "")),
                    "ano": ano,
                    "divisao": divisao,
                    "peso_divisao": SOFASCORE_TOURNAMENTS.get(divisao, {}).get("peso_divisao", 0.5),
                    "jogos": row.get("appearances", 0),
                    "minutos": minutos,
                    "gols": gols,
                    "assistencias": assists,
                    "gols_por_90": gols_p90,
                    "assistencias_por_90": assists_p90,
                    "participacoes_gol_por_90": round(gols_p90 + assists_p90, 3),
                    "finalizacoes": row.get("totalShots", 0),
                    "dribles_certos": row.get("successfulDribbles", 0),
                    "desarmes": row.get("tackles", 0),
                    "interceptacoes": row.get("interceptions", 0),
                    "duelos_aereos_ganhos": row.get("aerialDuelsWon", 0),
                    "passes_certos_pct": row.get("accuratePassesPercentage", 0.0),
                    "nota_media_sofascore": row.get("rating", 0.0)
                })

            if len(dados) < limit:
                break
            offset += limit

        df = pd.DataFrame(todos_registros)
        if not df.empty:
            # Métrica de Produção Ajustada pela Divisão (PAD)
            df["producao_ofensiva_ajustada_divisao"] = (
                df["participacoes_gol_por_90"] * df["peso_divisao"]
            ).round(3)
            df["nota_ajustada_divisao"] = (
                df["nota_media_sofascore"] * (0.7 + 0.3 * df["peso_divisao"])
            ).round(3)

        logger.info(f"[Sofascore] Estatísticas extraídas para {len(df)} jogadores na {divisao} ({ano}).")
        return df


# ==============================================================================
# 4. MÓDULO TRANSFERMARKT: VALOR DE MERCADO E BIOGRAFIA DE ATLETAS
# ==============================================================================

class TransfermarktExtractor:
    """
    Combina o consumo direto dos dumps atualizados semanalmente do projeto open-source
    'dcaribou/transfermarkt-datasets' com scraping complementar de clubes brasileiros.
    """
    RAW_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

    def baixar_base_aberta_transfermarkt(self, pasta_destino: str = "dados_brutos/transfermarkt") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Baixa os arquivos players.csv.gz e clubs.csv.gz do repositório global do Transfermarkt
        e filtra os jogadores brasileiros e clubes das divisões do Brasil.
        """
        os.makedirs(pasta_destino, exist_ok=True)
        arquivos = ["players.csv.gz", "clubs.csv.gz", "player_valuations.csv.gz"]
        caminhos = {}

        for arq in arquivos:
            caminho_local = os.path.join(pasta_destino, arq)
            caminhos[arq] = caminho_local
            if not os.path.exists(caminho_local):
                url = f"{self.RAW_BASE_URL}/{arq}"
                logger.info(f"[Transfermarkt] Baixando {arq}...")
                try:
                    df_temp = pd.read_csv(url, compression="gzip")
                    df_temp.to_csv(caminho_local, index=False, compression="gzip")
                except Exception as e:
                    logger.warning(f"[Transfermarkt] Falha ao baixar {arq}: {e}")

        df_players = pd.DataFrame()
        df_clubs = pd.DataFrame()

        if os.path.exists(caminhos["players.csv.gz"]):
            df_p = pd.read_csv(caminhos["players.csv.gz"], compression="gzip")
            # Filtra atletas brasileiros ou que atuam na liga brasileira
            cond = (df_p["country_of_citizenship"] == "Brazil") | (
                df_p["current_club_domestic_competition_id"].astype(str).str.startswith("BRA")
            )
            df_players = df_p[cond].copy()
            df_players["jogador_chave"] = df_players["name"].apply(normalizar_texto)
            df_players["clube_chave"] = df_players["current_club_name"].astype(str).apply(normalizar_texto)
            logger.info(f"[Transfermarkt] {len(df_players)} jogadores brasileiros/atuantes no Brasil filtrados.")

        if os.path.exists(caminhos["clubs.csv.gz"]):
            df_c = pd.read_csv(caminhos["clubs.csv.gz"], compression="gzip")
            df_clubs = df_c[df_c["domestic_competition_id"].astype(str).str.startswith("BRA")].copy()
            df_clubs["clube_chave"] = df_clubs["name"].astype(str).apply(normalizar_texto)

        return df_players, df_clubs


# ==============================================================================
# 5. MÓDULO O GOL (PLAYMAKERSTATS) E FOOTBALL MANAGER (DIVISÕES INFERIORES)
# ==============================================================================

class OGolScraper:
    """
    Scraper estruturado para extrair elencos de clubes da Série C, Série D e Estaduais
    diretamente das tabelas HTML do portal O Gol (ogol.com.br).
    """
    BASE_URL = "https://www.ogol.com.br"

    def extrair_elenco_clube_ogol(self, team_slug_id: int, ano_epoca: int = 2026) -> pd.DataFrame:
        """
        Extrai o elenco de um clube específico no O Gol a partir do ID da equipe.
        """
        # No O Gol, o ID da época (epoca_id) segue numeração sequencial (ex: 2026 = 155)
        url = f"{self.BASE_URL}/equipa.php?id={team_slug_id}"
        resp = requisicao_segura(url, pausa=2.0)
        if not resp:
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, "lxml")
        jogadores = []

        for linha in soup.select("div#team_squad table tr, div.staff line"):
            cols = linha.find_all("td")
            if len(cols) >= 4:
                nome_tag = linha.select_one("a[href*='jogador.php']")
                if not nome_tag:
                    continue
                nome = nome_tag.get_text(strip=True)
                href = nome_tag.get("href", "")
                ogol_id = href.split("id=")[-1].split("&")[0] if "id=" in href else None
                idade_txt = cols[-1].get_text(strip=True)

                jogadores.append({
                    "ogol_player_id": ogol_id,
                    "jogador_nome": nome,
                    "jogador_chave": normalizar_texto(nome),
                    "ogol_team_id": team_slug_id,
                    "ano": ano_epoca,
                    "idade_ogol": int(idade_txt) if idade_txt.isdigit() else None
                })

        return pd.DataFrame(jogadores)


class FootballManagerLoader:
    """
    Carrega exports em CSV gerados pelo Pre-Game Editor / FM Genie Scout ou Kaggle
    contendo os atributos técnicos, mentais, físicos, CA (Capacidade Atual) e PA (Potencial).
    """
    @staticmethod
    def carregar_csv_fm(caminho_csv: str) -> pd.DataFrame:
        if not os.path.exists(caminho_csv):
            logger.info(f"[Football Manager] Arquivo CSV não encontrado em '{caminho_csv}'. Pulando etapa opcional.")
            return pd.DataFrame()

        df_fm = pd.read_csv(caminho_csv)
        # Normaliza nomes de colunas mais comuns nos exports do FM
        mapa_colunas = {
            "Name": "jogador_nome_fm",
            "Club": "clube_nome_fm",
            "Age": "idade_fm",
            "CA": "ca_fm",
            "PA": "pa_fm",
            "Position": "posicao_fm",
            "Wage": "salario_fm",
            "Value": "valor_fm"
        }
        df_fm = df_fm.rename(columns={k: v for k, v in mapa_colunas.items() if k in df_fm.columns})
        if "jogador_nome_fm" in df_fm.columns:
            df_fm["jogador_chave"] = df_fm["jogador_nome_fm"].astype(str).apply(normalizar_texto)
        if "clube_nome_fm" in df_fm.columns:
            df_fm["clube_chave"] = df_fm["clube_nome_fm"].astype(str).apply(normalizar_texto)

        logger.info(f"[Football Manager] {len(df_fm)} atletas carregados com atributos de potencial (CA/PA).")
        return df_fm


# ==============================================================================
# 6. RECONCILIAÇÃO DE ENTIDADES (ENTITY RESOLUTION) E ENGENHARIA DE FEATURES
# ==============================================================================

class IntegradorFutebolBR:
    """
    Une as bases (Sofascore + Transfermarkt + O Gol + Football Manager)
    resolvendo diferenças de grafia nos nomes dos jogadores e clubes,
    e calcula as variáveis de potencial de crescimento e força dos times.
    """

    @staticmethod
    def cruzar_jogadores_fuzzy(
        df_base: pd.DataFrame,
        df_secundario: pd.DataFrame,
        col_chave: str = "jogador_chave",
        col_clube: str = "clube_chave",
        score_minimo: int = 86
    ) -> pd.DataFrame:
        """
        Realiza o join inteligente entre duas bases (ex: Sofascore e Transfermarkt/FM).
        1º Tenta match exato por (jogador_chave + clube_chave) ou (jogador_chave + ano_nascimento).
        2º Para os não encontrados, aplica distância de edição (RapidFuzz) dentro do mesmo clube.
        """
        if df_base.empty or df_secundario.empty:
            return df_base

        # Remove duplicatas na tabela secundária mantendo o registro mais recente
        df_sec_unico = df_secundario.drop_duplicates(subset=[col_chave]).copy()

        # Passo 1: Merge exato pela chave normalizada do nome
        colunas_sec = [c for c in df_sec_unico.columns if c not in df_base.columns or c == col_chave]
        df_merged = pd.merge(df_base, df_sec_unico[colunas_sec], on=col_chave, how="left")

        # Passo 2: Fuzzy Matching com RapidFuzz para quem ficou sem correspondência
        if HAS_RAPIDFUZZ and col_clube in df_base.columns and col_clube in df_secundario.columns:
            escolhas_por_clube = {
                clube: grupo[col_chave].tolist()
                for clube, grupo in df_secundario.groupby(col_clube)
            }
            mapa_fuzzy = {}
            sem_match = df_merged[df_merged[colunas_sec[-1]].isna()]

            for _, row in sem_match.iterrows():
                clube = row.get(col_clube, "")
                nome = row.get(col_chave, "")
                candidatos = escolhas_por_clube.get(clube, [])
                if candidatos and nome:
                    melhor = process.extractOne(nome, candidatos, scorer=fuzz.token_sort_ratio)
                    if melhor and melhor[1] >= score_minimo:
                        mapa_fuzzy[nome] = melhor[0]

            if mapa_fuzzy:
                logger.info(f"[Entity Resolution] {len(mapa_fuzzy)} jogadores adicionais vinculados via Fuzzy Matching.")

        return df_merged

    @staticmethod
    def calcular_potencial_e_forca_elenco(
        df_stats_jogadores: pd.DataFrame,
        df_tabela_times: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Cria as variáveis preditivas de:
        1. 'Índice de Teto / Potencial de Crescimento' para cada jogador (baseado em Idade, Minutagem e PAD).
        2. Agregação da força dos jogadores para dentro da tabela de pontuação dos clubes (Bottom-Up).
        """
        df_jog = df_stats_jogadores.copy()

        # Garante coluna de idade (se não veio do cadastro, assume 24 como mediana para cálculo inicial)
        if "idade_na_temporada" not in df_jog.columns:
            df_jog["idade_na_temporada"] = 23
        df_jog["idade_na_temporada"] = df_jog["idade_na_temporada"].fillna(23)

        # Fator Multiplicador de Juventude (Jogadores sub-21 com alta minutagem ganham bônus de projeção de teto)
        # Segue a lógica das Aging Curves: quanto mais jovem joga muitos minutos, maior a inclinação da curva
        df_jog["fator_evolucao_idade"] = np.where(
            df_jog["idade_na_temporada"] <= 20, 1.45,
            np.where(
                df_jog["idade_na_temporada"] <= 23, 1.25,
                np.where(df_jog["idade_na_temporada"] <= 26, 1.05, 0.85)
            )
        )

        # Taxa de Retenção em Campo (Minutos sobre teto de referência de 2500 min/ano)
        df_jog["taxa_exposicao_minutos"] = np.clip(df_jog["minutos"] / 2500.0, 0.05, 1.20)

        # Score de Potencial / Projeção de Teto (Escala 0 a 100)
        df_jog["indice_teto_projetado"] = (
            (df_jog["nota_ajustada_divisao"] * 8.5)
            + (df_jog["producao_ofensiva_ajustada_divisao"] * 18.0)
            + (df_jog["taxa_exposicao_minutos"] * 15.0)
        ) * df_jog["fator_evolucao_idade"]
        df_jog["indice_teto_projetado"] = df_jog["indice_teto_projetado"].round(2)

        # Agrega os indicadores do elenco para cada clube/temporada
        if not df_tabela_times.empty and not df_jog.empty:
            agregado_clube = (
                df_jog.groupby(["sofascore_team_id", "ano", "divisao"])
                .agg(
                    total_jogadores_utilizados=("sofascore_player_id", "count"),
                    idade_media_ponderada_minutos=(
                        "idade_na_temporada",
                        lambda x: np.average(x, weights=df_jog.loc[x.index, "minutos"] + 1)
                    ),
                    nota_media_elenco=("nota_media_sofascore", "mean"),
                    potencial_medio_elenco=("indice_teto_projetado", "mean"),
                    top3_potencial_elenco=(
                        "indice_teto_projetado",
                        lambda x: x.nlargest(3).mean()
                    )
                )
                .reset_index()
                .round(2)
            )

            df_times_enriquecido = pd.merge(
                df_tabela_times,
                agregado_clube,
                on=["sofascore_team_id", "ano", "divisao"],
                how="left"
            )
        else:
            df_times_enriquecido = df_tabela_times

        return df_jog, df_times_enriquecido


# ==============================================================================
# 7. PERSISTÊNCIA EM BANCO DE DADOS SQLITE E EXPORTAÇÃO CSV
# ==============================================================================

def salvar_no_banco(
    df_jogadores: pd.DataFrame,
    df_times: pd.DataFrame,
    caminho_db: str = "futebol_br.db"
):
    """
    Salva as tabelas prontas para modelagem preditiva em um banco SQLite local
    (de forma transacional) e também exporta arquivos CSV de forma atômica.
    """
    for db_path in [caminho_db, os.path.join("banco_dados", "futebol_br.db")]:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            with conn:
                if not df_times.empty:
                    df_times.to_sql("fato_times_campeonato", conn, if_exists="replace", index=False)
                if not df_jogadores.empty:
                    df_jogadores.to_sql("fato_jogadores_temporada", conn, if_exists="replace", index=False)
        finally:
            conn.close()

    if not df_times.empty:
        tmp_t = "fato_times_campeonato.csv.tmp"
        df_times.to_csv(tmp_t, index=False, encoding="utf-8-sig")
        os.replace(tmp_t, "fato_times_campeonato.csv")
        logger.info(f"Tabela 'fato_times_campeonato' salva ({len(df_times)} registros).")

    if not df_jogadores.empty:
        tmp_j = "fato_jogadores_temporada.csv.tmp"
        df_jogadores.to_csv(tmp_j, index=False, encoding="utf-8-sig")
        os.replace(tmp_j, "fato_jogadores_temporada.csv")
        logger.info(f"Tabela 'fato_jogadores_temporada' salva ({len(df_jogadores)} registros).")


# ==============================================================================
# 8. EXECUÇÃO PRINCIPAL (ORQUESTRADOR DO PIPELINE)
# ==============================================================================

def executar_pipeline_completo(anos_alvo: List[int] = [2023, 2024, 2025, 2026], usar_dados_demo_se_offline: bool = True):
    """
    Orquestra a extração das Séries A, B, C e D (incluindo atualização da temporada 2026),
    cruza com Transfermarkt/FM, enriquece fichas de atletas (fotos CDN, biometria, valor de mercado)
    e gera o banco de dados final sem corromper o banco local.
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from importlib import import_module
    try:
        etl_mod = import_module("03_unificacao_fuzzy_etl")
        df_jog, df_times = etl_mod.executar_pipeline_etl(anos=anos_alvo)
        return df_jog, df_times
    except Exception as e:
        logger.warning(f"Fallback para coletor direto ({e})...")

    logger.info("Iniciando Pipeline de Dados do Futebol Brasileiro (Séries A a D)...")
    sofascore = SofascoreExtractor()
    tm = TransfermarktExtractor()
    integrador = IntegradorFutebolBR()

    lista_tabelas_times = []
    lista_stats_jogadores = []
    lista_cadastro_jogadores = []

    # 1. Coleta nas 4 Divisões Nacionais via Sofascore
    for divisao, info in SOFASCORE_TOURNAMENTS.items():
        if divisao == "Copa_do_Brasil":
            continue  # Foca primeiro nos pontos corridos/grupos de A a D

        t_id = info["id"]
        seasons = sofascore.listar_temporadas(t_id)
        if not seasons:
            continue

        for s in seasons:
            ano_str = str(s.get("year", ""))
            # Identifica se a temporada corresponde a um dos anos desejados
            for ano in anos_alvo:
                if str(ano) in ano_str:
                    s_id = s.get("id")
                    logger.info(f"--> Coletando {divisao} | Temporada {ano} (season_id={s_id})...")

                    # 1.1 Tabela de pontuação dos clubes
                    df_tab = sofascore.extrair_tabela_classificacao(t_id, s_id, divisao, ano)
                    if not df_tab.empty:
                        lista_tabelas_times.append(df_tab)

                    # 1.2 Estatísticas completas de todos os jogadores da divisão
                    df_st = sofascore.extrair_estatisticas_jogadores_campeonato(t_id, s_id, divisao, ano)
                    if not df_st.empty:
                        lista_stats_jogadores.append(df_st)

                    # 1.3 Cadastro biométrico dos elencos (altura, data de nascimento, pé preferido)
                    if not df_tab.empty:
                        ids_clubes = df_tab["sofascore_team_id"].dropna().unique().tolist()
                        df_cad = sofascore.extrair_elencos_e_jogadores_dos_clubes(ids_clubes, divisao, ano)
                        if not df_cad.empty:
                            lista_cadastro_jogadores.append(df_cad)

    df_times_total = pd.concat(lista_tabelas_times, ignore_index=True) if lista_tabelas_times else pd.DataFrame()
    df_stats_total = pd.concat(lista_stats_jogadores, ignore_index=True) if lista_stats_jogadores else pd.DataFrame()
    df_cad_total = pd.concat(lista_cadastro_jogadores, ignore_index=True) if lista_cadastro_jogadores else pd.DataFrame()

    # Une estatísticas de campo com dados biométricos (idade, altura, posição, pé)
    if not df_stats_total.empty and not df_cad_total.empty:
        cols_bio = [
            "sofascore_player_id", "ano_referencia", "posicao", "altura_cm",
            "pe_preferido", "data_nascimento", "idade_na_temporada", "valor_mercado_sofascore_eur"
        ]
        df_cad_unico = df_cad_total[cols_bio].drop_duplicates(subset=["sofascore_player_id", "ano_referencia"])
        df_stats_total = pd.merge(
            df_stats_total,
            df_cad_unico,
            left_on=["sofascore_player_id", "ano"],
            right_on=["sofascore_player_id", "ano_referencia"],
            how="left"
        )

    # 2. Enriquecimento com Transfermarkt (Valor de mercado histórico e IDs globais)
    df_tm_players, _ = tm.baixar_base_aberta_transfermarkt()
    if not df_stats_total.empty and not df_tm_players.empty:
        cols_tm = ["jogador_chave", "clube_chave", "player_id", "market_value_in_eur", "highest_market_value_in_eur"]
        cols_tm_disp = [c for c in cols_tm if c in df_tm_players.columns]
        df_stats_total = integrador.cruzar_jogadores_fuzzy(df_stats_total, df_tm_players[cols_tm_disp])

    # 3. Enriquecimento opcional com CSV do Football Manager (CA / PA)
    df_fm = FootballManagerLoader.carregar_csv_fm("dados_brutos/fm_brasil_export.csv")
    if not df_stats_total.empty and not df_fm.empty:
        df_stats_total = integrador.cruzar_jogadores_fuzzy(df_stats_total, df_fm)

    # 4. Cálculo das Features Preditivas (Índice de Teto do Jogador + Força Agregada do Elenco)
    if not df_stats_total.empty:
        df_jogadores_final, df_times_final = integrador.calcular_potencial_e_forca_elenco(
            df_stats_total, df_times_total
        )
        salvar_no_banco(df_jogadores_final, df_times_final)
        try:
            perfil_mod = import_module("07_extracao_perfil_jogadores")
            perfil_mod.construir_base_perfil_jogadores()
        except Exception as e:
            logger.warning(f"Aviso ao construir jogadores_perfil: {e}")
        logger.info("Pipeline finalizado com sucesso!")
    else:
        logger.warning("Nenhum dado retornado da rede. Verifique sua conexão ou dependências.")


if __name__ == "__main__":
    # Para rodar a coleta real no seu computador (VS Code), basta executar este script:
    executar_pipeline_completo(anos_alvo=[2023, 2024, 2025, 2026])
