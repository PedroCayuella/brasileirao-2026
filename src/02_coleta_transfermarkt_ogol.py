"""
================================================================================
MÓDULO 2: COLETA DE DADOS TRANSFERMARKT E COMPLEMENTO O GOL
================================================================================
Responsável por:
1. Baixar as bases abertas players.csv.gz e clubs.csv.gz do repositório
   dcaribou/transfermarkt-datasets (Cloudflare R2).
2. Filtrar atletas brasileiros e clubes atuantes no Brasil.
3. Disponibilizar scraper auxiliar em BeautifulSoup para ogol.com.br
   para complemento de elencos (especialmente Série D).
"""

import os
import io
import logging
import unicodedata
from typing import Dict, Optional, Tuple, Any

import pandas as pd
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as cf_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as cf_requests
    HAS_CURL_CFFI = False

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TM_OgolCollector")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
}


def normalizar_texto(texto: Any) -> str:
    """Normaliza texto para permitir Entity Resolution entre bases."""
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.lower().strip()
    for token in [" fc", " ec", " ac", " sc", " saf", " clube", " associacao", " atletico"]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())


class TransfermarktExtractor:
    """
    Consome os dumps abertos de players e clubs do repositório transfermarkt-datasets.
    """
    RAW_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
    DIR_DESTINO = os.path.join("dados_brutos", "transfermarkt")

    def __init__(self, pasta_destino: Optional[str] = None):
        self.pasta_destino = pasta_destino or self.DIR_DESTINO
        os.makedirs(self.pasta_destino, exist_ok=True)

    def _baixar_arquivo(self, nome_arquivo: str) -> Optional[bytes]:
        """Faz o download de um arquivo do R2 usando curl_cffi para contornar 403."""
        caminho_local = os.path.join(self.pasta_destino, nome_arquivo)
        if os.path.exists(caminho_local):
            logger.info(f"[Transfermarkt] Utilizando arquivo em cache: {caminho_local}")
            with open(caminho_local, "rb") as f:
                return f.read()

        url = f"{self.RAW_BASE_URL}/{nome_arquivo}"
        logger.info(f"[Transfermarkt] Baixando {url}...")
        try:
            kwargs: Dict[str, Any] = {"headers": DEFAULT_HEADERS, "timeout": 45}
            if HAS_CURL_CFFI:
                kwargs["impersonate"] = "chrome"

            resp = cf_requests.get(url, **kwargs)
            if resp.status_code == 200 and len(resp.content) > 0:
                with open(caminho_local, "wb") as f:
                    f.write(resp.content)
                logger.info(f"[Transfermarkt] Salvo {nome_arquivo} ({len(resp.content):,} bytes).")
                return resp.content
            else:
                logger.error(f"[Transfermarkt] Status {resp.status_code} ao baixar {nome_arquivo}")
                return None
        except Exception as e:
            logger.error(f"[Transfermarkt] Erro ao baixar {nome_arquivo}: {e}")
            return None

    def baixar_base_aberta_transfermarkt(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Baixa players.csv.gz e clubs.csv.gz e filtra atletas brasileiros
        e clubes que disputam competições brasileiras.
        """
        conteudo_players = self._baixar_arquivo("players.csv.gz")
        conteudo_clubs = self._baixar_arquivo("clubs.csv.gz")

        df_players = pd.DataFrame()
        df_clubs = pd.DataFrame()

        if conteudo_clubs:
            try:
                df_c = pd.read_csv(io.BytesIO(conteudo_clubs), compression="gzip")
                cond_c = df_c["domestic_competition_id"].astype(str).str.contains("BRA", na=False)
                df_clubs = df_c[cond_c].copy()
                df_clubs["clube_chave"] = df_clubs["name"].astype(str).apply(normalizar_texto)
                logger.info(f"[Transfermarkt] {len(df_clubs)} clubes brasileiros identificados.")
            except Exception as e:
                logger.error(f"[Transfermarkt] Erro ao processar clubs.csv.gz: {e}")

        if conteudo_players:
            try:
                df_p = pd.read_csv(io.BytesIO(conteudo_players), compression="gzip")
                # Filtra atletas brasileiros ou que atuam em ligas com 'BRA'
                cond_p = (
                    (df_p["country_of_citizenship"] == "Brazil")
                    | df_p["current_club_domestic_competition_id"].astype(str).str.contains("BRA", na=False)
                )
                df_players = df_p[cond_p].copy()
                df_players["jogador_chave"] = df_players["name"].astype(str).apply(normalizar_texto)
                df_players["clube_chave"] = df_players["current_club_name"].astype(str).apply(normalizar_texto)

                # Extrai ano de nascimento se disponível
                if "date_of_birth" in df_players.columns:
                    df_players["ano_nascimento_tm"] = pd.to_datetime(
                        df_players["date_of_birth"], errors="coerce"
                    ).dt.year

                logger.info(
                    f"[Transfermarkt] {len(df_players)} jogadores brasileiros ou atuando no Brasil filtrados com sucesso."
                )
            except Exception as e:
                logger.error(f"[Transfermarkt] Erro ao processar players.csv.gz: {e}")

        return df_players, df_clubs


class OGolScraper:
    """
    Scraper auxiliar para o portal ogol.com.br para complementar elencos
    de clubes menores (ex: Série D e campeonatos regionais).
    """
    BASE_URL = "https://www.ogol.com.br"

    def extrair_elenco_clube_ogol(self, team_slug_id: int, ano_epoca: int = 2025) -> pd.DataFrame:
        """
        Extrai o elenco de um clube específico no O Gol via BeautifulSoup.
        """
        url = f"{self.BASE_URL}/equipa.php?id={team_slug_id}"
        logger.info(f"[O Gol] Extraindo elenco de team_id={team_slug_id}...")
        try:
            kwargs: Dict[str, Any] = {"headers": DEFAULT_HEADERS, "timeout": 20}
            if HAS_CURL_CFFI:
                kwargs["impersonate"] = "chrome"

            resp = cf_requests.get(url, **kwargs)
            if resp.status_code != 200:
                logger.warning(f"[O Gol] Status {resp.status_code} para {url}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"[O Gol] Erro ao acessar {url}: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, "lxml")
        jogadores = []

        # Procura por linhas de tabela de elenco
        linhas = soup.select("div#team_squad table tr, div.staff line, table.zztable tr")
        for linha in linhas:
            cols = linha.find_all("td")
            if len(cols) >= 3:
                nome_tag = linha.select_one("a[href*='jogador.php']")
                if not nome_tag:
                    continue
                nome = nome_tag.get_text(strip=True)
                href = nome_tag.get("href", "")
                ogol_id = None
                if "id=" in href:
                    ogol_id = href.split("id=")[-1].split("&")[0]

                # Tenta localizar idade
                idade_txt = cols[-1].get_text(strip=True)
                idade = int(idade_txt) if idade_txt.isdigit() else None

                pos_tag = linha.select_one(".pos, td.pos")
                posicao = pos_tag.get_text(strip=True) if pos_tag else None

                jogadores.append({
                    "ogol_player_id": ogol_id,
                    "jogador_nome": nome,
                    "jogador_chave": normalizar_texto(nome),
                    "ogol_team_id": team_slug_id,
                    "ano": ano_epoca,
                    "posicao_ogol": posicao,
                    "idade_ogol": idade,
                })

        df = pd.DataFrame(jogadores)
        logger.info(f"[O Gol] {len(df)} jogadores extraídos para equipe {team_slug_id}.")
        return df


def executar_coleta_transfermarkt() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Função de entrada para o Módulo 2."""
    tm = TransfermarktExtractor()
    return tm.baixar_base_aberta_transfermarkt()


if __name__ == "__main__":
    df_p, df_c = executar_coleta_transfermarkt()
    print(f"Transfermarkt finalizado: {len(df_p)} jogadores, {len(df_c)} clubes.")

