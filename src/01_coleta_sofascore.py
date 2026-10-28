"""
================================================================================
MÓDULO 1: COLETA DE DADOS DO SOFASCORE (SÉRIES A, B, C E D)
================================================================================
Responsável por consumir a API do Sofascore com bypass TLS via curl_cffi,
extrair tabelas de classificação, estatísticas de jogadores e dados biométricos
de elencos, com mecanismo resiliente de cache em disco e biometria global.
"""

import os
import glob
import time
import json
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

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
logger = logging.getLogger("SofascoreCollector")

# IDs oficiais dos Torneios Brasileiros no Sofascore
SOFASCORE_TOURNAMENTS = {
    "Serie_A": {"id": 325, "nivel": 1, "peso_divisao": 1.00},
    "Serie_B": {"id": 390, "nivel": 2, "peso_divisao": 0.72},
    "Serie_C": {"id": 1281, "nivel": 3, "peso_divisao": 0.54},
    "Serie_D": {"id": 10326, "legacy_id": 1598, "nivel": 4, "peso_divisao": 0.40},
}


def normalizar_texto(texto: Any) -> str:
    """Normaliza texto removendo acentos, minúsculas e limpando espaços."""
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.lower().strip()
    for token in [" fc", " ec", " ac", " sc", " saf", " clube", " associacao", " atletico"]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())


def requisicao_segura(url: str, pausa: float = 1.0, max_tentativas: int = 3) -> Optional[Any]:
    """
    Executa requisição HTTP usando curl_cffi para emular fingerprint TLS do Chrome.
    Implementa retentativas resilientes com pausa configurada.
    """
    time.sleep(pausa)
    tentativas = 0
    impersonates = ["chrome", "chrome120"] if HAS_CURL_CFFI else [None]

    while tentativas < max_tentativas:
        for imp in impersonates:
            try:
                kwargs: Dict[str, Any] = {"timeout": 20}
                if HAS_CURL_CFFI and imp:
                    kwargs["impersonate"] = imp

                resp = cf_requests.get(url, **kwargs)
                if resp.status_code == 200:
                    return resp
                elif resp.status_code == 404:
                    logger.warning(f"[404 Não Encontrado]: {url}")
                    return None
                elif resp.status_code == 403:
                    continue
                else:
                    logger.warning(f"Status {resp.status_code} na requisição: {url}")
            except Exception as e:
                logger.error(f"Erro de conexão ({url}): {e}")
        
        tentativas += 1
        time.sleep(pausa)

    logger.error(f"Falha após {max_tentativas} tentativas para {url}")
    return None


class CacheBiometriaGlobal:
    """
    Mantém um dicionário global em disco de todos os atletas conhecidos:
    data de nascimento, altura, pé preferido, posição e nacionalidade.
    Evita requisições repetidas para o mesmo atleta em diferentes anos.
    """
    def __init__(self, caminho_cache: str = "dados_brutos/cache_biometria_global.json"):
        self.caminho = caminho_cache
        self.dados: Dict[int, Dict[str, Any]] = {}
        self._carregar_ou_inicializar()

    def _carregar_ou_inicializar(self):
        if os.path.exists(self.caminho):
            try:
                with open(self.caminho, "r", encoding="utf-8") as f:
                    carregado = json.load(f)
                    self.dados = {int(k): v for k, v in carregado.items()}
                logger.info(f"[Biometria Global] {len(self.dados)} atletas carregados do cache global.")
                return
            except Exception as e:
                logger.warning(f"Falha ao carregar {self.caminho}: {e}")

        # Popula a partir dos CSVs de elencos existentes
        arquivos_existentes = glob.glob("dados_brutos/sofascore/*_elencos.csv")
        for arq in arquivos_existentes:
            try:
                df = pd.read_csv(arq)
                for _, row in df.iterrows():
                    pid = row.get("sofascore_player_id")
                    if pd.notna(pid) and int(pid) not in self.dados:
                        ano_nasc = None
                        dn = row.get("data_nascimento")
                        if pd.notna(dn) and isinstance(dn, str) and "-" in dn:
                            try:
                                ano_nasc = int(dn.split("-")[0])
                            except Exception:
                                pass

                        self.dados[int(pid)] = {
                            "jogador_nome": row.get("jogador_nome"),
                            "jogador_chave": row.get("jogador_chave"),
                            "data_nascimento": dn if pd.notna(dn) else None,
                            "ano_nascimento": ano_nasc,
                            "altura_cm": row.get("altura_cm") if pd.notna(row.get("altura_cm")) else None,
                            "pe_preferido": row.get("pe_preferido") if pd.notna(row.get("pe_preferido")) else None,
                            "posicao": row.get("posicao") if pd.notna(row.get("posicao")) else None,
                            "nacionalidade": row.get("nacionalidade") if pd.notna(row.get("nacionalidade")) else None,
                            "valor_mercado_sofascore_eur": row.get("valor_mercado_sofascore_eur") if pd.notna(row.get("valor_mercado_sofascore_eur")) else None,
                        }
            except Exception as e:
                logger.warning(f"Erro ao processar {arq} para biometria global: {e}")

        self.salvar()
        logger.info(f"[Biometria Global] Inicializado com {len(self.dados)} atletas únicos.")

    def salvar(self):
        os.makedirs(os.path.dirname(self.caminho) or ".", exist_ok=True)
        try:
            with open(self.caminho, "w", encoding="utf-8") as f:
                json.dump(self.dados, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar cache de biometria global: {e}")

    def contem(self, player_id: int) -> bool:
        return int(player_id) in self.dados

    def obter(self, player_id: int) -> Optional[Dict[str, Any]]:
        return self.dados.get(int(player_id))

    def adicionar(self, player_id: int, info: Dict[str, Any]):
        self.dados[int(player_id)] = info


class SofascoreExtractor:
    """
    Módulo especializado em extração das Séries A, B, C e D do Sofascore
    com cache automático e reaproveitamento biométrico inteligente.
    """
    BASE_API = "https://www.sofascore.com/api/v1"
    DIR_CACHE = os.path.join("dados_brutos", "sofascore")

    def __init__(self, dir_cache: Optional[str] = None):
        self.dir_cache = dir_cache or self.DIR_CACHE
        os.makedirs(self.dir_cache, exist_ok=True)
        self.cache_bio = CacheBiometriaGlobal()

    def listar_temporadas(self, tournament_id: int) -> List[Dict]:
        """Busca todas as temporadas registradas no Sofascore para um torneio."""
        url = f"{self.BASE_API}/unique-tournament/{tournament_id}/seasons"
        resp = requisicao_segura(url)
        if not resp:
            return []

        try:
            return resp.json().get("seasons", [])
        except Exception as e:
            logger.error(f"Erro ao decodificar temporadas de {tournament_id}: {e}")
            return []

    def obter_season_id(self, tournament_id: int, ano: int, legacy_id: Optional[int] = None) -> Optional[int]:
        """Obtém o season_id correspondente ao ano, testando legacy se necessário."""
        seasons = self.listar_temporadas(tournament_id)
        if not seasons and legacy_id:
            seasons = self.listar_temporadas(legacy_id)

        for s in seasons:
            ano_str = str(s.get("year", ""))
            if str(ano) in ano_str:
                return s.get("id")

        if legacy_id and legacy_id != tournament_id:
            seasons_leg = self.listar_temporadas(legacy_id)
            for s in seasons_leg:
                ano_str = str(s.get("year", ""))
                if str(ano) in ano_str:
                    return s.get("id")

        return None

    def extrair_tabela_classificacao(
        self,
        tournament_id: int,
        season_id: int,
        divisao: str,
        ano: int,
        forcar_atualizacao: bool = False
    ) -> pd.DataFrame:
        """
        Extrai a tabela de classificação completa. Suporta pontos corridos (A e B)
        e múltiplos grupos (C e D), persistindo cache em CSV e JSON de forma atômica.
        """
        cache_csv = os.path.join(self.dir_cache, f"{divisao}_{ano}_tabela.csv")
        cache_json = os.path.join(self.dir_cache, f"{divisao}_{ano}_tabela.json")

        if not forcar_atualizacao and os.path.exists(cache_csv):
            try:
                df_cache = pd.read_csv(cache_csv)
                if not df_cache.empty:
                    logger.info(f"[Cache] Tabela de classificação carregada de {cache_csv} ({len(df_cache)} clubes).")
                    return df_cache
            except Exception as e:
                logger.warning(f"Falha ao ler cache {cache_csv}: {e}")

        url = f"{self.BASE_API}/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
        resp = requisicao_segura(url)
        if not resp:
            if os.path.exists(cache_csv):
                logger.info(f"[Fallback Seguro] Mantendo cache local existente {cache_csv}.")
                return pd.read_csv(cache_csv)
            return pd.DataFrame()

        try:
            dados = resp.json()
        except Exception as e:
            logger.error(f"Erro ao parsear JSON de classificação: {e}")
            if os.path.exists(cache_csv):
                return pd.read_csv(cache_csv)
            return pd.DataFrame()

        standings_groups = dados.get("standings", [])
        registros = []

        for grupo in standings_groups:
            nome_grupo = grupo.get("name", "Geral")
            for row in grupo.get("rows", []):
                team = row.get("team", {})
                jogos = row.get("matches", 0) or 0
                pontos = row.get("points", 0) or 0
                vitorias = row.get("wins", 0) or 0
                empates = row.get("draws", 0) or 0
                derrotas = row.get("losses", 0) or 0
                gols_pro = row.get("scoresFor", 0) or 0
                gols_contra = row.get("scoresAgainst", 0) or 0

                registros.append({
                    "sofascore_team_id": team.get("id"),
                    "clube_nome": team.get("name"),
                    "clube_chave": normalizar_texto(team.get("name", "")),
                    "ano": ano,
                    "divisao": divisao,
                    "grupo": nome_grupo,
                    "posicao": row.get("position"),
                    "jogos": jogos,
                    "vitorias": vitorias,
                    "empates": empates,
                    "derrotas": derrotas,
                    "gols_pro": gols_pro,
                    "gols_contra": gols_contra,
                    "saldo_gols": gols_pro - gols_contra,
                    "pontos": pontos,
                    "aproveitamento_pct": round((pontos / (jogos * 3)) * 100, 2) if jogos > 0 else 0.0
                })

        df = pd.DataFrame(registros)
        if not df.empty:
            tmp_csv = cache_csv + ".tmp"
            df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
            os.replace(tmp_csv, cache_csv)
            with open(cache_json, "w", encoding="utf-8") as f:
                json.dump(dados, f, ensure_ascii=False, indent=2)
            logger.info(f"[Sofascore] {len(df)} clubes extraídos e atualizados em {cache_csv}.")
            return df
        elif os.path.exists(cache_csv):
            return pd.read_csv(cache_csv)
        return df

    def extrair_estatisticas_jogadores_campeonato(
        self,
        tournament_id: int,
        season_id: int,
        divisao: str,
        ano: int,
        forcar_atualizacao: bool = False
    ) -> pd.DataFrame:
        """
        Extrai estatísticas acumuladas de todos os atletas paginando até o fim.
        Calcula gols_por_90, assistencias_por_90, producao_ofensiva_ajustada_divisao.
        """
        cache_csv = os.path.join(self.dir_cache, f"{divisao}_{ano}_stats.csv")
        if not forcar_atualizacao and os.path.exists(cache_csv):
            try:
                df_cache = pd.read_csv(cache_csv)
                if not df_cache.empty:
                    logger.info(f"[Cache] Estatísticas de jogadores carregadas de {cache_csv} ({len(df_cache)} registros).")
                    return df_cache
            except Exception as e:
                logger.warning(f"Falha ao ler cache {cache_csv}: {e}")

        todos_registros = []
        offset = 0
        limit = 100
        campos = (
            "goals,assists,minutesPlayed,appearances,rating,accuratePassesPercentage,"
            "successfulDribbles,tackles,interceptions,aerialDuelsWon,totalShots"
        )
        peso_div = SOFASCORE_TOURNAMENTS.get(divisao, {}).get("peso_divisao", 0.5)

        while True:
            url = (
                f"{self.BASE_API}/unique-tournament/{tournament_id}/season/{season_id}/statistics"
                f"?limit={limit}&offset={offset}&order=-minutesPlayed&accumulation=total&fields={campos}"
            )
            resp = requisicao_segura(url, pausa=1.0)
            if not resp:
                break

            try:
                dados_json = resp.json()
            except Exception as e:
                logger.error(f"Erro ao parsear JSON estatísticas offset={offset}: {e}")
                break

            resultados = dados_json.get("results", [])
            if not resultados:
                break

            for row in resultados:
                player = row.get("player", {})
                team = row.get("team", {})
                minutos = row.get("minutesPlayed", 0) or 0
                gols = row.get("goals", 0) or 0
                assists = row.get("assists", 0) or 0
                rating = row.get("rating", 0.0) or 0.0

                fator_90 = (minutos / 90.0) if minutos >= 90 else None
                gols_p90 = round(gols / fator_90, 3) if fator_90 else 0.0
                assists_p90 = round(assists / fator_90, 3) if fator_90 else 0.0
                part_p90 = round(gols_p90 + assists_p90, 3)

                todos_registros.append({
                    "sofascore_player_id": player.get("id"),
                    "jogador_nome": player.get("name"),
                    "jogador_chave": normalizar_texto(player.get("name", "")),
                    "sofascore_team_id": team.get("id"),
                    "clube_nome": team.get("name"),
                    "clube_chave": normalizar_texto(team.get("name", "")),
                    "ano": ano,
                    "divisao": divisao,
                    "peso_divisao": peso_div,
                    "jogos": row.get("appearances", 0) or 0,
                    "minutos": minutos,
                    "gols": gols,
                    "assistencias": assists,
                    "gols_por_90": gols_p90,
                    "assistencias_por_90": assists_p90,
                    "participacoes_gol_por_90": part_p90,
                    "finalizacoes": row.get("totalShots", 0) or 0,
                    "dribles_certos": row.get("successfulDribbles", 0) or 0,
                    "desarmes": row.get("tackles", 0) or 0,
                    "interceptacoes": row.get("interceptions", 0) or 0,
                    "duelos_aereos_ganhos": row.get("aerialDuelsWon", 0) or 0,
                    "passes_certos_pct": round(row.get("accuratePassesPercentage", 0.0) or 0.0, 2),
                    "nota_media_sofascore": round(rating, 2)
                })

            if len(resultados) < limit:
                break
            offset += limit

        df = pd.DataFrame(todos_registros)
        if not df.empty:
            df["producao_ofensiva_ajustada_divisao"] = (
                df["participacoes_gol_por_90"] * df["peso_divisao"]
            ).round(3)
            df["nota_ajustada_divisao"] = (
                df["nota_media_sofascore"] * (0.7 + 0.3 * df["peso_divisao"])
            ).round(3)
            tmp_csv = cache_csv + ".tmp"
            df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
            os.replace(tmp_csv, cache_csv)
            logger.info(f"[Sofascore] {len(df)} jogadores extraídos e atualizados em {cache_csv}.")
            return df
        elif os.path.exists(cache_csv):
            logger.info(f"[Fallback Seguro] Mantendo cache local de estatísticas {cache_csv}.")
            return pd.read_csv(cache_csv)

        return df

    def extrair_elencos_e_jogadores_dos_clubes(
        self,
        team_ids: List[int],
        divisao: str,
        ano: int,
        jogadores_conhecidos_stats: Optional[List[Dict]] = None,
        sincronizar_novos_atletas: bool = False
    ) -> pd.DataFrame:
        """
        Extrai a ficha biométrica dos jogadores dos clubes.
        Otimização: reutiliza imediatamente atletas presentes no CacheBiometriaGlobal,
        apenas recalculando a idade_na_temporada = ano - ano_nascimento.
        Só consulta a rede para clubes ou jogadores novos.
        """
        cache_csv = os.path.join(self.dir_cache, f"{divisao}_{ano}_elencos.csv")
        if os.path.exists(cache_csv) and not sincronizar_novos_atletas:
            try:
                df_cache = pd.read_csv(cache_csv)
                if not df_cache.empty:
                    logger.info(f"[Cache] Elencos carregados de {cache_csv} ({len(df_cache)} atletas).")
                    return df_cache
            except Exception as e:
                logger.warning(f"Falha ao ler cache {cache_csv}: {e}")

        if os.path.exists(cache_csv) and sincronizar_novos_atletas:
            try:
                df_cache = pd.read_csv(cache_csv)
                pids_existentes = set(df_cache["sofascore_player_id"].dropna().astype(int).tolist())
                novos_registros = []
                if jogadores_conhecidos_stats:
                    for item in jogadores_conhecidos_stats:
                        pid = item.get("sofascore_player_id")
                        tid = item.get("sofascore_team_id")
                        if pid and int(pid) not in pids_existentes and self.cache_bio.contem(pid):
                            bio = self.cache_bio.obter(pid)
                            ano_nasc = bio.get("ano_nascimento")
                            idade = (ano - ano_nasc) if ano_nasc else None
                            novos_registros.append({
                                "sofascore_player_id": pid,
                                "jogador_nome": bio.get("jogador_nome") or item.get("jogador_nome"),
                                "jogador_chave": bio.get("jogador_chave") or item.get("jogador_chave"),
                                "sofascore_team_id": tid,
                                "divisao_atual": divisao,
                                "ano_referencia": ano,
                                "posicao": bio.get("posicao"),
                                "altura_cm": bio.get("altura_cm"),
                                "pe_preferido": bio.get("pe_preferido"),
                                "data_nascimento": bio.get("data_nascimento"),
                                "idade_na_temporada": idade,
                                "nacionalidade": bio.get("nacionalidade"),
                                "valor_mercado_sofascore_eur": bio.get("valor_mercado_sofascore_eur")
                            })
                            pids_existentes.add(int(pid))
                if novos_registros:
                    df_cache = pd.concat([df_cache, pd.DataFrame(novos_registros)], ignore_index=True)
                    tmp_csv = cache_csv + ".tmp"
                    df_cache.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
                    os.replace(tmp_csv, cache_csv)
                    logger.info(f"[Elencos Sync] +{len(novos_registros)} atletas sincronizados em {cache_csv}.")
                return df_cache
            except Exception as e:
                logger.warning(f"Aviso ao sincronizar elencos {cache_csv}: {e}")

        jogadores_saida = []
        clubes_unicos = [int(tid) for tid in team_ids if pd.notna(tid)]

        # 1. Se já conhecemos os atletas das estatísticas, preenche direto pelo Cache Global
        if jogadores_conhecidos_stats:
            for item in jogadores_conhecidos_stats:
                pid = item.get("sofascore_player_id")
                tid = item.get("sofascore_team_id")
                if pid and self.cache_bio.contem(pid):
                    bio = self.cache_bio.obter(pid)
                    ano_nasc = bio.get("ano_nascimento")
                    idade = (ano - ano_nasc) if ano_nasc else None

                    jogadores_saida.append({
                        "sofascore_player_id": pid,
                        "jogador_nome": bio.get("jogador_nome") or item.get("jogador_nome"),
                        "jogador_chave": bio.get("jogador_chave") or item.get("jogador_chave"),
                        "sofascore_team_id": tid,
                        "divisao_atual": divisao,
                        "ano_referencia": ano,
                        "posicao": bio.get("posicao"),
                        "altura_cm": bio.get("altura_cm"),
                        "pe_preferido": bio.get("pe_preferido"),
                        "data_nascimento": bio.get("data_nascimento"),
                        "idade_na_temporada": idade,
                        "nacionalidade": bio.get("nacionalidade"),
                        "valor_mercado_sofascore_eur": bio.get("valor_mercado_sofascore_eur")
                    })

        # Identifica quantos atletas foram recuperados sem rede
        pids_recuperados = set(j["sofascore_player_id"] for j in jogadores_saida)

        # 2. Busca na rede para clubes onde ainda faltam atletas
        novos_cadastrados = 0
        total_clubes = len(clubes_unicos)

        for i, team_id in enumerate(clubes_unicos, start=1):
            url = f"{self.BASE_API}/team/{team_id}/players"
            resp = requisicao_segura(url, pausa=0.5)
            if not resp:
                continue

            try:
                players_list = resp.json().get("players", [])
            except Exception:
                continue

            for item in players_list:
                p = item.get("player", {})
                if not p:
                    continue

                pid = p.get("id")
                if pid in pids_recuperados:
                    continue

                ts_nasc = p.get("dateOfBirthTimestamp")
                data_nasc = None
                ano_nasc = None
                idade_ano = None

                if ts_nasc is not None:
                    try:
                        dt = datetime.fromtimestamp(ts_nasc, tz=timezone.utc)
                        data_nasc = dt.strftime("%Y-%m-%d")
                        ano_nasc = dt.year
                        idade_ano = ano - ano_nasc
                    except Exception:
                        pass

                info_bio = {
                    "jogador_nome": p.get("name"),
                    "jogador_chave": normalizar_texto(p.get("name", "")),
                    "data_nascimento": data_nasc,
                    "ano_nascimento": ano_nasc,
                    "altura_cm": p.get("height"),
                    "pe_preferido": p.get("preferredFoot"),
                    "posicao": p.get("position"),
                    "nacionalidade": p.get("country", {}).get("name"),
                    "valor_mercado_sofascore_eur": p.get("proposedMarketValue"),
                }
                self.cache_bio.adicionar(pid, info_bio)
                novos_cadastrados += 1
                pids_recuperados.add(pid)

                jogadores_saida.append({
                    "sofascore_player_id": pid,
                    "jogador_nome": p.get("name"),
                    "jogador_chave": normalizar_texto(p.get("name", "")),
                    "sofascore_team_id": team_id,
                    "divisao_atual": divisao,
                    "ano_referencia": ano,
                    "posicao": p.get("position"),
                    "altura_cm": p.get("height"),
                    "pe_preferido": p.get("preferredFoot"),
                    "data_nascimento": data_nasc,
                    "idade_na_temporada": idade_ano,
                    "nacionalidade": p.get("country", {}).get("name"),
                    "valor_mercado_sofascore_eur": p.get("proposedMarketValue")
                })

            if i % 10 == 0 or i == total_clubes:
                logger.info(f"[Sofascore Elencos] Processados {i}/{total_clubes} clubes de {divisao} ({ano}).")

        if novos_cadastrados > 0:
            self.cache_bio.salvar()
            logger.info(f"[Biometria Global] {novos_cadastrados} novos atletas adicionados ao cache permanente.")

        df = pd.DataFrame(jogadores_saida)
        if not df.empty:
            tmp_csv = cache_csv + ".tmp"
            df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
            os.replace(tmp_csv, cache_csv)
            logger.info(f"[Sofascore] {len(df)} jogadores de elenco consolidados em {cache_csv}.")
        return df


SEASON_IDS_2026 = {
    325: 87678,
    390: 89840,
    1281: 90642,
    10326: 92447,
}


def executar_coleta_sofascore(
    anos: List[int] = [2023, 2024, 2025, 2026],
    atualizar_temporada_2026: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Função executora incremental do Módulo 1.
    Coleta temporadas ausentes e, quando atualizar_temporada_2026=True,
    atualiza tabelas e métricas recentes da temporada ativa de 2026 sem corromper o cache local.
    """
    logger.info(f"=== INICIANDO COLETA INCREMENTAL SOFASCORE (ANOS: {anos}) ===")
    extractor = SofascoreExtractor()

    todas_tabelas = []
    todos_stats = []
    todos_elencos = []

    for divisao, info in SOFASCORE_TOURNAMENTS.items():
        t_id = info["id"]
        legacy_id = info.get("legacy_id")

        for ano in anos:
            cache_tabela = os.path.join(extractor.dir_cache, f"{divisao}_{ano}_tabela.csv")
            cache_stats = os.path.join(extractor.dir_cache, f"{divisao}_{ano}_stats.csv")
            cache_elencos = os.path.join(extractor.dir_cache, f"{divisao}_{ano}_elencos.csv")

            ja_coletado = (
                os.path.exists(cache_tabela) and os.path.getsize(cache_tabela) > 100
                and os.path.exists(cache_stats) and os.path.getsize(cache_stats) > 100
                and os.path.exists(cache_elencos) and os.path.getsize(cache_elencos) > 100
            )

            deve_atualizar_2026 = (ano == 2026 and atualizar_temporada_2026)

            if ja_coletado and not deve_atualizar_2026:
                logger.info(f"[Cache Inteligente] {divisao} ({ano}) já coletado. Carregando do disco...")
                df_tab = pd.read_csv(cache_tabela)
                df_st = pd.read_csv(cache_stats)
                df_el = pd.read_csv(cache_elencos)
                todas_tabelas.append(df_tab)
                todos_stats.append(df_st)
                todos_elencos.append(df_el)
                continue

            logger.info(f"\n--- Coletando / Atualizando {divisao} ({ano}) ---")
            s_id = SEASON_IDS_2026.get(t_id) if ano == 2026 else None
            if not s_id:
                s_id = extractor.obter_season_id(t_id, ano, legacy_id=legacy_id)
            if not s_id:
                logger.warning(f"Temporada {ano} não encontrada para {divisao}.")
                if ja_coletado:
                    todas_tabelas.append(pd.read_csv(cache_tabela))
                    todos_stats.append(pd.read_csv(cache_stats))
                    todos_elencos.append(pd.read_csv(cache_elencos))
                continue

            # a) Tabela de Classificação
            df_tab = extractor.extrair_tabela_classificacao(
                t_id, s_id, divisao, ano, forcar_atualizacao=deve_atualizar_2026
            )
            if not df_tab.empty:
                todas_tabelas.append(df_tab)

            # b) Estatísticas de Jogadores
            df_st = extractor.extrair_estatisticas_jogadores_campeonato(
                t_id, s_id, divisao, ano, forcar_atualizacao=deve_atualizar_2026
            )
            if not df_st.empty:
                todos_stats.append(df_st)

            # c) Elencos / Biometria (Reaproveita cache global e sincroniza novos atletas de 2026)
            if not df_tab.empty:
                team_ids = df_tab["sofascore_team_id"].dropna().unique().tolist()
                stats_list = df_st.to_dict(orient="records") if not df_st.empty else None
                df_el = extractor.extrair_elencos_e_jogadores_dos_clubes(
                    team_ids,
                    divisao,
                    ano,
                    jogadores_conhecidos_stats=stats_list,
                    sincronizar_novos_atletas=deve_atualizar_2026
                )
                if not df_el.empty:
                    todos_elencos.append(df_el)

    df_times_final = pd.concat(todas_tabelas, ignore_index=True) if todas_tabelas else pd.DataFrame()
    df_stats_final = pd.concat(todos_stats, ignore_index=True) if todos_stats else pd.DataFrame()
    df_elencos_final = pd.concat(todos_elencos, ignore_index=True) if todos_elencos else pd.DataFrame()

    logger.info(
        f"Coleta Sofascore concluída: {len(df_times_final)} clubes, "
        f"{len(df_stats_final)} registros de estatísticas, {len(df_elencos_final)} registros de elenco."
    )
    return df_times_final, df_stats_final, df_elencos_final


if __name__ == "__main__":
    executar_coleta_sofascore(anos=[2023, 2024, 2025, 2026])
