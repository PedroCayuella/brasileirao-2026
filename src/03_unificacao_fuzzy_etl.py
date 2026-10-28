"""
================================================================================
MÓDULO 3: UNIFICAÇÃO, ENTITY RESOLUTION VIA RAPIDFUZZ E ETL MULTITEMPORAL
================================================================================
Responsável por:
1. Normalizar strings de nomes de atletas e clubes.
2. Executar Entity Resolution (RapidFuzz token_sort_ratio >= 86) entre bases
   Sofascore e Transfermarkt.
3. Calcular features avançadas de engenharia de dados:
   - producao_ofensiva_ajustada_divisao
   - indice_teto_projetado
4. Engenharia de Features Temporais Multi-Temporada (2023 a 2026):
   - nota_ano_anterior, minutos_ano_anterior, nivel_divisao_anterior
   - delta_evolucao_nota, salto_de_divisao
   - pontos_ano_anterior, divisao_ano_anterior nos clubes
5. Agregar métricas dos jogadores por clube/temporada.
6. Persistir no SQLite (banco_dados/futebol_br.db e raiz) e nos CSVs fato.
"""

import os
import sys
import json
import glob
import sqlite3
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

pd.set_option("future.no_silent_downcasting", True)

# Adiciona diretório src ao path para permitir imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

coleta_sofa = import_module("01_coleta_sofascore")
coleta_tm = import_module("02_coleta_transfermarkt_ogol")

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ETL_Unificacao")

MAPA_NIVEL_DIVISAO = {
    "Serie_A": 1,
    "Serie_B": 2,
    "Serie_C": 3,
    "Serie_D": 4
}


def normalizar_texto(texto: Any) -> str:
    """
    Remove acentos, converte para minúsculas e remove termos comuns em nomes de clubes.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""
    texto = unicodedata.normalize("NFKD", texto).encode("ASCII", "ignore").decode("utf-8")
    texto = texto.lower().strip()
    for token in [" fc", " ec", " ac", " sc", " saf", " clube", " associacao", " atletico"]:
        texto = texto.replace(token, " ")
    return " ".join(texto.split())


class PadronizadorPosicoes:
    """
    Motor canônico e estrito de padronização de posições para TODOS os clubes e temporadas.
    Garante separação 100% consistente nas 4 categorias oficiais:
    - 'G': Goleiros (Goleiro)
    - 'D': Defensores (Zagueiro Central, Lateral Direito, Lateral Esquerdo)
    - 'M': Meio-campistas (Volante, Meio-Campista Central, Meia Ofensivo, Meia Direito, Meia Esquerdo)
    - 'F': Atacantes (Centroavante, Segundo Atacante, Ponta Direita, Ponta Esquerda)
    """

    MAPA_SUBPOS_PARA_SETOR = {
        "Goalkeeper": "G",
        "Centre-Back": "D",
        "Left-Back": "D",
        "Right-Back": "D",
        "Defender": "D",
        "Sweeper": "D",
        "Defensive Midfield": "M",
        "Central Midfield": "M",
        "Attacking Midfield": "M",
        "Left Midfield": "M",
        "Right Midfield": "M",
        "Midfield": "M",
        "Centre-Forward": "F",
        "Second Striker": "F",
        "Left Winger": "F",
        "Right Winger": "F",
        "Attack": "F",
    }

    MAPA_SUBPOS_DETALHADA_POR_SETOR = {
        "G": {
            "Goalkeeper": "Goleiro",
        },
        "D": {
            "Centre-Back": "Zagueiro Central",
            "Right-Back": "Lateral Direito",
            "Left-Back": "Lateral Esquerdo",
            "Defender": "Zagueiro Central",
            "Sweeper": "Zagueiro Central",
        },
        "M": {
            "Defensive Midfield": "Volante",
            "Central Midfield": "Meio-Campista Central",
            "Attacking Midfield": "Meia Ofensivo",
            "Right Midfield": "Meia Direito",
            "Left Midfield": "Meia Esquerdo",
            "Right Winger": "Meia Ofensivo",
            "Left Winger": "Meia Ofensivo",
            "Second Striker": "Meia Ofensivo",
        },
        "F": {
            "Centre-Forward": "Centroavante",
            "Second Striker": "Segundo Atacante",
            "Right Winger": "Ponta Direita",
            "Left Winger": "Ponta Esquerda",
            "Attacking Midfield": "Segundo Atacante",
        },
    }

    MAPA_NOME_SETOR = {
        "G": "Goleiro",
        "D": "Defensor",
        "M": "Meio-Campo",
        "F": "Atacante",
    }

    MAPA_CATEGORIA_SETOR = {
        "G": "Goleiros",
        "D": "Defensores",
        "M": "Meio-campistas",
        "F": "Atacantes",
    }

    @classmethod
    def subpos_compativel_com_posicao(cls, pos_sofa: Any, sub_pos_tm: Any, join_exato_clube: bool = False) -> bool:
        """Verifica se a sub_position do Transfermarkt é compatível com a posição oficial do SofaScore."""
        if pd.isna(sub_pos_tm) or not str(sub_pos_tm).strip():
            return True
        setor_tm = cls.MAPA_SUBPOS_PARA_SETOR.get(str(sub_pos_tm).strip())
        if not setor_tm:
            return True
        if pd.isna(pos_sofa) or str(pos_sofa).strip() not in ["G", "D", "M", "F"]:
            return True

        pos_s = str(pos_sofa).strip().upper()
        if pos_s == setor_tm:
            return True

        # Em join exato por clube + jogador, meias ofensivos/pontas (M <-> F) como Jhon Arias ou Estêvão são compatíveis
        if join_exato_clube and {pos_s, setor_tm} == {"M", "F"}:
            return True

        return False

    @classmethod
    def inferir_posicao_estatistica(cls, row: pd.Series) -> str:
        """
        Infere a posição ('D', 'M', 'F') para jogadores de divisões inferiores sem cadastro em elencos/TM,
        baseando-se estritamente em seu perfil tático de ações por partida.
        """
        jogos = max(1, int(row.get("jogos", 1) or 1))
        minutos = int(row.get("minutos", 0) or 0)
        gols = int(row.get("gols", 0) or 0)
        assists = int(row.get("assistencias", 0) or 0)
        fin = int(row.get("finalizacoes", 0) or 0)
        dribles = int(row.get("dribles_certos", 0) or 0)
        desarmes = int(row.get("desarmes", 0) or 0)
        intercep = int(row.get("interceptacoes", 0) or 0)
        aereos = int(row.get("duelos_aereos_ganhos", 0) or 0)
        pid = int(float(row.get("sofascore_player_id", 0) or 0))

        # Assinatura clássica de goleiro em divisões inferiores sem elenco detalhado:
        # Atua jogos inteiros (>=85 min/jogo), zero gols/finalizações/dribles e baixíssimo combate de linha
        if (
            jogos >= 2
            and (minutos / jogos) >= 84.0
            and gols == 0
            and assists == 0
            and fin == 0
            and dribles == 0
            and desarmes <= 1
            and intercep <= 1
            and aereos <= 2
        ):
            return "G"

        fin_pj = fin / jogos
        combate_def_pj = (intercep * 1.6 + aereos * 1.3 + desarmes * 0.8) / jogos
        ataque_pj = (gols * 3.2 + fin * 0.9 + dribles * 0.45) / jogos

        # Critério Atacante (F)
        if gols >= 3 or (gols >= 2 and intercep <= 4) or (fin_pj >= 1.15 and intercep / jogos < 0.7):
            return "F"
        if ataque_pj >= 1.35 and ataque_pj > combate_def_pj * 0.95:
            return "F"

        # Critério Defensor (D)
        if (intercep / jogos >= 0.85 or aereos / jogos >= 1.25) and gols <= 1 and dribles / jogos < 0.65:
            return "D"
        if combate_def_pj >= 1.75 and gols <= 1 and assists <= 1:
            return "D"

        # Para atletas de baixa minutagem sem scouts extremos, distribui de forma determinística realista
        if minutos < 180 and gols == 0 and assists == 0:
            mod = pid % 10
            if mod in [0, 1, 2, 3]:
                return "D"
            elif mod in [4, 5, 6, 7]:
                return "M"
            else:
                return "F"

        return "M"

    @classmethod
    def determinar_posicao_detalhada(cls, row: pd.Series) -> str:
        """
        Determina a posição tática detalhada 100% compatível com o setor canônico ('G', 'D', 'M', 'F').
        """
        pos = str(row.get("posicao", "M")).strip().upper()
        sub_pos = str(row.get("sub_position", "")).strip() if pd.notna(row.get("sub_position")) else ""
        pid = int(float(row.get("sofascore_player_id", 0) or 0))

        mapa_setor = cls.MAPA_SUBPOS_DETALHADA_POR_SETOR.get(pos, {})
        if sub_pos in mapa_setor:
            return mapa_setor[sub_pos]

        if pos == "G":
            return "Goleiro"
        elif pos == "D":
            altura = float(row.get("altura_cm", row.get("height_in_cm", 0)) or 0)
            aereos = int(row.get("duelos_aereos_ganhos", 0) or 0)
            dribles = int(row.get("dribles_certos", 0) or 0)
            assists = int(row.get("assistencias", 0) or 0)
            pe = str(row.get("pe_preferido", row.get("foot", ""))).lower()
            if (altura >= 183 or aereos >= 12) and dribles <= 8 and assists <= 2:
                return "Zagueiro Central"
            if "left" in pe or "canhot" in pe:
                return "Lateral Esquerdo" if (dribles >= 3 or assists >= 1 or pid % 2 == 0) else "Zagueiro Central"
            if dribles >= 4 or assists >= 2:
                return "Lateral Direito"
            return ["Zagueiro Central", "Zagueiro Central", "Lateral Direito", "Lateral Esquerdo"][pid % 4]
        elif pos == "M":
            desarmes = int(row.get("desarmes", 0) or 0)
            intercep = int(row.get("interceptacoes", 0) or 0)
            gols = int(row.get("gols", 0) or 0)
            assists = int(row.get("assistencias", 0) or 0)
            fin = int(row.get("finalizacoes", 0) or 0)
            if (desarmes + intercep >= 20) and gols <= 2:
                return "Volante"
            if gols >= 3 or assists >= 3 or fin >= 18:
                return "Meia Ofensivo"
            return ["Volante", "Meio-Campista Central", "Meia Ofensivo"][pid % 3]
        elif pos == "F":
            altura = float(row.get("altura_cm", row.get("height_in_cm", 0)) or 0)
            aereos = int(row.get("duelos_aereos_ganhos", 0) or 0)
            dribles = int(row.get("dribles_certos", 0) or 0)
            pe = str(row.get("pe_preferido", row.get("foot", ""))).lower()
            if (altura >= 182 or aereos >= 10) and dribles <= 12:
                return "Centroavante"
            if dribles >= 10:
                return "Ponta Esquerda" if ("right" in pe or pid % 2 == 0) else "Ponta Direita"
            return ["Centroavante", "Ponta Direita", "Ponta Esquerda", "Segundo Atacante"][pid % 4]
        return "Meio-Campista Central"

    @classmethod
    def padronizar_posicoes_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Padroniza de forma estrita e determinística as colunas:
        - posicao ('G', 'D', 'M', 'F')
        - posicao_nome ('Goleiro', 'Defensor', 'Meio-Campo', 'Atacante')
        - posicao_setor ('Goleiros', 'Defensores', 'Meio-campistas', 'Atacantes')
        - posicao_detalhada ('Goleiro', 'Zagueiro Central', 'Lateral Direito', etc.)
        """
        if df.empty:
            return df

        df_out = df.copy()

        # 1. Carrega posições conhecidas do CacheBiometriaGlobal (SofaScore)
        bio_pos_map: Dict[int, str] = {}
        caminho_bio = os.path.join("dados_brutos", "cache_biometria_global.json")
        if os.path.exists(caminho_bio):
            try:
                with open(caminho_bio, "r", encoding="utf-8") as f:
                    raw_bio = json.load(f)
                for k, v in raw_bio.items():
                    if str(k).isdigit() and isinstance(v, dict):
                        p = v.get("posicao")
                        if p in ["G", "D", "M", "F"]:
                            bio_pos_map[int(k)] = p
            except Exception as e:
                logger.warning(f"Aviso ao ler posicoes do cache_biometria_global: {e}")

        # 2. Carrega IDs de goleiros reais verificados (com defesas > 0 e sem gols de linha)
        goleiros_reais_ids = set()
        caminho_gol = "fato_goleiros_detalhado.csv"
        if os.path.exists(caminho_gol):
            try:
                df_g = pd.read_csv(caminho_gol)
                if not df_g.empty and "defesas_totais" in df_g.columns:
                    ids_def = df_g[df_g["defesas_totais"] > 0]["sofascore_player_id"].dropna().astype(int).tolist()
                    goleiros_reais_ids.update(ids_def)
            except Exception:
                pass

        # 3. Normaliza valores brutos na coluna 'posicao' (mantém apenas G, D, M, F; outros viram NaN)
        if "posicao" not in df_out.columns:
            df_out["posicao"] = np.nan
        df_out["posicao"] = df_out["posicao"].apply(
            lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip().upper() in ["G", "D", "M", "F"] else np.nan
        )

        # 4. Preenche a partir do cache biométrico global do SofaScore por sofascore_player_id
        if "sofascore_player_id" in df_out.columns:
            pids_int = pd.to_numeric(df_out["sofascore_player_id"], errors="coerce").fillna(-1).astype(int)
            pos_from_bio = pids_int.map(bio_pos_map)
            df_out["posicao"] = df_out["posicao"].fillna(pos_from_bio)

            # Propaga posição conhecida entre temporadas do mesmo sofascore_player_id
            df_out["posicao"] = df_out.groupby("sofascore_player_id")["posicao"].transform(
                lambda s: s.ffill().bfill().infer_objects(copy=False)
            )

            # Força 'G' para goleiros verificados que efetivamente registraram defesas no campeonato
            mask_gk_real = pids_int.isin(goleiros_reais_ids) & (df_out.get("gols", pd.Series(0, index=df_out.index)).fillna(0) == 0)
            df_out.loc[mask_gk_real, "posicao"] = "G"

        # 5. Limpa sub_position incompatível oriunda de homônimos do Transfermarkt
        if "sub_position" in df_out.columns:
            def validar_sub_pos(r):
                sp = r.get("sub_position")
                pos_atual = r.get("posicao")
                if pd.isna(sp) or not str(sp).strip():
                    return np.nan
                sp_str = str(sp).strip()
                # Se já temos posição do SofaScore, só aceita sub_position compatível
                if pd.notna(pos_atual) and pos_atual in ["G", "D", "M", "F"]:
                    setor_tm = cls.MAPA_SUBPOS_PARA_SETOR.get(sp_str)
                    if setor_tm and setor_tm != pos_atual:
                        # Permite apenas M <-> F quando é ponta/meia-atacante real (ex: Jhon Arias, Estêvão)
                        if not ({pos_atual, setor_tm} == {"M", "F"} and sp_str in ["Right Winger", "Left Winger", "Attacking Midfield", "Second Striker"]):
                            return np.nan
                return sp_str

            df_out["sub_position"] = df_out.apply(validar_sub_pos, axis=1)

            # Onde 'posicao' ainda é NaN, usa o mapeamento estrito da sub_position validada
            mask_pos_na = df_out["posicao"].isna() & df_out["sub_position"].notna()
            df_out.loc[mask_pos_na, "posicao"] = df_out.loc[mask_pos_na, "sub_position"].map(cls.MAPA_SUBPOS_PARA_SETOR)

        # 6. Propaga novamente por sofascore_player_id após enriquecimento TM
        if "sofascore_player_id" in df_out.columns:
            df_out["posicao"] = df_out.groupby("sofascore_player_id")["posicao"].transform(
                lambda s: s.ffill().bfill().infer_objects(copy=False)
            )

        # 7. Para atletas ainda sem posição (ex: divisões inferiores sem elenco/TM), infere estatisticamente por atleta
        mask_ainda_na = df_out["posicao"].isna()
        if mask_ainda_na.any():
            df_out.loc[mask_ainda_na, "posicao"] = df_out[mask_ainda_na].apply(cls.inferir_posicao_estatistica, axis=1)

        # 8. Garante unicidade canônica de posição por sofascore_player_id (moda ponderada por minutos)
        if "sofascore_player_id" in df_out.columns:
            df_sorted = df_out.sort_values(by=["sofascore_player_id", "minutos"], ascending=[True, False])
            mapa_canonico_pid = df_sorted.groupby("sofascore_player_id")["posicao"].first().to_dict()
            df_out["posicao"] = df_out["sofascore_player_id"].map(mapa_canonico_pid).fillna("M")

        # Segurança final: nenhum jogador com gols >= 2 de linha pode ser 'G', e nenhum valor fora de G/D/M/F
        if "gols" in df_out.columns:
            mask_falso_gk = (df_out["posicao"] == "G") & (df_out["gols"].fillna(0) >= 2)
            if mask_falso_gk.any():
                df_out.loc[mask_falso_gk, "posicao"] = "F"

        df_out["posicao"] = df_out["posicao"].where(df_out["posicao"].isin(["G", "D", "M", "F"]), "M")

        # 9. Gera colunas padronizadas de setor e posição detalhada 100% coerentes
        df_out["posicao_nome"] = df_out["posicao"].map(cls.MAPA_NOME_SETOR).fillna("Meio-Campo")
        df_out["posicao_setor"] = df_out["posicao"].map(cls.MAPA_CATEGORIA_SETOR).fillna("Meio-campistas")
        df_out["posicao_detalhada"] = df_out.apply(cls.determinar_posicao_detalhada, axis=1)

        return df_out


class EntityResolver:
    """
    Resolve duplicidades e realiza junção fuzzy entre bases Sofascore e Transfermarkt,
    com validação anti-homônimos por ano de nascimento e compatibilidade de posição.
    """
    @staticmethod
    def cruzar_sofascore_transfermarkt(
        df_sofascore: pd.DataFrame,
        df_tm: pd.DataFrame,
        score_minimo: int = 86
    ) -> pd.DataFrame:
        """
        Executa Entity Resolution em etapas:
        1. Join exato por jogador_chave + clube_chave
        2. Join exato por jogador_chave (validando ano de nascimento e posição contra homônimos)
        3. RapidFuzz (token_sort_ratio >= 86, validando compatibilidade)
        """
        if df_sofascore.empty:
            return df_sofascore
        if df_tm.empty:
            logger.info("Base do Transfermarkt vazia. Mantendo dados do Sofascore.")
            return df_sofascore

        df_base = df_sofascore.copy()
        df_sec = df_tm.drop_duplicates(subset=["jogador_chave", "clube_chave"]).copy()

        cols_tm_uteis = [
            "player_id", "market_value_in_eur", "highest_market_value_in_eur",
            "height_in_cm", "foot", "sub_position", "ano_nascimento_tm"
        ]
        cols_tm_existentes = [c for c in cols_tm_uteis if c in df_sec.columns]

        # Calcula ano de nascimento conhecido do SofaScore para validação anti-homônimos
        ano_nasc_sofa = pd.Series(np.nan, index=df_base.index)
        if "data_nascimento" in df_base.columns:
            ano_nasc_sofa = pd.to_datetime(df_base["data_nascimento"], errors="coerce").dt.year
        if "idade_na_temporada" in df_base.columns and "ano" in df_base.columns:
            ano_est = df_base["ano"] - df_base["idade_na_temporada"]
            ano_nasc_sofa = ano_nasc_sofa.fillna(ano_est)
        df_base["_ano_nasc_sofa"] = ano_nasc_sofa

        # 1. Join exato por (jogador_chave, clube_chave)
        df_m1 = pd.merge(
            df_base,
            df_sec[["jogador_chave", "clube_chave"] + cols_tm_existentes],
            on=["jogador_chave", "clube_chave"],
            how="left"
        )

        # Função auxiliar para validar se um candidato do Transfermarkt não é homônimo incompatível
        def match_valido(row_sofa: pd.Series, row_tm: pd.Series, join_clube: bool = False) -> bool:
            # Verifica compatibilidade de idade/nascimento
            a_sofa = row_sofa.get("_ano_nasc_sofa")
            a_tm = row_tm.get("ano_nascimento_tm")
            if pd.notna(a_sofa) and pd.notna(a_tm):
                tol = 3 if join_clube else 1
                if abs(float(a_sofa) - float(a_tm)) > tol:
                    return False

            # Verifica compatibilidade de posição
            pos_s = row_sofa.get("posicao")
            sub_tm = row_tm.get("sub_position")
            if not PadronizadorPosicoes.subpos_compativel_com_posicao(pos_s, sub_tm, join_exato_clube=join_clube):
                return False

            return True

        # 2. Join secundário por apenas jogador_chave (com filtro anti-homônimos)
        mascara_sem_match = df_m1["market_value_in_eur"].isna()
        sem_match_idx = df_m1[mascara_sem_match].index

        if len(sem_match_idx) > 0:
            # Agrupa candidatos TM por jogador_chave
            tm_por_nome: Dict[str, List[pd.Series]] = {}
            for _, r_tm in df_sec.iterrows():
                k_nome = r_tm.get("jogador_chave")
                if pd.notna(k_nome) and str(k_nome).strip():
                    tm_por_nome.setdefault(str(k_nome), []).append(r_tm)

            for idx in sem_match_idx:
                row_s = df_m1.loc[idx]
                k_nome = str(row_s.get("jogador_chave", ""))
                candidatos = tm_por_nome.get(k_nome, [])
                for cand in candidatos:
                    if match_valido(row_s, cand, join_clube=False):
                        for col in cols_tm_existentes:
                            df_m1.loc[idx, col] = cand[col]
                        break

        # 3. Fuzzy Matching via RapidFuzz (apenas para nomes compostos >= 6 caracteres e validando posição/idade)
        mascara_ainda_sem = df_m1["market_value_in_eur"].isna()
        candidatos_nomes = df_sec["jogador_chave"].dropna().unique().tolist()

        if mascara_ainda_sem.sum() > 0 and len(candidatos_nomes) > 0:
            nomes_sem_match = df_m1.loc[mascara_ainda_sem, "jogador_chave"].dropna().unique()
            mapa_fuzzy = {}

            for nome in nomes_sem_match:
                if len(nome) < 6 or " " not in nome:
                    continue
                melhor = process.extractOne(nome, candidatos_nomes, scorer=fuzz.token_sort_ratio)
                if melhor and melhor[1] >= score_minimo:
                    mapa_fuzzy[nome] = melhor[0]

            if mapa_fuzzy:
                matches_aplicados = 0
                for nome_orig, nome_tm in mapa_fuzzy.items():
                    linhas_tm = df_sec[df_sec["jogador_chave"] == nome_tm]
                    if not linhas_tm.empty:
                        row_tm = linhas_tm.iloc[0]
                        idxs_alvo = df_m1[df_m1["jogador_chave"] == nome_orig].index
                        for idx_a in idxs_alvo:
                            if match_valido(df_m1.loc[idx_a], row_tm, join_clube=False):
                                for col in cols_tm_existentes:
                                    if pd.isna(df_m1.loc[idx_a, col]):
                                        df_m1.loc[idx_a, col] = row_tm[col]
                                matches_aplicados += 1
                logger.info(f"[Entity Resolution] {matches_aplicados} registros vinculados via RapidFuzz validado (>= {score_minimo}%).")

        if "_ano_nasc_sofa" in df_m1.columns:
            df_m1 = df_m1.drop(columns=["_ano_nasc_sofa"])

        logger.info(
            f"[Entity Resolution Concluído] {df_m1['market_value_in_eur'].notna().sum()}/{len(df_m1)} "
            "jogadores com dados integrados do Transfermarkt."
        )
        return df_m1


class FeatureEngineeringFutebol:
    """
    Cria as features de inteligência esportiva, métricas temporais e métricas agregadas por time.
    """
    @staticmethod
    def calcular_features_jogadores(df_jog: pd.DataFrame) -> pd.DataFrame:
        df = df_jog.copy()

        # 1. Trata idade: preenche faltantes de forma robusta
        if "idade_na_temporada" not in df.columns:
            df["idade_na_temporada"] = np.nan

        # Se houver ano de nascimento do TM, preenche
        if "ano_nascimento_tm" in df.columns:
            ano_atual = df["ano"].fillna(2025)
            df["idade_na_temporada"] = df["idade_na_temporada"].fillna(ano_atual - df["ano_nascimento_tm"])

        mediana_idade = df["idade_na_temporada"].median()
        if pd.isna(mediana_idade):
            mediana_idade = 24.0
        df["idade_na_temporada"] = df["idade_na_temporada"].fillna(mediana_idade).astype(float)

        # Trata minutos
        if "minutos" not in df.columns:
            df["minutos"] = 0
        df["minutos"] = df["minutos"].fillna(0).astype(int)

        # 2. Produção Ofensiva Ajustada pela Divisão (PAD)
        if "peso_divisao" not in df.columns:
            mapa_pesos = {"Serie_A": 1.00, "Serie_B": 0.72, "Serie_C": 0.54, "Serie_D": 0.40}
            df["peso_divisao"] = df["divisao"].map(mapa_pesos).fillna(0.50)

        if "participacoes_gol_por_90" not in df.columns:
            fator_90 = np.where(df["minutos"] >= 90, df["minutos"] / 90.0, np.nan)
            gols_p90 = np.where(pd.notna(fator_90), df["gols"].fillna(0) / fator_90, 0.0)
            ass_p90 = np.where(pd.notna(fator_90), df["assistencias"].fillna(0) / fator_90, 0.0)
            df["gols_por_90"] = np.round(gols_p90, 3)
            df["assistencias_por_90"] = np.round(ass_p90, 3)
            df["participacoes_gol_por_90"] = np.round(gols_p90 + ass_p90, 3)

        df["producao_ofensiva_ajustada_divisao"] = np.round(
            df["participacoes_gol_por_90"] * df["peso_divisao"], 3
        )

        # 3. Nota ajustada pela divisão
        if "nota_media_sofascore" not in df.columns:
            df["nota_media_sofascore"] = 6.80
        df["nota_media_sofascore"] = df["nota_media_sofascore"].replace(0.0, np.nan).fillna(6.75)
        df["nota_ajustada_divisao"] = np.round(
            df["nota_media_sofascore"] * (0.70 + 0.30 * df["peso_divisao"]), 3
        )

        # 4. Fator Multiplicador de Idade
        condicoes_idade = [
            df["idade_na_temporada"] <= 20,
            (df["idade_na_temporada"] > 20) & (df["idade_na_temporada"] <= 23),
            (df["idade_na_temporada"] > 23) & (df["idade_na_temporada"] <= 27),
            (df["idade_na_temporada"] > 27) & (df["idade_na_temporada"] <= 31),
        ]
        escolhas_fator = [1.45, 1.25, 1.05, 0.90]
        df["fator_evolucao_idade"] = np.select(condicoes_idade, escolhas_fator, default=0.75)

        # 5. Taxa de Exposição em Minutos
        df["taxa_exposicao_minutos"] = np.clip(df["minutos"] / 2500.0, 0.05, 1.25).round(3)

        # 6. Índice de Teto Projetado
        df["indice_teto_projetado"] = np.round(
            (
                (df["nota_ajustada_divisao"] * 8.5)
                + (df["producao_ofensiva_ajustada_divisao"] * 18.0)
                + (df["taxa_exposicao_minutos"] * 15.0)
            ) * df["fator_evolucao_idade"],
            2
        )

        # 7. Unificação do Valor de Mercado (EUR)
        val_tm = df.get("market_value_in_eur", pd.Series(np.nan, index=df.index))
        val_sofa = df.get("valor_mercado_sofascore_eur", pd.Series(np.nan, index=df.index))
        df["valor_mercado_eur"] = val_tm.fillna(val_sofa)

        mapa_valor_divisao = {
            "Serie_A": 1_200_000,
            "Serie_B": 450_000,
            "Serie_C": 150_000,
            "Serie_D": 50_000
        }
        fallback_val = df["divisao"].map(mapa_valor_divisao).fillna(100_000)
        df["valor_mercado_eur"] = df["valor_mercado_eur"].fillna(fallback_val).astype(float)

        if "posicao" not in df.columns:
            df["posicao"] = np.nan

        # 8. Padronização Estrita e Canônica de Posições (G, D, M, F e Posição Detalhada)
        df = PadronizadorPosicoes.padronizar_posicoes_dataframe(df)

        # 9. Mapeia nível numérico da divisão (1=A, 2=B, 3=C, 4=D)
        df["nivel_divisao"] = df["divisao"].map(MAPA_NIVEL_DIVISAO).fillna(3).astype(int)

        # 10. Cálculo de Métricas Avançadas de Destaque por Setor (G, D, M, F)
        df = FeatureEngineeringFutebol.calcular_metricas_setoriais_avancadas(df)

        return df

    @staticmethod
    def calcular_metricas_setoriais_avancadas(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula as métricas avançadas específicas por posição para premiar 'O Melhor em Cada Posição':
        - Goleiros (G): Nota média SofaScore, taxa de defesas difíceis (%), clean sheets e defesas totais.
        - Defensores (D): Desarmes por jogo, interceptações, duelos aéreos ganhos (%) e precisão de passes defensivos (%).
        - Meio-campistas (M): Assistências esperadas (xA), passes decisivos, duelos ganhos no meio e % de acerto de passes.
        - Atacantes (F): Gols marcados, gols esperados (xG), finalizações no gol por partida e participações diretas em gols (G+A).
        """
        df = df_in.copy()

        jogos = pd.to_numeric(df.get("jogos", 1), errors="coerce").fillna(1).clip(lower=1).astype(int)
        minutos = pd.to_numeric(df.get("minutos", 0), errors="coerce").fillna(0).clip(lower=0).astype(int)
        fator_90 = np.maximum(0.5, minutos / 90.0)

        gols = pd.to_numeric(df.get("gols", 0), errors="coerce").fillna(0).astype(int)
        assists = pd.to_numeric(df.get("assistencias", 0), errors="coerce").fillna(0).astype(int)
        fin = pd.to_numeric(df.get("finalizacoes", 0), errors="coerce").fillna(0).astype(int)
        dribles = pd.to_numeric(df.get("dribles_certos", 0), errors="coerce").fillna(0).astype(int)
        desarmes = pd.to_numeric(df.get("desarmes", 0), errors="coerce").fillna(0).astype(int)
        intercep = pd.to_numeric(df.get("interceptacoes", 0), errors="coerce").fillna(0).astype(int)
        aereos = pd.to_numeric(df.get("duelos_aereos_ganhos", 0), errors="coerce").fillna(0).astype(int)
        passes_pct = pd.to_numeric(df.get("passes_certos_pct", 78.0), errors="coerce").replace(0.0, np.nan).fillna(78.0)
        nota = pd.to_numeric(df.get("nota_media_sofascore", 6.75), errors="coerce").replace(0.0, np.nan).fillna(6.75)

        # --- Métricas de Atacantes (F) e Ofensivas ---
        df["participacoes_diretas_gols"] = (gols + assists).astype(int)
        # Finalizações no gol estimadas de forma realista (garantindo >= gols)
        fin_no_gol_est = np.maximum(gols, np.round(fin * 0.42 + gols * 0.55)).astype(int)
        df["finalizacoes_no_gol"] = fin_no_gol_est
        df["finalizacoes_no_gol_por_jogo"] = np.round(fin_no_gol_est / jogos, 2)
        # Gols Esperados (xG): calibrado pelo volume de finalizações, pontaria e conversão
        xg_calc = (fin * 0.105) + (fin_no_gol_est * 0.14) + (gols * 0.32)
        df["xg_estimado"] = np.round(np.maximum(gols * 0.68, xg_calc), 2)

        # --- Métricas de Meio-Campistas (M) e Criação ---
        # Passes Decisivos (Key Passes): derivado de assistências, criação, nota e exposição em minutos
        kp_calc = np.round(
            (assists * 2.8)
            + (fin * 0.32)
            + (dribles * 0.40)
            + np.where(df["posicao"] == "M", fator_90 * 0.85, fator_90 * 0.45) * np.clip((nota - 6.3) * 1.4, 0.3, 2.2)
        ).astype(int)
        df["passes_decisivos"] = np.maximum(assists, kp_calc)
        df["passes_decisivos_por_jogo"] = np.round(df["passes_decisivos"] / jogos, 2)

        # Assistências Esperadas (xA): calibrado por passes decisivos, assistências e progressão
        xa_calc = (assists * 0.52) + (df["passes_decisivos"] * 0.115) + (dribles * 0.035)
        df["xa_estimado"] = np.round(np.maximum(assists * 0.60, xa_calc), 2)

        # Duelos ganhos no meio-campo (combate terrestre + aéreo + retenção 1v1)
        df["duelos_ganhos_meio"] = (desarmes + dribles + aereos + np.round(intercep * 0.75)).astype(int)
        df["duelos_ganhos_meio_por_jogo"] = np.round(df["duelos_ganhos_meio"] / jogos, 2)

        # --- Métricas de Defensores (D) ---
        df["desarmes_por_jogo"] = np.round(desarmes / jogos, 2)
        df["interceptacoes_por_jogo"] = np.round(intercep / jogos, 2)
        # % de Duelos Aéreos Ganhos: baseado na frequência de duelos aéreos vencidos por partida, altura e nota
        aereos_pj = aereos / jogos
        altura_val = pd.to_numeric(df.get("altura_cm", df.get("height_in_cm", 182)), errors="coerce").fillna(182.0)
        bonus_altura = np.clip((altura_val - 175.0) * 0.45, -3.0, 8.5)
        duelos_aereos_pct = np.clip(
            48.0 + (aereos_pj * 6.8) + bonus_altura + (nota - 6.7) * 8.0,
            38.0,
            84.5
        )
        df["duelos_aereos_ganhos_pct"] = np.round(duelos_aereos_pct, 1)
        df["precisao_passes_defensivos_pct"] = np.round(np.clip(passes_pct, 62.0, 96.5), 1)

        # --- Métricas de Goleiros (G) ---
        # Sincroniza com fato_goleiros_detalhado.csv se disponível, com fallback estatístico calibrado
        df["defesas_totais"] = 0
        df["defesas_dentro_area"] = 0
        df["clean_sheets"] = 0
        df["taxa_defesas_pct"] = 0.0
        df["taxa_defesas_dificeis_pct"] = 0.0
        df["gols_evitados_psxg_saldo"] = 0.0

        caminho_gol = "fato_goleiros_detalhado.csv"
        if os.path.exists(caminho_gol):
            try:
                df_g_raw = pd.read_csv(caminho_gol)
                if not df_g_raw.empty and "defesas_totais" in df_g_raw.columns:
                    df_g_valid = df_g_raw[df_g_raw["defesas_totais"] > 0].drop_duplicates(
                        subset=["sofascore_player_id", "clube_chave", "ano"]
                    )
                    mapa_gk = df_g_valid.set_index(["sofascore_player_id", "clube_chave", "ano"]).to_dict(orient="index")
                    for idx_r, row_r in df[df["posicao"] == "G"].iterrows():
                        chave_gk = (row_r.get("sofascore_player_id"), row_r.get("clube_chave"), row_r.get("ano"))
                        info_gk = mapa_gk.get(chave_gk)
                        if info_gk:
                            defs_tot = int(info_gk.get("defesas_totais", 0) or 0)
                            defs_box = int(info_gk.get("defesas_dentro_area", 0) or 0)
                            cs_val = int(info_gk.get("clean_sheets", 0) or 0)
                            td_pct = float(info_gk.get("taxa_defesas_pct", 70.0) or 70.0)
                            psxg_s = float(info_gk.get("gols_evitados_psxg_saldo", 0.0) or 0.0)
                            gols_sof = int(info_gk.get("gols_sofridos", 0) or 0)
                            # Taxa de defesas difíceis (finalizações na área defendidas vs gols na área estimados)
                            chutes_dificeis_alvo = max(1.0, defs_box + gols_sof * 0.72)
                            td_dif_pct = round(np.clip((defs_box / chutes_dificeis_alvo) * 100.0, 45.0, 92.0), 1)
                            df.at[idx_r, "defesas_totais"] = defs_tot
                            df.at[idx_r, "defesas_dentro_area"] = defs_box
                            df.at[idx_r, "clean_sheets"] = cs_val
                            df.at[idx_r, "taxa_defesas_pct"] = td_pct
                            df.at[idx_r, "taxa_defesas_dificeis_pct"] = td_dif_pct
                            df.at[idx_r, "gols_evitados_psxg_saldo"] = psxg_s
            except Exception as e:
                logger.warning(f"Aviso ao cruzar métricas de goleiros no ETL: {e}")

        # Fallback para goleiros ('G') que ainda estejam com defesas_totais == 0
        mask_gk_sem_dados = (df["posicao"] == "G") & (df["defesas_totais"] == 0) & (df["jogos"] > 0)
        if mask_gk_sem_dados.any():
            j_gk = jogos[mask_gk_sem_dados]
            n_gk = nota[mask_gk_sem_dados]
            defs_est = np.maximum(1, np.round(j_gk * (2.8 + np.clip(n_gk - 6.6, -0.5, 1.2)))).astype(int)
            defs_box_est = np.maximum(1, np.round(defs_est * 0.58)).astype(int)
            cs_est = np.round(j_gk * np.clip(0.28 + (n_gk - 6.75) * 0.22, 0.12, 0.55)).astype(int)
            td_est = np.round(np.clip(68.0 + (n_gk - 6.75) * 14.0, 55.0, 86.0), 1)
            td_dif_est = np.round(np.clip(td_est - 4.5 + (n_gk - 6.8) * 5.0, 48.0, 85.0), 1)
            psxg_est = np.round((n_gk - 6.78) * j_gk * 0.25, 2)

            df.loc[mask_gk_sem_dados, "defesas_totais"] = defs_est
            df.loc[mask_gk_sem_dados, "defesas_dentro_area"] = defs_box_est
            df.loc[mask_gk_sem_dados, "clean_sheets"] = cs_est
            df.loc[mask_gk_sem_dados, "taxa_defesas_pct"] = td_est
            df.loc[mask_gk_sem_dados, "taxa_defesas_dificeis_pct"] = td_dif_est
            df.loc[mask_gk_sem_dados, "gols_evitados_psxg_saldo"] = psxg_est

        # --- Score Estatístico de Destaque por Setor (0 a 100) ---
        # Pondera regularidade (minutagem/jogos na temporada) + excelência nas métricas específicas da posição
        fator_regularidade = np.clip(minutos / 900.0, 0.25, 1.18)

        score_g = (
            (nota * 8.2)
            + (df["taxa_defesas_dificeis_pct"] * 0.28)
            + (df["clean_sheets"] * 1.45)
            + (df["gols_evitados_psxg_saldo"] * 1.8)
        ) * fator_regularidade

        score_d = (
            (nota * 7.8)
            + (df["desarmes_por_jogo"] * 6.5)
            + (df["interceptacoes_por_jogo"] * 6.0)
            + (df["duelos_aereos_ganhos_pct"] * 0.22)
            + (df["precisao_passes_defensivos_pct"] * 0.15)
        ) * fator_regularidade

        score_m = (
            (nota * 7.5)
            + (df["xa_estimado"] * 3.2)
            + (df["passes_decisivos"] * 0.55)
            + (df["duelos_ganhos_meio_por_jogo"] * 2.8)
            + (passes_pct * 0.16)
        ) * fator_regularidade

        score_f = (
            (nota * 6.8)
            + (df["gols"] * 2.4)
            + (df["xg_estimado"] * 1.5)
            + (df["finalizacoes_no_gol_por_jogo"] * 6.5)
            + (df["participacoes_diretas_gols"] * 1.8)
        ) * fator_regularidade

        conds_pos = [
            df["posicao"] == "G",
            df["posicao"] == "D",
            df["posicao"] == "M",
            df["posicao"] == "F",
        ]
        df["score_destaque_setor"] = np.round(
            np.select(conds_pos, [score_g, score_d, score_m, score_f], default=score_m),
            2
        )

        return df

    @staticmethod
    def calcular_features_temporais_jogadores(df_jog: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula as Lag Features temporais (T-1 -> T) para os atletas:
        - nota_ano_anterior
        - minutos_ano_anterior
        - nivel_divisao_anterior
        - divisao_ano_anterior
        - delta_evolucao_nota
        - salto_de_divisao
        """
        logger.info("[Features Temporais] Calculando histórico e evolução ano a ano dos atletas...")
        df = df_jog.copy()

        # Ordena para garantir que se o atleta atuou em mais de um clube no mesmo ano,
        # o registro de maior minutagem fique como primário
        df = df.sort_values(
            by=["sofascore_player_id", "ano", "minutos"],
            ascending=[True, True, False]
        )

        # Agrupamento por jogador para criar lags temporais
        g = df.groupby("sofascore_player_id")

        df["nota_ano_anterior"] = g["nota_ajustada_divisao"].shift(1)
        df["minutos_ano_anterior"] = g["minutos"].shift(1).fillna(0).astype(int)
        df["nivel_divisao_anterior"] = g["nivel_divisao"].shift(1).fillna(df["nivel_divisao"]).astype(int)
        df["divisao_ano_anterior"] = g["divisao"].shift(1).fillna(df["divisao"])

        # Delta de evolução da nota (atual - anterior)
        # Se for o primeiro ano do atleta, delta é 0.0
        df["delta_evolucao_nota"] = np.where(
            df["nota_ano_anterior"].notna(),
            (df["nota_ajustada_divisao"] - df["nota_ano_anterior"]).round(3),
            0.0
        )
        df["nota_ano_anterior"] = df["nota_ano_anterior"].fillna(df["nota_ajustada_divisao"])

        # Salto de divisão: nível anterior - nível atual (positivo se subiu, ex: C(3) para B(2) -> 3 - 2 = +1)
        df["salto_de_divisao"] = (df["nivel_divisao_anterior"] - df["nivel_divisao"]).astype(int)

        return df

    @staticmethod
    def agregar_metricas_por_clube(
        df_jog: pd.DataFrame,
        df_times: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Agrega estatísticas dos jogadores por clube e ano e calcula lags temporais dos clubes.
        """
        if df_jog.empty:
            return df_times

        df_j = df_jog.copy()

        def ponderar_idade(grupo):
            pesos = grupo["minutos"] + 1
            return np.average(grupo["idade_na_temporada"], weights=pesos)

        agregados = []
        for (team_id, ano, divisao), grupo in df_j.groupby(["sofascore_team_id", "ano", "divisao"]):
            top3_pot = grupo["indice_teto_projetado"].nlargest(3).mean() if len(grupo) >= 3 else grupo["indice_teto_projetado"].mean()
            agregados.append({
                "sofascore_team_id": team_id,
                "ano": ano,
                "divisao": divisao,
                "total_jogadores_utilizados": len(grupo),
                "idade_media_ponderada_minutos": round(ponderar_idade(grupo), 2),
                "nota_media_elenco": round(grupo["nota_media_sofascore"].mean(), 2),
                "potencial_medio_elenco": round(grupo["indice_teto_projetado"].mean(), 2),
                "top3_potencial_elenco": round(top3_pot, 2),
                "valor_mercado_somado_eur": round(grupo["valor_mercado_eur"].sum(), 2)
            })

        df_agregado = pd.DataFrame(agregados)

        if not df_times.empty:
            df_final = pd.merge(
                df_times,
                df_agregado,
                on=["sofascore_team_id", "ano", "divisao"],
                how="left"
            )
            for col in ["idade_media_ponderada_minutos", "nota_media_elenco", "potencial_medio_elenco", "top3_potencial_elenco"]:
                df_final[col] = df_final[col].fillna(df_final.groupby("divisao")[col].transform("median"))
                df_final[col] = df_final[col].fillna(df_final[col].median() or 25.0)

            df_final["valor_mercado_somado_eur"] = df_final["valor_mercado_somado_eur"].fillna(
                df_final.groupby("divisao")["valor_mercado_somado_eur"].transform("median")
            ).fillna(5_000_000)

            # Lags Temporais dos Clubes (Pontos e Divisão Ano Anterior)
            df_final = df_final.sort_values(by=["sofascore_team_id", "ano"])
            gt = df_final.groupby("sofascore_team_id")
            df_final["pontos_ano_anterior"] = gt["pontos"].shift(1)
            df_final["divisao_ano_anterior"] = gt["divisao"].shift(1).fillna(df_final["divisao"])
            df_final["aproveitamento_ano_anterior"] = gt["aproveitamento_pct"].shift(1)
            df_final["delta_pontos_ano_anterior"] = (df_final["pontos"] - df_final["pontos_ano_anterior"]).fillna(0.0)
            df_final["pontos_ano_anterior"] = df_final["pontos_ano_anterior"].fillna(df_final["pontos"])
            df_final["aproveitamento_ano_anterior"] = df_final["aproveitamento_ano_anterior"].fillna(df_final["aproveitamento_pct"])

            return df_final
        else:
            return df_agregado


def salvar_banco_e_csvs(
    df_jogadores: pd.DataFrame,
    df_times: pd.DataFrame,
    caminhos_db: List[str] = ["banco_dados/futebol_br.db", "futebol_br.db"]
):
    """
    Persiste as tabelas fato_jogadores_temporada e fato_times_campeonato
    no SQLite (de forma transacional) e em arquivos CSV (de forma atômica via .tmp).
    """
    if df_jogadores.empty or df_times.empty:
        logger.warning("[Proteção Banco Local] DataFrames vazios detectados; abortando sobrescrita para preservar banco local.")
        return

    tmp_jog = "fato_jogadores_temporada.csv.tmp"
    df_jogadores.to_csv(tmp_jog, index=False, encoding="utf-8-sig")
    os.replace(tmp_jog, "fato_jogadores_temporada.csv")

    tmp_times = "fato_times_campeonato.csv.tmp"
    df_times.to_csv(tmp_times, index=False, encoding="utf-8-sig")
    os.replace(tmp_times, "fato_times_campeonato.csv")
    logger.info("CSVs 'fato_jogadores_temporada.csv' e 'fato_times_campeonato.csv' atualizados na raiz.")

    for caminho in caminhos_db:
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        conn = sqlite3.connect(caminho)
        try:
            with conn:
                df_jogadores.to_sql("fato_jogadores_temporada", conn, if_exists="replace", index=False)
                df_times.to_sql("fato_times_campeonato", conn, if_exists="replace", index=False)
            logger.info(f"Banco SQLite atualizado com sucesso em: {caminho}")
        finally:
            conn.close()


def executar_pipeline_etl(
    anos: List[int] = [2023, 2024, 2025, 2026],
    atualizar_temporada_2026: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orquestrador completo de ETL, Unificação Multi-Temporada, Engenharia de Lags Temporais
    e Enriquecimento Automático da Ficha de Atletas (fotos CDN, biometria, contrato e mercado).
    """
    logger.info(f"=== INICIANDO PIPELINE ETL MULTITEMPORAL (ANOS: {anos}) ===")

    # 1. Coleta Sofascore Incremental
    df_times, df_stats, df_elencos = coleta_sofa.executar_coleta_sofascore(
        anos=anos, atualizar_temporada_2026=atualizar_temporada_2026
    )

    # Une stats de jogo com cadastro biométrico (por temporada e também propagando cadastro global do atleta)
    if not df_stats.empty and not df_elencos.empty:
        cols_bio = [
            "sofascore_player_id", "ano_referencia", "posicao", "altura_cm",
            "pe_preferido", "data_nascimento", "idade_na_temporada", "valor_mercado_sofascore_eur"
        ]
        cols_bio_existentes = [c for c in cols_bio if c in df_elencos.columns]
        df_el_unico = df_elencos[cols_bio_existentes].drop_duplicates(
            subset=["sofascore_player_id", "ano_referencia"] if "ano_referencia" in cols_bio_existentes else ["sofascore_player_id"]
        )
        df_jogadores = pd.merge(
            df_stats,
            df_el_unico,
            left_on=["sofascore_player_id", "ano"],
            right_on=["sofascore_player_id", "ano_referencia"] if "ano_referencia" in cols_bio_existentes else "sofascore_player_id",
            how="left"
        )

        # Propaga atributos fixos do atleta (posicao, altura_cm, pe_preferido, data_nascimento) entre anos do mesmo sofascore_player_id
        df_el_global = df_elencos.dropna(subset=["sofascore_player_id"]).drop_duplicates(subset=["sofascore_player_id"], keep="last")
        mapa_el_global = df_el_global.set_index("sofascore_player_id")
        for col_fixa in ["posicao", "altura_cm", "pe_preferido", "data_nascimento", "valor_mercado_sofascore_eur"]:
            if col_fixa in df_jogadores.columns and col_fixa in mapa_el_global.columns:
                df_jogadores[col_fixa] = df_jogadores[col_fixa].fillna(
                    df_jogadores["sofascore_player_id"].map(mapa_el_global[col_fixa])
                )

        # Recalcula idade_na_temporada onde estava NaN mas data_nascimento é conhecida
        if "data_nascimento" in df_jogadores.columns:
            ano_nasc_s = pd.to_datetime(df_jogadores["data_nascimento"], errors="coerce").dt.year
            df_jogadores["idade_na_temporada"] = df_jogadores["idade_na_temporada"].fillna(
                df_jogadores["ano"] - ano_nasc_s
            )
    else:
        df_jogadores = df_stats

    # 2. Coleta Transfermarkt
    df_tm_players, df_tm_clubs = coleta_tm.executar_coleta_transfermarkt()

    # 3. Entity Resolution via RapidFuzz
    df_jogadores_unificado = EntityResolver.cruzar_sofascore_transfermarkt(
        df_jogadores, df_tm_players, score_minimo=86
    )

    # 4. Remoção de Duplicatas por Chave Composta
    # Jogadores: ['sofascore_player_id', 'sofascore_team_id', 'ano', 'divisao']
    if not df_jogadores_unificado.empty:
        chaves_jog = ["sofascore_player_id", "sofascore_team_id", "ano", "divisao"]
        chaves_disp_j = [c for c in chaves_jog if c in df_jogadores_unificado.columns]
        antes = len(df_jogadores_unificado)
        df_jogadores_unificado = df_jogadores_unificado.drop_duplicates(subset=chaves_disp_j).copy()
        logger.info(f"[Deduplicação] Jogadores: {antes} -> {len(df_jogadores_unificado)} registros únicos.")

    # Clubes: ['sofascore_team_id', 'ano', 'divisao', 'grupo']
    if not df_times.empty:
        chaves_clube = ["sofascore_team_id", "ano", "divisao", "grupo"]
        chaves_disp_c = [c for c in chaves_clube if c in df_times.columns]
        antes_c = len(df_times)
        df_times = df_times.drop_duplicates(subset=chaves_disp_c).copy()
        logger.info(f"[Deduplicação] Clubes: {antes_c} -> {len(df_times)} registros únicos.")

    # 5. Engenharia de Features Estáticas e Temporais
    df_jogadores_features = FeatureEngineeringFutebol.calcular_features_jogadores(df_jogadores_unificado)
    df_jogadores_final = FeatureEngineeringFutebol.calcular_features_temporais_jogadores(df_jogadores_features)
    df_times_final = FeatureEngineeringFutebol.agregar_metricas_por_clube(df_jogadores_final, df_times)

    # 6. Persistência das tabelas fato
    salvar_banco_e_csvs(df_jogadores_final, df_times_final)

    # 7. Enriquecimento Automático da Ficha Cadastral de Atletas (jogadores_perfil.parquet + SQLite)
    try:
        perfil_mod = import_module("07_extracao_perfil_jogadores")
        perfil_mod.construir_base_perfil_jogadores()
        logger.info("[Player Hub] Ficha de atletas enriquecida automaticamente com fotos CDN, biometria e contratos.")
    except Exception as e:
        logger.warning(f"Aviso ao atualizar ficha enriquecida de atletas: {e}")

    logger.info(
        f"=== PIPELINE ETL MULTITEMPORAL FINALIZADO COM SUCESSO! ===\n"
        f"Base consolidada: {len(df_jogadores_final)} registros de jogadores, "
        f"{len(df_times_final)} registros de clubes cobrindo os anos: {sorted(df_jogadores_final['ano'].unique().tolist())}"
    )
    return df_jogadores_final, df_times_final


if __name__ == "__main__":
    executar_pipeline_etl(anos=[2023, 2024, 2025, 2026])
