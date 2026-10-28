"""
================================================================================
MÓDULO 5: VISUALIZAÇÃO TÁTICA AVANÇADA (SOFASCORE ON-DEMAND + MPLSOCCER + RADAR)
================================================================================
Responsável por:
1. Buscar dados on-demand com cache em disco (dados_brutos/cache_heatmaps/):
   - Heatmap individual (/heatmap/overall)
   - Shotmap individual com xG (/shotmap)
2. Desenhar mapas de campo com mplsoccer:
   - Heatmap do Jogador na Temporada (Pitch KDE)
   - Shotmap de Finalizações e xG (VerticalPitch Half)
   - Heatmap Coletivo do Time + Posição Média dos 11 Titulares
3. Calcular percentis por posição/ano e gerar Gráficos de Radar (Spider Chart)
   comparativos interativos com Plotly.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mplsoccer import Pitch, VerticalPitch
import plotly.graph_objects as go

try:
    from curl_cffi import requests as cf_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as cf_requests
    HAS_CURL_CFFI = False

logger = logging.getLogger("VisualizacaoTatica")

# Mapeamento Oficial de Torneios e Temporadas no Sofascore
TOURNAMENT_MAP = {
    "Serie_A": 325,
    "Serie_B": 390,
    "Serie_C": 1281,
    "Serie_D": 10326,
    "A": 325,
    "B": 390,
    "C": 1281,
    "D": 10326,
}

SEASON_MAP = {
    325: {2023: 48982, 2024: 58766, 2025: 72034, 2026: 87678},
    390: {2023: 49058, 2024: 59015, 2025: 72603, 2026: 89840},
    1281: {2023: 49328, 2024: 59016, 2025: 72841, 2026: 90642},
    10326: {2023: 49355, 2024: 59244, 2025: 72842, 2026: 92447},
}

DIR_CACHE_HEATMAPS = os.path.join("dados_brutos", "cache_heatmaps")
os.makedirs(DIR_CACHE_HEATMAPS, exist_ok=True)


def obter_tournament_e_season_id(divisao: str, ano: int) -> Tuple[Optional[int], Optional[int]]:
    """Identifica o tournament_id e season_id oficial do Sofascore."""
    div_limpa = str(divisao).replace("Série ", "Serie_").strip()
    tid = TOURNAMENT_MAP.get(div_limpa)
    if not tid:
        for k, v in TOURNAMENT_MAP.items():
            if k in div_limpa:
                tid = v
                break
    if not tid:
        tid = 325  # Padrão Série A se não identificado

    ano_int = int(ano)
    sid = SEASON_MAP.get(tid, {}).get(ano_int)
    return tid, sid


def requisicao_sofascore(url: str, timeout: int = 6) -> Optional[Dict[str, Any]]:
    """Faz requisição rápida e direta à API do Sofascore com curl_cffi."""
    u = url.replace("api.sofascore.com", "www.sofascore.com")
    kwargs: Dict[str, Any] = {"timeout": timeout}
    if HAS_CURL_CFFI:
        kwargs["impersonate"] = "chrome"

    try:
        resp = cf_requests.get(u, **kwargs)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Falha na requisição {u}: {e}")

    return None


def buscar_heatmap_jogador(
    player_id: int,
    tournament_id: int,
    season_id: int
) -> List[Dict[str, Any]]:
    """
    Busca os pontos de heatmap do jogador na temporada.
    Possui cache inteligente em disco (JSON).
    """
    cache_path = os.path.join(DIR_CACHE_HEATMAPS, f"heatmap_{player_id}_{tournament_id}_{season_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                dados = json.load(f)
                return dados.get("points", [])
        except Exception:
            pass

    url = f"https://www.sofascore.com/api/v1/player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/heatmap/overall"
    dados = requisicao_sofascore(url)
    pontos = []
    if dados and isinstance(dados, dict):
        pontos = dados.get("points", [])

    # Salva cache
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"points": pontos}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Erro ao salvar cache de heatmap: {e}")

    return pontos


def buscar_shotmap_jogador(
    player_id: int,
    tournament_id: int,
    season_id: int
) -> List[Dict[str, Any]]:
    """
    Busca o mapa de chutes (shotmap com xG) do jogador na temporada.
    Possui cache inteligente em disco (JSON).
    """
    cache_path = os.path.join(DIR_CACHE_HEATMAPS, f"shotmap_{player_id}_{tournament_id}_{season_id}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                dados = json.load(f)
                return dados.get("shotmap", [])
        except Exception:
            pass

    url = f"https://www.sofascore.com/api/v1/player/{player_id}/unique-tournament/{tournament_id}/season/{season_id}/shotmap"
    dados = requisicao_sofascore(url)
    shots_planificados = []
    if dados and isinstance(dados, dict):
        eventos = dados.get("shotmap", [])
        for ev in eventos:
            ev_info = ev.get("event", {})
            for s in ev.get("shots", []):
                s_copy = dict(s)
                s_copy["event_slug"] = ev_info.get("slug", "")
                shots_planificados.append(s_copy)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"shotmap": shots_planificados}, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Erro ao salvar cache de shotmap: {e}")

    return shots_planificados


def plotar_heatmap_jogador(
    pontos: List[Dict[str, Any]],
    jogador_nome: str,
    clube_nome: str,
    ano: int,
    posicao: str = ""
) -> plt.Figure:
    """
    Desenha o gramado com mapa de calor (KDE fluido) usando mplsoccer.
    """
    pitch = Pitch(
        pitch_type="opta",
        pitch_color="#181d24",
        line_color="#455a64",
        linewidth=1.5,
        goal_type="box"
    )
    fig, ax = pitch.draw(figsize=(8.5, 5.6))
    fig.patch.set_facecolor("#181d24")

    if not pontos or len(pontos) < 5:
        ax.text(
            50, 50,
            "Sem dados de mapa de calor suficientes para esta temporada\n(Atleta atuou poucos minutos ou competição sem rastreio de GPS)",
            ha="center", va="center",
            color="#90a4ae", fontsize=11, fontweight="bold"
        )
        ax.set_title(f"Mapa de Calor: {jogador_nome} ({clube_nome} - {ano})", color="white", fontsize=13, pad=12)
        return fig

    xs = [p.get("x", 0) for p in pontos]
    ys = [p.get("y", 0) for p in pontos]

    # Contorno fluido de calor (KDE)
    try:
        kde = pitch.kdeplot(
            xs, ys,
            ax=ax,
            cmap="YlOrRd",
            fill=True,
            levels=60,
            thresh=0.08,
            alpha=0.68,
            cut=4
        )
    except Exception:
        # Fallback para hexbin caso KDE encontre matriz singular
        pitch.hexbin(xs, ys, ax=ax, edgecolors="none", gridsize=(18, 12), cmap="YlOrRd", alpha=0.7)

    # Adiciona seta de sentido de ataque
    pitch.arrows(
        15, 5, 85, 5,
        ax=ax,
        color="#78909c",
        width=1.5,
        headwidth=4,
        headlength=4
    )
    ax.text(50, 7.5, "Direção de Ataque →", color="#b0bec5", fontsize=9, ha="center", fontweight="bold")

    sub_pos = f" - {posicao}" if posicao else ""
    ax.set_title(
        f"🔥 Mapa de Calor no Gramado: {jogador_nome} ({clube_nome} {ano}{sub_pos})\nTotal de Ações Mapeadas: {len(pontos)}",
        color="#f1f5f9",
        fontsize=12,
        fontweight="bold",
        pad=10
    )
    return fig


def plotar_shotmap_jogador(
    shots: List[Dict[str, Any]],
    jogador_nome: str,
    clube_nome: str,
    ano: int
) -> plt.Figure:
    """
    Plota meio-campo ofensivo com mapa de finalizações destacando gols e xG.
    """
    pitch = VerticalPitch(
        pitch_type="opta",
        half=True,
        pitch_color="#181d24",
        line_color="#455a64",
        linewidth=1.5,
        goal_type="box"
    )
    fig, ax = pitch.draw(figsize=(6.5, 6))
    fig.patch.set_facecolor("#181d24")

    if not shots:
        ax.text(
            50, 78,
            "Nenhuma finalização registrada nesta temporada\n(ou sem dados de xG disponíveis)",
            ha="center", va="center",
            color="#90a4ae", fontsize=11, fontweight="bold"
        )
        ax.set_title(f"Mapa de Finalizações: {jogador_nome} ({clube_nome} - {ano})", color="white", fontsize=12, pad=12)
        return fig

    total_xg = 0.0
    gols = 0
    finalizacoes = len(shots)

    for s in shots:
        coords = s.get("playerCoordinates", {})
        if not coords or "x" not in coords:
            continue

        raw_x = coords.get("x", 0)
        raw_y = coords.get("y", 50)
        # Opta: 100 é a linha do gol adversário
        pitch_x = 100.0 - raw_x
        pitch_y = raw_y

        xg = s.get("xg", 0.05) or 0.05
        total_xg += xg
        shot_type = s.get("shotType", "").lower()
        body_part = s.get("bodyPart", "pé")

        tam = max(35, min(xg * 650, 320))

        if shot_type == "goal":
            gols += 1
            # Gol: Estrela dourada/verde brilhante com borda preta
            pitch.scatter(
                pitch_x, pitch_y,
                ax=ax,
                s=tam * 1.6,
                marker="*",
                color="#00e676",
                edgecolors="#ffffff",
                linewidth=1.4,
                zorder=5,
                label="Gol" if gols == 1 else ""
            )
        elif shot_type == "save":
            pitch.scatter(
                pitch_x, pitch_y,
                ax=ax,
                s=tam,
                marker="o",
                color="#00b0ff",
                edgecolors="#ffffff",
                linewidth=0.8,
                alpha=0.85,
                zorder=4,
                label="Defesa" if "Defesa" not in [l.get_label() for l in ax.collections] else ""
            )
        elif shot_type == "post":
            pitch.scatter(
                pitch_x, pitch_y,
                ax=ax,
                s=tam,
                marker="s",
                color="#ffea00",
                edgecolors="#ffffff",
                linewidth=0.8,
                alpha=0.9,
                zorder=4,
                label="Trave" if "Trave" not in [l.get_label() for l in ax.collections] else ""
            )
        else:  # miss / block
            pitch.scatter(
                pitch_x, pitch_y,
                ax=ax,
                s=tam,
                marker="o",
                color="#ff5252",
                edgecolors="#181d24",
                linewidth=0.8,
                alpha=0.75,
                zorder=3,
                label="Para fora / Bloqueado" if "Para fora / Bloqueado" not in [l.get_label() for l in ax.collections] else ""
            )

    ax.set_title(
        f"🎯 Mapa de Finalizações: {jogador_nome}\n{clube_nome} ({ano}) | Chutes: {finalizacoes} | Gols: {gols} | xG Acumulado: {total_xg:.2f}",
        color="#f1f5f9",
        fontsize=11,
        fontweight="bold",
        pad=10
    )

    # Legenda estilizada
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        leg = ax.legend(
            by_label.values(), by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            fontsize=8,
            frameon=True,
            facecolor="#263238",
            edgecolor="#455a64",
            labelcolor="#eceff1"
        )
    return fig


def buscar_heatmap_coletivo_time(
    df_jogadores_clube: pd.DataFrame,
    tournament_id: int,
    season_id: int,
    top_n: int = 11
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Busca heatmaps dos 11 titulares mais utilizados do time e calcula a posição média de cada um.
    """
    titulares = df_jogadores_clube.sort_values(by="minutos", ascending=False).head(top_n)
    todos_pontos: List[Dict[str, Any]] = []
    posicoes_medias: List[Dict[str, Any]] = []

    for _, row in titulares.iterrows():
        pid = row.get("sofascore_player_id")
        nome = row.get("jogador_nome", "Atleta")
        pos = row.get("posicao", "-")
        mins = row.get("minutos", 0)

        if pd.isna(pid):
            continue

        pts = buscar_heatmap_jogador(int(pid), tournament_id, season_id)
        if pts:
            todos_pontos.extend(pts)
            xs = [p["x"] for p in pts if "x" in p]
            ys = [p["y"] for p in pts if "y" in p]
            if xs and ys:
                posicoes_medias.append({
                    "jogador_nome": nome,
                    "posicao": pos,
                    "minutos": mins,
                    "media_x": float(np.mean(xs)),
                    "media_y": float(np.mean(ys)),
                    "acoes": len(pts)
                })

    return todos_pontos, posicoes_medias


def plotar_heatmap_coletivo_time(
    todos_pontos: List[Dict[str, Any]],
    posicoes_medias: List[Dict[str, Any]],
    clube_nome: str,
    ano: int,
    divisao: str = ""
) -> plt.Figure:
    """
    Desenha o mapa de calor coletivo do elenco e plota a posição tática média dos 11 titulares.
    """
    pitch = Pitch(
        pitch_type="opta",
        pitch_color="#181d24",
        line_color="#455a64",
        linewidth=1.5,
        goal_type="box"
    )
    fig, ax = pitch.draw(figsize=(10, 6.5))
    fig.patch.set_facecolor("#181d24")

    if not todos_pontos:
        ax.text(
            50, 50,
            f"Sem dados de heatmap suficientes para os titulares do {clube_nome} em {ano}",
            ha="center", va="center",
            color="#90a4ae", fontsize=11, fontweight="bold"
        )
        ax.set_title(f"Ocupação Tática: {clube_nome} ({ano})", color="white", fontsize=13, pad=12)
        return fig

    xs = [p.get("x", 0) for p in todos_pontos]
    ys = [p.get("y", 0) for p in todos_pontos]

    # Densidade coletiva do time
    try:
        pitch.kdeplot(
            xs, ys,
            ax=ax,
            cmap="inferno",
            fill=True,
            levels=55,
            thresh=0.06,
            alpha=0.62,
            cut=4
        )
    except Exception:
        pitch.hexbin(xs, ys, ax=ax, edgecolors="none", gridsize=(20, 14), cmap="inferno", alpha=0.65)

    # Cores por posição para os marcadores médios
    cores_pos = {
        "G": "#29b6f6",  # Azul Goleiro
        "D": "#66bb6a",  # Verde Defensor
        "M": "#ffa726",  # Laranja Meio
        "F": "#ef5350",  # Vermelho Atacante
    }

    # Plota a posição média de cada titular
    for pos_info in posicoes_medias:
        px = pos_info["media_x"]
        py = pos_info["media_y"]
        nome = pos_info["jogador_nome"].split()
        nome_curto = nome[0] if len(nome) == 1 else f"{nome[0]} {nome[-1]}"
        pos = pos_info.get("posicao", "M")
        cor = cores_pos.get(pos, "#ffffff")

        # Ponto central
        pitch.scatter(
            px, py,
            ax=ax,
            s=220,
            color=cor,
            edgecolors="#ffffff",
            linewidth=1.8,
            zorder=6
        )

        # Rótulo de texto com contorno legível
        txt = ax.text(
            px, py - 3.8,
            f"{nome_curto}\n({pos})",
            ha="center", va="top",
            color="#ffffff",
            fontsize=8.5,
            fontweight="bold",
            zorder=7
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=2.5, foreground="#000000"),
            path_effects.Normal()
        ])

    # Seta de direção de ataque
    pitch.arrows(15, 3.5, 85, 3.5, ax=ax, color="#90a4ae", width=1.5, headwidth=4, headlength=4)
    ax.text(50, 6.0, "Direção de Ataque →", color="#b0bec5", fontsize=9, ha="center", fontweight="bold")

    div_str = f" - {divisao}" if divisao else ""
    ax.set_title(
        f"🛡️ Ocupação Territorial e Posições Médias dos Titulares\n{clube_nome} ({ano}{div_str}) | {len(posicoes_medias)} Atletas Titulares Mapeados",
        color="#f1f5f9",
        fontsize=13,
        fontweight="bold",
        pad=12
    )
    return fig


def calcular_percentis_radar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula percentis (0 a 100) agrupados por [posicao, ano] para os 6 pilares:
    1. Poder de Fogo (gols_por_90)
    2. Criação (assistencias_por_90)
    3. Dribles (dribles_certos por 90)
    4. Combate Defensivo (desarmes + interceptacoes por 90)
    5. Jogo Aéreo (duelos_aereos_ganhos por 90)
    6. Retenção/Qualidade (passes_certos_pct + nota_media_sofascore)
    """
    df_calc = df.copy()
    p90 = np.maximum(df_calc["minutos"].fillna(0) / 90.0, 0.1)

    df_calc["dribles_por_90"] = df_calc["dribles_certos"].fillna(0) / p90
    df_calc["combate_defensivo_por_90"] = (df_calc["desarmes"].fillna(0) + df_calc["interceptacoes"].fillna(0)) / p90
    df_calc["jogo_aereo_por_90"] = df_calc["duelos_aereos_ganhos"].fillna(0) / p90

    # Percentis dentro do grupo (posicao, ano)
    metricas = [
        ("gols_por_90", "pct_poder_fogo"),
        ("assistencias_por_90", "pct_criacao"),
        ("dribles_por_90", "pct_dribles"),
        ("combate_defensivo_por_90", "pct_combate_defensivo"),
        ("jogo_aereo_por_90", "pct_jogo_aereo"),
        ("passes_certos_pct", "pct_passes"),
        ("nota_media_sofascore", "pct_nota"),
    ]

    for col_raw, col_pct in metricas:
        if col_raw in df_calc.columns:
            df_calc[col_pct] = (
                df_calc.groupby(["posicao", "ano"])[col_raw]
                .rank(pct=True)
                .fillna(0.5) * 100.0
            )
        else:
            df_calc[col_pct] = 50.0

    # Pilar Retenção & Qualidade = média de passes e nota
    df_calc["pct_retencao_qualidade"] = (df_calc["pct_passes"] + df_calc["pct_nota"]) / 2.0
    return df_calc


def plotar_radar_comparativo(
    atleta_1: Dict[str, Any],
    atleta_2: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Desenha o Gráfico de Radar (Spider Chart) dos 6 Pilares em Percentis (0-100)
    permitindo comparar Atleta 1 vs. Atleta 2.
    """
    categorias = [
        "Poder de Fogo<br>(Gols/90)",
        "Criação<br>(Assist./90)",
        "Dribles<br>(Dribles/90)",
        "Combate Defensivo<br>(Desarmes+Interc./90)",
        "Jogo Aéreo<br>(Aéreos/90)",
        "Retenção & Qualidade<br>(Passes & Nota)"
    ]

    def extrair_valores(atleta: Dict[str, Any]) -> List[float]:
        vals = [
            float(atleta.get("pct_poder_fogo", 50)),
            float(atleta.get("pct_criacao", 50)),
            float(atleta.get("pct_dribles", 50)),
            float(atleta.get("pct_combate_defensivo", 50)),
            float(atleta.get("pct_jogo_aereo", 50)),
            float(atleta.get("pct_retencao_qualidade", 50)),
        ]
        # Fecha o ciclo do radar
        vals.append(vals[0])
        return vals

    categorias_fechadas = categorias + [categorias[0]]
    valores_1 = extrair_valores(atleta_1)
    nome_1 = f"{atleta_1.get('jogador_nome', 'Atleta 1')} ({atleta_1.get('clube_nome', '')} - {atleta_1.get('ano', '')})"

    fig = go.Figure()

    # Traço Atleta 1
    fig.add_trace(go.Scatterpolar(
        r=valores_1,
        theta=categorias_fechadas,
        fill="toself",
        name=nome_1,
        line=dict(color="#00d2ff", width=2.5),
        fillcolor="rgba(0, 210, 255, 0.35)",
        marker=dict(size=7, color="#00d2ff")
    ))

    # Traço Atleta 2 (se houver)
    if atleta_2:
        valores_2 = extrair_valores(atleta_2)
        nome_2 = f"{atleta_2.get('jogador_nome', 'Atleta 2')} ({atleta_2.get('clube_nome', '')} - {atleta_2.get('ano', '')})"
        fig.add_trace(go.Scatterpolar(
            r=valores_2,
            theta=categorias_fechadas,
            fill="toself",
            name=nome_2,
            line=dict(color="#ff3366", width=2.5),
            fillcolor="rgba(255, 51, 102, 0.35)",
            marker=dict(size=7, color="#ff3366")
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#94a3b8", size=10),
                gridcolor="#334155",
                linecolor="#334155"
            ),
            angularaxis=dict(
                tickfont=dict(color="#f8fafc", size=11, family="Arial, sans-serif"),
                gridcolor="#334155",
                linecolor="#334155"
            ),
            bgcolor="#181d24"
        ),
        paper_bgcolor="#181d24",
        font=dict(color="#f8fafc"),
        title=dict(
            text="🕸️ Radar de Habilidades: Percentis por Posição & Temporada (0-100)",
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#f8fafc")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#e2e8f0")
        ),
        margin=dict(l=50, r=50, t=50, b=60),
        height=480
    )
    return fig

