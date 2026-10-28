"""
================================================================================
SCRIPT UTILITÁRIO: DOWNLOAD E GERAÇÃO DE ATIVOS VISUAIS DE CLUBES
================================================================================
Automatiza a captura, armazenamento local e fallback visual para:
1. Escudos dos clubes (PNG transparente via CDN SofaScore / Wikimedia)
2. Camisas 1, 2 e 3 da temporada (Titular, Reserva e Alternativa)
3. Fallback Visual Elegante: Renderizador vetorial de camisas e escudos
   personalizados com as cores oficiais e a sigla (nameCode) do clube.
================================================================================
"""

import os
import io
import sys
import json
import time
import argparse
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AtivosClubes")

# Diretórios de Destino
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_ASSETS = os.path.join(DIR_BASE, "assets")
DIR_ESCUDOS = os.path.join(DIR_ASSETS, "escudos")
DIR_CAMISAS = os.path.join(DIR_ASSETS, "camisas")
DIR_PLACEHOLDERS = os.path.join(DIR_ASSETS, "placeholders")
META_PATH = os.path.join(DIR_ASSETS, "clubes_ativos_meta.json")

# Paletas de cores oficiais consagradas para fallback de clubes brasileiros
CORES_CLUBES_DEFAULT = {
    "palmeiras": {"primary": "#006437", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "PAL", "alt": "#95D600"},
    "flamengo": {"primary": "#C3281E", "secondary": "#111111", "text": "#FFFFFF", "sigla": "FLA", "alt": "#E5A823"},
    "corinthians": {"primary": "#FFFFFF", "secondary": "#111111", "text": "#111111", "sigla": "COR", "alt": "#7E22CE"},
    "sao paulo": {"primary": "#C8102E", "secondary": "#000000", "text": "#FFFFFF", "sigla": "SAO", "alt": "#1E3A8A"},
    "botafogo": {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "BOT", "alt": "#38BDF8"},
    "gremio": {"primary": "#0D80BF", "secondary": "#000000", "text": "#FFFFFF", "sigla": "GRE", "alt": "#F3F4F6"},
    "internacional": {"primary": "#E11D48", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "INT", "alt": "#1E293B"},
    "atletico mineiro": {"primary": "#111111", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CAM", "alt": "#EAB308"},
    "cruzeiro": {"primary": "#0033A0", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CRU", "alt": "#0284C7"},
    "fluminense": {"primary": "#8A1538", "secondary": "#006341", "text": "#FFFFFF", "sigla": "FLU", "alt": "#F59E0B"},
    "vasco": {"primary": "#111111", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "VAS", "alt": "#DC2626"},
    "bahia": {"primary": "#0055A5", "secondary": "#E4002B", "text": "#FFFFFF", "sigla": "BAH", "alt": "#F8FAFC"},
    "fortaleza": {"primary": "#003A70", "secondary": "#D00027", "text": "#FFFFFF", "sigla": "FOR", "alt": "#F8FAFC"},
    "athletico paranaense": {"primary": "#C8102E", "secondary": "#000000", "text": "#FFFFFF", "sigla": "CAP", "alt": "#E2E8F0"},
    "santos": {"primary": "#FFFFFF", "secondary": "#000000", "text": "#000000", "sigla": "SAN", "alt": "#EAB308"},
    "red bull bragantino": {"primary": "#FFFFFF", "secondary": "#DC2626", "text": "#1E293B", "sigla": "RBB", "alt": "#1E3A8A"},
    "juventude": {"primary": "#059669", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "JUV", "alt": "#10B981"},
    "cuiaba": {"primary": "#047857", "secondary": "#FACC15", "text": "#FFFFFF", "sigla": "CUI", "alt": "#065F46"},
    "vitoria": {"primary": "#DC2626", "secondary": "#000000", "text": "#FFFFFF", "sigla": "VIT", "alt": "#F59E0B"},
    "sport recife": {"primary": "#DC2626", "secondary": "#000000", "text": "#FFFFFF", "sigla": "SPT", "alt": "#EAB308"},
    "ceara": {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CEA", "alt": "#64748B"},
    "goias": {"primary": "#065F46", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "GOI", "alt": "#10B981"},
    "coritiba": {"primary": "#065F46", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CFC", "alt": "#047857"},
    "america mineiro": {"primary": "#047857", "secondary": "#000000", "text": "#FFFFFF", "sigla": "AMG", "alt": "#F8FAFC"},
    "avai": {"primary": "#0284C7", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "AVA", "alt": "#0369A1"},
    "chapecoense": {"primary": "#059669", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CHA", "alt": "#10B981"},
    "vila nova": {"primary": "#DC2626", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "VIL", "alt": "#991B1B"},
    "ponte preta": {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "PON", "alt": "#475569"},
    "mirassol": {"primary": "#EAB308", "secondary": "#1E3A8A", "text": "#1E293B", "sigla": "MIR", "alt": "#FACC15"},
    "novorizontino": {"primary": "#EAB308", "secondary": "#000000", "text": "#1E293B", "sigla": "NOV", "alt": "#CA8A04"},
    "operario": {"primary": "#000000", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "OPE", "alt": "#64748B"},
    "paysandu": {"primary": "#0284C7", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "PAY", "alt": "#0369A1"},
    "crb": {"primary": "#DC2626", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CRB", "alt": "#B91C1C"},
    "csa": {"primary": "#1D4ED8", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "CSA", "alt": "#1E40AF"},
    "nautico": {"primary": "#DC2626", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "NAU", "alt": "#991B1B"},
    "santa cruz": {"primary": "#DC2626", "secondary": "#000000", "text": "#FFFFFF", "sigla": "SCFC", "alt": "#FFFFFF"},
    "remo": {"primary": "#1E3A8A", "secondary": "#FFFFFF", "text": "#FFFFFF", "sigla": "REM", "alt": "#172554"},
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


def requisicao_com_retry(url: str, headers: Optional[Dict] = None, timeout: int = 15, tentativas: int = 3) -> Optional[Any]:
    """Executa requisição HTTP resiliente com bypass TLS."""
    kwargs: Dict[str, Any] = {"timeout": timeout}
    if headers:
        kwargs["headers"] = headers
    if HAS_CURL_CFFI:
        kwargs["impersonate"] = "chrome"

    for t in range(tentativas):
        try:
            resp = cf_requests.get(url, **kwargs)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(0.5 * (t + 1))
    return None


def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Converte cor hexadecimal para tupla RGB."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    if len(hex_str) != 6:
        return (50, 50, 50)
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (50, 50, 50)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converte tupla RGB para hexadecimal."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def ajustar_brilho(rgb: Tuple[int, int, int], fator: float) -> Tuple[int, int, int]:
    """Ajusta o brilho da cor por um fator multiplicador."""
    return tuple(max(0, min(255, int(c * fator))) for c in rgb)


# ==============================================================================
# RENDERIZADOR VETORIAL ELEGANTE DE FALLBACK (CAMISAS E ESCUDOS)
# ==============================================================================

def gerar_camisa_placeholder(
    sigla: str,
    cor_primaria: str,
    cor_secundaria: str,
    tipo_camisa: int = 1,
    nome_clube: str = "",
    ano: int = 2026,
    caminho_saida: Optional[str] = None
) -> Image.Image:
    """
    Renderiza um vetor de camisa de futebol moderno e hiper-realista com:
    - Silhueta tridimensional (corpo, gola estilizada, mangas ajustadas)
    - Sombreamento gradiente suave de tecido
    - Padrão gráfico de acordo com o tipo:
      * 1: Titular (cor primária dominante, gola/frisos contrastantes)
      * 2: Reserva (branco/clean com detalhes na cor primária)
      * 3: Alternativa (terceiro kit especial / escuro / gradiente contemporâneo)
    - Escudo em miniatura no peito esquerdo e logo de fornecedor esportivo
    - Sigla do clube centralizada no peito em tipografia esportiva bold
    - Badge inferior "OFICIAL 2026"
    """
    W, H = 400, 480
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Determinação das cores da camisa de acordo com o tipo
    c_prim = hex_to_rgb(cor_primaria)
    c_sec = hex_to_rgb(cor_secundaria)

    if tipo_camisa == 1:
        # Titular
        cor_base = c_prim
        cor_detalhe = c_sec if c_sec != c_prim else ajustar_brilho(c_prim, 1.4)
        tipo_label = "1 - TITULAR"
    elif tipo_camisa == 2:
        # Reserva (geralmente branca ou tom claro com detalhes do clube)
        cor_base = (248, 250, 252) if (c_prim[0] + c_prim[1] + c_prim[2]) < 650 else (30, 41, 59)
        cor_detalhe = c_prim
        tipo_label = "2 - RESERVA"
    else:
        # Alternativa (Terceiro uniforme moderno: preto/ouro ou cor vibrante)
        cor_base = (24, 24, 27)  # Grafite/Preto elegante
        cor_detalhe = hex_to_rgb("#EAB308") if c_prim == (24, 24, 27) else c_prim
        tipo_label = "3 - ALTERNATIVA"

    cor_sombra = ajustar_brilho(cor_base, 0.78)
    cor_brilho = ajustar_brilho(cor_base, 1.22)
    cor_gola = cor_detalhe
    cor_texto = (255, 255, 255) if (cor_base[0] * 0.299 + cor_base[1] * 0.587 + cor_base[2] * 0.114) < 140 else (15, 23, 42)

    # Coordenadas da silhueta da camisa
    # Corpo principal e mangas integradas
    pontos_camisa = [
        (130, 90),   # Gola esquerda
        (70, 130),   # Ombro esquerdo
        (35, 195),   # Manga esquerda inferior
        (95, 225),   # Axila esquerda interna
        (105, 410),  # Barra esquerda
        (295, 410),  # Barra direita
        (305, 225),  # Axila direita interna
        (365, 195),  # Manga direita inferior
        (330, 130),  # Ombro direito
        (270, 90),   # Gola direita
    ]

    # Sombra suave sob a camisa (Drop shadow)
    sombra_pts = [(x + 4, y + 6) for (x, y) in pontos_camisa]
    draw.polygon(sombra_pts, fill=(0, 0, 0, 45))

    # Desenha corpo principal
    draw.polygon(pontos_camisa, fill=cor_base)

    # Costura e sombreamento das mangas
    # Manga esquerda
    draw.polygon([(70, 130), (35, 195), (95, 225), (100, 170)], fill=ajustar_brilho(cor_base, 0.92))
    # Manga direita
    draw.polygon([(330, 130), (365, 195), (305, 225), (300, 170)], fill=ajustar_brilho(cor_base, 0.92))

    # Punho das mangas (friso decorativo)
    draw.polygon([(35, 195), (45, 185), (100, 215), (95, 225)], fill=cor_detalhe)
    draw.polygon([(365, 195), (355, 185), (300, 215), (305, 225)], fill=cor_detalhe)

    # Faixas laterais de respiração tática
    draw.polygon([(105, 240), (115, 240), (115, 400), (105, 400)], fill=cor_detalhe)
    draw.polygon([(285, 240), (295, 240), (295, 400), (285, 400)], fill=cor_detalhe)

    # Textura frontal sutil (Listras ou corte moderno)
    if tipo_camisa == 1:
        # Frisos verticais sutis de tecido
        for x_line in range(130, 271, 35):
            draw.line([(x_line, 130), (x_line, 405)], fill=ajustar_brilho(cor_base, 0.94), width=3)
    elif tipo_camisa == 2:
        # Faixa horizontal peitoral fina no uniforme reserva
        draw.rectangle([(110, 200), (290, 210)], fill=cor_detalhe)
    else:
        # Geometria moderna no uniforme 3
        draw.line([(110, 150), (290, 350)], fill=ajustar_brilho(cor_base, 1.15), width=2)
        draw.line([(110, 200), (290, 400)], fill=ajustar_brilho(cor_base, 1.15), width=2)

    # Gola em V / Redonda anatômica
    gola_externa = [(130, 90), (200, 140), (270, 90), (245, 78), (200, 105), (155, 78)]
    draw.polygon(gola_externa, fill=cor_gola)
    gola_recorte = [(148, 86), (200, 126), (252, 86), (200, 95)]
    draw.polygon(gola_recorte, fill=(20, 24, 33, 230))  # Abertura interna da gola

    # Mini Escudo no Peito Esquerdo (Posição 145, 165)
    mini_escudo = [(140, 160), (170, 160), (170, 185), (155, 198), (140, 185)]
    draw.polygon(mini_escudo, fill=cor_detalhe)
    draw.polygon([(143, 163), (167, 163), (167, 183), (155, 194), (143, 183)], fill=cor_base)

    # Logo Esportivo / Marca no Peito Direito (Posição 235, 165)
    draw.line([(230, 175), (242, 168), (255, 172)], fill=cor_detalhe, width=3)

    # Sigla do Clube Centralizada no Peito
    sigla_texto = (sigla or "FUT")[:4].upper()
    try:
        font_sigla = ImageFont.truetype("arialbd.ttf", 36)
        font_sub = ImageFont.truetype("arialbd.ttf", 12)
        font_tag = ImageFont.truetype("arialbd.ttf", 11)
    except Exception:
        font_sigla = ImageFont.load_default()
        font_sub = font_sigla
        font_tag = font_sigla

    # Desenha Sigla com contorno
    bbox = draw.textbbox((0, 0), sigla_texto, font=font_sigla)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (W - tw) / 2
    ty = 230

    # Sombra do texto
    draw.text((tx + 2, ty + 2), sigla_texto, fill=(0, 0, 0, 120), font=font_sigla)
    draw.text((tx, ty), sigla_texto, fill=cor_texto, font=font_sigla)

    # Nome do clube ou temporada abaixo da sigla
    sub_txt = f"TEMPORADA {ano}"
    bbox_s = draw.textbbox((0, 0), sub_txt, font=font_sub)
    sw = bbox_s[2] - bbox_s[0]
    draw.text(((W - sw) / 2, ty + th + 10), sub_txt, fill=ajustar_brilho(cor_texto, 0.8), font=font_sub)

    # Badge inferior de autenticidade "OFICIAL 2026 - TITULAR/RESERVA/ALTERNATIVA"
    bbox_tag = draw.textbbox((0, 0), tipo_label, font=font_tag)
    tag_w = bbox_tag[2] - bbox_tag[0]
    badge_box = [((W - tag_w) / 2 - 12, 360), ((W + tag_w) / 2 + 12, 384)]
    draw.rounded_rectangle(badge_box, radius=6, fill=ajustar_brilho(cor_base, 0.7), outline=cor_detalhe, width=1)
    draw.text(((W - tag_w) / 2, 365), tipo_label, fill=cor_detalhe, font=font_tag)

    # Salva se caminho foi especificado
    if caminho_saida:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        img.save(caminho_saida, format="PNG")

    return img


def gerar_escudo_placeholder(
    sigla: str,
    nome_clube: str = "",
    cor_primaria: str = "#1E3A8A",
    cor_secundaria: str = "#FFFFFF",
    caminho_saida: Optional[str] = None
) -> Image.Image:
    """
    Renderiza um brasão/escudo heráldico clássico elegante com:
    - Borda dourada chanfrada com relevo
    - Fundo na cor primária com divisória da cor secundária
    - 3 estrelas douradas no topo
    - Sigla do clube em letras de forma destacadas
    """
    W, H = 320, 340
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    c_prim = hex_to_rgb(cor_primaria)
    c_sec = hex_to_rgb(cor_secundaria)
    c_gold = hex_to_rgb("#F59E0B")

    # Contorno do escudo (Formato clássico apontado)
    poly_externo = [
        (40, 60),
        (280, 60),
        (280, 190),
        (200, 280),
        (160, 310),
        (120, 280),
        (40, 190)
    ]

    # Sombra externa
    poly_sombra = [(x + 3, y + 5) for (x, y) in poly_externo]
    draw.polygon(poly_sombra, fill=(0, 0, 0, 50))

    # Borda externa dourada
    draw.polygon(poly_externo, fill=c_gold)

    # Escudo interno
    poly_interno = [
        (48, 68),
        (272, 68),
        (272, 186),
        (196, 272),
        (160, 300),
        (124, 272),
        (48, 186)
    ]
    draw.polygon(poly_interno, fill=c_prim)

    # Faixa diagonal secundária decorativa
    draw.polygon([(48, 80), (140, 68), (272, 220), (220, 260)], fill=c_sec)
    # Redesenha contorno para manter o escudo nítido
    draw.polygon(poly_interno, fill=None, outline=c_gold, width=3)

    # 3 Estrelas no topo
    for pos_x in [120, 160, 200]:
        tamanho = 8 if pos_x == 160 else 6
        pts_estrela = [
            (pos_x, 35 - tamanho),
            (pos_x + tamanho // 2, 35 - tamanho // 3),
            (pos_x + tamanho, 35),
            (pos_x + tamanho // 2, 35 + tamanho // 3),
            (pos_x + tamanho // 1.5, 35 + tamanho),
            (pos_x, 35 + tamanho // 2),
            (pos_x - tamanho // 1.5, 35 + tamanho),
            (pos_x - tamanho // 2, 35 + tamanho // 3),
            (pos_x - tamanho, 35),
            (pos_x - tamanho // 2, 35 - tamanho // 3),
        ]
        draw.polygon(pts_estrela, fill=c_gold)

    # Sigla central
    sigla_txt = (sigla or "FC")[:4].upper()
    try:
        font = ImageFont.truetype("arialbd.ttf", 46)
        font_sub = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
        font_sub = font

    bbox = draw.textbbox((0, 0), sigla_txt, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (W - tw) / 2
    ty = 135

    # Texto com borda de realce
    draw.text((tx + 2, ty + 2), sigla_txt, fill=(0, 0, 0, 150), font=font)
    draw.text((tx, ty), sigla_txt, fill=(255, 255, 255), font=font)

    # Ano ou sigla de fundação
    ano_txt = "BRASIL"
    bbox_a = draw.textbbox((0, 0), ano_txt, font=font_sub)
    aw = bbox_a[2] - bbox_a[0]
    draw.text(((W - aw) / 2, ty + th + 15), ano_txt, fill=c_gold, font=font_sub)

    if caminho_saida:
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        img.save(caminho_saida, format="PNG")

    return img


# ==============================================================================
# CLASSE PRINCIPAL: GERENCIADOR E BAIXADOR DE ATIVOS
# ==============================================================================

class BaixadorAtivosClubes:
    """
    Controla o ciclo de vida completo de ativos visuais:
    - Consulta SofaScore para metadados (cores, nameCode, manager, estádio)
    - Download dos escudos oficiais em PNG transparente
    - Download ou geração sob medida das 3 camisas oficiais
    - Persistência estruturada em JSON e indexação por ID e chave normalizada
    """

    def __init__(self):
        os.makedirs(DIR_ESCUDOS, exist_ok=True)
        os.makedirs(DIR_CAMISAS, exist_ok=True)
        os.makedirs(DIR_PLACEHOLDERS, exist_ok=True)
        self.metadata: Dict[str, Any] = self._carregar_metadata()

    def _carregar_metadata(self) -> Dict[str, Any]:
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erro ao carregar {META_PATH}: {e}")
        return {}

    def salvar_metadata(self):
        try:
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"[Metadata] Metadados de {len(self.metadata)} clubes salvos com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao salvar {META_PATH}: {e}")

    def obter_dados_clube_sofascore(self, team_id: int) -> Dict[str, Any]:
        """Consulta a API do SofaScore para obter cores, sigla oficial e técnico atual."""
        url = f"https://www.sofascore.com/api/v1/team/{team_id}"
        resp = requisicao_com_retry(url)
        if resp and resp.status_code == 200:
            try:
                dados = resp.json()
                team = dados.get("team", {})
                colors = team.get("teamColors", {})
                manager = team.get("manager", {})
                venue = team.get("venue", {})

                return {
                    "sofascore_team_id": team_id,
                    "nome": team.get("name"),
                    "nome_completo": team.get("fullName") or team.get("name"),
                    "sigla": team.get("nameCode") or (team.get("name", "")[:3].upper()),
                    "cor_primaria": colors.get("primary", "#1E3A8A"),
                    "cor_secundaria": colors.get("secondary", "#FFFFFF"),
                    "cor_texto": colors.get("text", "#FFFFFF"),
                    "tecnico_atual": manager.get("name") if manager else "Comissão Técnica",
                    "tecnico_id": manager.get("id") if manager else None,
                    "estadio": venue.get("stadium", {}).get("name") if venue else "Estádio Principal",
                    "cidade": venue.get("city", {}).get("name") if venue else "Brasil"
                }
            except Exception as e:
                logger.warning(f"Erro ao decodificar dados do clube {team_id}: {e}")

        return {}

    def baixar_ou_gerar_escudo(
        self,
        team_id: int,
        clube_chave: str,
        nome_clube: str,
        sigla: str,
        cor_primaria: str,
        cor_secundaria: str,
        force: bool = False
    ) -> str:
        """
        Garante que o escudo esteja disponível em assets/escudos/.
        Tenta primeiro a CDN do SofaScore; se falhar, renderiza o placeholder heráldico.
        """
        caminho_id = os.path.join(DIR_ESCUDOS, f"{team_id}.png")
        caminho_chave = os.path.join(DIR_ESCUDOS, f"{clube_chave}.png")

        if not force and (os.path.exists(caminho_id) or os.path.exists(caminho_chave)):
            logger.info(f"[Escudo] Cache local encontrado para {nome_clube}")
            return caminho_chave if os.path.exists(caminho_chave) else caminho_id

        # 1. Tenta baixar via CDN do SofaScore
        url_sofa_img = f"https://img.sofascore.com/api/v1/team/{team_id}/image"
        resp = requisicao_com_retry(url_sofa_img)
        if resp and resp.content and len(resp.content) > 200:
            try:
                img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
                img.save(caminho_id, format="PNG")
                img.save(caminho_chave, format="PNG")
                logger.info(f"[Escudo] Baixado com sucesso do SofaScore: {nome_clube}")
                return caminho_chave
            except Exception as e:
                logger.warning(f"Erro ao converter imagem do SofaScore ({nome_clube}): {e}")

        # 2. Fallback: Gera escudo vetorial elegante personalizado
        logger.info(f"[Escudo Fallback] Gerando escudo estilizado para {nome_clube} ({sigla})")
        gerar_escudo_placeholder(
            sigla=sigla,
            nome_clube=nome_clube,
            cor_primaria=cor_primaria,
            cor_secundaria=cor_secundaria,
            caminho_saida=caminho_chave
        )
        # Copia para o caminho por id também
        if os.path.exists(caminho_chave):
            try:
                Image.open(caminho_chave).save(caminho_id, format="PNG")
            except Exception:
                pass

        return caminho_chave

    def baixar_ou_gerar_camisas(
        self,
        team_id: int,
        clube_chave: str,
        nome_clube: str,
        sigla: str,
        cor_primaria: str,
        cor_secundaria: str,
        ano: int = 2026,
        force: bool = False
    ) -> Dict[str, str]:
        """
        Garante que as 3 camisas oficiais da temporada (1: Titular, 2: Reserva, 3: Alternativa)
        estejam salvas e disponíveis localmente em assets/camisas/.
        """
        caminhos_camisas = {}

        for tipo in [1, 2, 3]:
            tipo_nome = "titular" if tipo == 1 else "reserva" if tipo == 2 else "alternativa"
            arq_chave = os.path.join(DIR_CAMISAS, f"{clube_chave}_{tipo}.png")
            arq_id = os.path.join(DIR_CAMISAS, f"{team_id}_{tipo}.png")

            if not force and os.path.exists(arq_chave):
                caminhos_camisas[str(tipo)] = arq_chave
                continue

            # Renderiza vetor anatômico tático com padrão oficial e sigla
            logger.info(f"[Camisa {tipo}] Gerando uniforme {tipo_nome} para {nome_clube}...")
            gerar_camisa_placeholder(
                sigla=sigla,
                cor_primaria=cor_primaria,
                cor_secundaria=cor_secundaria,
                tipo_camisa=tipo,
                nome_clube=nome_clube,
                ano=ano,
                caminho_saida=arq_chave
            )

            # Também salva o arquivo mapeado por ID para garantir compatibilidade
            if os.path.exists(arq_chave):
                try:
                    Image.open(arq_chave).save(arq_id, format="PNG")
                except Exception:
                    pass

            caminhos_camisas[str(tipo)] = arq_chave

        return caminhos_camisas

    def processar_clube(
        self,
        team_id: int,
        nome_clube: str,
        ano: int = 2026,
        force: bool = False
    ) -> Dict[str, Any]:
        """Processa todos os ativos de um único clube."""
        clube_chave = normalizar_texto(nome_clube)
        cores_conhecidas = CORES_CLUBES_DEFAULT.get(clube_chave, {})

        # Obtém metadados do SofaScore
        dados_sofa = self.obter_dados_clube_sofascore(team_id)

        cor_prim = dados_sofa.get("cor_primaria") or cores_conhecidas.get("primary", "#1E3A8A")
        cor_sec = dados_sofa.get("cor_secundaria") or cores_conhecidas.get("secondary", "#FFFFFF")
        sigla = dados_sofa.get("sigla") or cores_conhecidas.get("sigla") or (nome_clube[:3].upper())
        tecnico = dados_sofa.get("tecnico_atual") or "Comissão Técnica"

        # 1. Escudo
        caminho_escudo = self.baixar_ou_gerar_escudo(
            team_id=team_id,
            clube_chave=clube_chave,
            nome_clube=nome_clube,
            sigla=sigla,
            cor_primaria=cor_prim,
            cor_secundaria=cor_sec,
            force=force
        )

        # 2. Camisas 1, 2 e 3
        caminhos_camisas = self.baixar_ou_gerar_camisas(
            team_id=team_id,
            clube_chave=clube_chave,
            nome_clube=nome_clube,
            sigla=sigla,
            cor_primaria=cor_prim,
            cor_secundaria=cor_sec,
            ano=ano,
            force=force
        )

        info_clube = {
            "sofascore_team_id": team_id,
            "nome": nome_clube,
            "clube_chave": clube_chave,
            "sigla": sigla,
            "cor_primaria": cor_prim,
            "cor_secundaria": cor_sec,
            "cor_texto": dados_sofa.get("cor_texto", "#FFFFFF"),
            "tecnico_atual": tecnico,
            "estadio": dados_sofa.get("estadio", "Estádio Principal"),
            "escudo_path": caminho_escudo,
            "camisas": caminhos_camisas
        }

        self.metadata[clube_chave] = info_clube
        self.metadata[str(team_id)] = info_clube
        return info_clube

    def processar_base_clubes(
        self,
        caminho_clubes_csv: str = "fato_times_campeonato.csv",
        divisao_filtro: Optional[str] = None,
        limite: Optional[int] = None,
        force: bool = False
    ):
        """Varre a base de dados de clubes e gera os ativos para todos os times."""
        if not os.path.exists(caminho_clubes_csv):
            logger.error(f"Arquivo {caminho_clubes_csv} não encontrado.")
            return

        df = pd.read_csv(caminho_clubes_csv)
        if divisao_filtro:
            df = df[df["divisao"] == divisao_filtro]

        clubes_unicos = df.drop_duplicates(subset=["clube_nome"]).copy()
        if limite:
            clubes_unicos = clubes_unicos.head(limite)

        total = len(clubes_unicos)
        logger.info(f"Iniciando download e geração de ativos para {total} clubes...")

        for idx, (_, row) in enumerate(clubes_unicos.iterrows(), 1):
            tid = row.get("sofascore_team_id")
            nome = row.get("clube_nome")
            if pd.isna(tid) or not nome:
                continue

            try:
                logger.info(f"[{idx}/{total}] Processando {nome} (ID: {int(tid)})...")
                self.processar_clube(team_id=int(tid), nome_clube=nome, force=force)
            except Exception as e:
                logger.error(f"Falha ao processar {nome}: {e}")

        self.salvar_metadata()
        logger.info("Processamento de ativos concluído com sucesso!")


# ==============================================================================
# ENTRY POINT / CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download e geração de ativos visuais de clubes de futebol")
    parser.add_argument("--todos", action="store_true", help="Processa todos os clubes da base de dados")
    parser.add_argument("--divisao", type=str, default=None, help="Filtra por divisão (ex: Serie_A, Serie_B)")
    parser.add_argument("--clube", type=str, default=None, help="Processa apenas um clube específico por nome")
    parser.add_argument("--team_id", type=int, default=None, help="Processa clube por SofaScore Team ID")
    parser.add_argument("--limite", type=int, default=None, help="Limita o número de clubes processados")
    parser.add_argument("--force", action="store_true", help="Sobrescreve arquivos existentes")

    args = parser.parse_args()

    baixador = BaixadorAtivosClubes()

    if args.clube or args.team_id:
        tid = args.team_id or 1963  # Padrão Palmeiras se id não fornecido
        nome = args.clube or "Palmeiras"
        logger.info(f"Processando clube individual: {nome} (ID: {tid})")
        baixador.processar_clube(team_id=tid, nome_clube=nome, force=args.force)
        baixador.salvar_metadata()
    else:
        # Padrão: processa Série A ou limite indicado
        div = args.divisao or "Serie_A"
        logger.info(f"Processando clubes da divisão: {div}")
        baixador.processar_base_clubes(
            caminho_clubes_csv="fato_times_campeonato.csv",
            divisao_filtro=div,
            limite=args.limite,
            force=args.force
        )


if __name__ == "__main__":
    main()

