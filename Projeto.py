"""
================================================================================
PROJETO: INTELIGÊNCIA E ENGENHARIA DE DADOS NO FUTEBOL BRASILEIRO (SÉRIES A A D)
================================================================================
Arquivo Principal (Orquestrador Central)
Permite executar:
1. Pipeline de ETL (Coleta Sofascore + Transfermarkt + Entity Resolution RapidFuzz)
2. Modelagem Preditiva (Aging Curves, KNN Statistical Twins, Previsão XGBoost/Ridge)
3. Dashboard Interativo (Streamlit + Plotly Express)
4. Relatório Executivo e Resumos Analíticos
================================================================================
"""

import os
import sys
import argparse
import subprocess
import sqlite3
import pandas as pd

# Garante suporte a UTF-8 no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona o diretório src ao path para imports modulares
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from importlib import import_module


def exibir_banner():
    banner = """
    ======================================================================
    ⚽ SISTEMA DE INTELIGÊNCIA ESPORTIVA - FUTEBOL BRASILEIRO (SÉRIES A A D)
    ======================================================================
    [1] Executar Pipeline ETL de Dados (Sofascore + Transfermarkt + Fuzzy ETL)
    [2] Executar Modelagem Preditiva (Aging Curves, Joias Sub-23, Modelo XGBoost)
    [3] Iniciar Dashboard Interativo Streamlit (app_dashboard.py)
    [4] Exibir Resumo Executivo das Tabelas e Métricas
    [5] Executar Pipeline Completo de Ponta a Ponta
    [6] Testar Visualização Tática (Heatmap, Shotmap e Ocupação Coletiva)
    [0] Sair
    ======================================================================
    """
    print(banner)


def executar_etl(anos=[2023, 2024, 2025, 2026]):
    print(f"\n[+] Iniciando Pipeline ETL para a(s) temporada(s): {anos}...")
    etl_module = import_module("03_unificacao_fuzzy_etl")
    df_jog, df_times = etl_module.executar_pipeline_etl(anos=anos)
    print(f"\n[✓] ETL Concluído com sucesso! {len(df_jog)} jogadores e {len(df_times)} clubes salvos.")
    return df_jog, df_times


def executar_modelagem():
    print("\n[+] Iniciando Modelagem Preditiva e Inteligência Esportiva...")
    modelo_module = import_module("04_modelo_preditivo")
    ranking_joias, df_clubes_estudo = modelo_module.executar_modelagem_preditiva()
    print(f"\n[✓] Modelagem Preditiva concluída! Ranking de joias e estudo de clubes gerados em relatorios/.")
    return ranking_joias, df_clubes_estudo


def iniciar_dashboard():
    print("\n[+] Iniciando Dashboard Streamlit...")
    comando = [sys.executable, "-m", "streamlit", "run", "app_dashboard.py"]
    try:
        subprocess.run(comando)
    except KeyboardInterrupt:
        print("\n[!] Dashboard encerrado.")


def exibir_resumo():
    caminho_db = "banco_dados/futebol_br.db"
    if not os.path.exists(caminho_db):
        caminho_db = "futebol_br.db"

    if not os.path.exists(caminho_db):
        print("\n[!] Banco de dados não encontrado. Execute o ETL primeiro (Opção 1).")
        return

    conn = sqlite3.connect(caminho_db)
    df_times = pd.read_sql_query("SELECT * FROM fato_times_campeonato", conn)
    df_jog = pd.read_sql_query("SELECT * FROM fato_jogadores_temporada", conn)
    conn.close()

    print("\n" + "="*70)
    print("📊 RESUMO GERAL DO BANCO DE DADOS MULTITEMPORAL (futebol_br.db)")
    print("="*70)
    print(f"Total de Clubes: {len(df_times)} distribuídos em {df_times['divisao'].nunique()} divisões")
    print(f"Temporadas Cobertas: {sorted(df_times['ano'].unique().tolist())}")
    print("\nDistribuição de Clubes por Ano e Divisão:")
    print(df_times.groupby(['ano', 'divisao']).size().unstack(fill_value=0).to_string())

    print(f"\nTotal de Registros de Atletas: {len(df_jog)}")
    print(f"Jogadores Únicos: {df_jog['sofascore_player_id'].nunique()}")
    print("Distribuição de Atletas por Ano:")
    print(df_jog['ano'].value_counts().sort_index().to_string())

    print(f"\nIdade Média Geral: {df_jog['idade_na_temporada'].mean():.1f} anos")
    print(f"Média de Minutos: {df_jog['minutos'].mean():.0f} min")
    print(f"Valor de Mercado Agregado: € {df_jog['valor_mercado_eur'].sum():,.2f}")

    if os.path.exists("ranking_joias_escondidas_A_ate_D.csv"):
        df_j = pd.read_csv("ranking_joias_escondidas_A_ate_D.csv")
        print("\n💎 TOP 5 JOIAS COM MAIOR POTENCIAL DE EXPLOSÃO (2026+):")
        cols_j = ['jogador_nome', 'clube_nome', 'divisao', 'ano', 'posicao', 'idade_na_temporada', 'delta_evolucao_nota', 'score_explosao_proximo_ciclo']
        cols_j_disp = [c for c in cols_j if c in df_j.columns]
        print(df_j[cols_j_disp].head(5).to_string())

    if os.path.exists("estudo_desempenho_clubes.csv"):
        df_c = pd.read_csv("estudo_desempenho_clubes.csv")
        print("\n🏆 TOP 5 SUPERAÇÕES TÁTICAS HISTÓRICAS (Overperforming):")
        print(df_c[['clube_nome', 'divisao', 'ano', 'pontos', 'pontos_esperados_elenco', 'residuo_desempenho']].sort_values('residuo_desempenho', ascending=False).head(5).to_string())
    print("="*70 + "\n")


def testar_tatica():
    print("\n[+] Testando Visualização Tática Sofascore (Heatmap, Shotmap e Ocupação Coletiva)...")
    tatica_module = import_module("05_visualizacao_tatica")
    tid, sid = tatica_module.obter_tournament_e_season_id("Serie_A", 2024)
    print(f"Torneio Série A: {tid} | Temporada 2024: {sid}")
    # Testa busca de um jogador conhecido (Hulk - 34705)
    pts = tatica_module.buscar_heatmap_jogador(34705, tid, sid)
    print(f"[✓] Heatmap baixado/carregado do cache com sucesso: {len(pts)} coordenadas.")
    shots = tatica_module.buscar_shotmap_jogador(34705, tid, sid)
    print(f"[✓] Shotmap baixado/carregado do cache com sucesso: {len(shots)} finalizações registradas.")
    print("[✓] Módulo tático operacional e integrado ao app_dashboard.py!")


def main():
    parser = argparse.ArgumentParser(description="Sistema de Inteligência Esportiva - Futebol Brasileiro")
    parser.add_argument("--etl", action="store_true", help="Executa o pipeline ETL")
    parser.add_argument("--modelo", action="store_true", help="Executa a modelagem preditiva")
    parser.add_argument("--dashboard", action="store_true", help="Inicia o dashboard interativo")
    parser.add_argument("--resumo", action="store_true", help="Exibe o resumo do banco de dados")
    parser.add_argument("--tatica", action="store_true", help="Testa o módulo visual tático")
    parser.add_argument("--all", action="store_true", help="Executa ETL + Modelagem de ponta a ponta")
    parser.add_argument("--anos", nargs="+", type=int, default=[2023, 2024, 2025, 2026], help="Anos das temporadas a coletar")

    args = parser.parse_args()

    # Modo argumentos de linha de comando
    if args.all:
        executar_etl(anos=args.anos)
        executar_modelagem()
        exibir_resumo()
        return
    if args.etl:
        executar_etl(anos=args.anos)
        return
    if args.modelo:
        executar_modelagem()
        return
    if args.dashboard:
        iniciar_dashboard()
        return
    if args.resumo:
        exibir_resumo()
        return
    if args.tatica:
        testar_tatica()
        return

    # Modo interativo no terminal
    while True:
        exibir_banner()
        opcao = input("Digite a opção desejada [0-6]: ").strip()
        if opcao == "1":
            anos_input = input("Digite o(s) ano(s) separados por espaço [padrão: 2023 2024 2025 2026]: ").strip()
            anos = [int(a) for a in anos_input.split()] if anos_input else [2023, 2024, 2025, 2026]
            executar_etl(anos=anos)
        elif opcao == "2":
            executar_modelagem()
        elif opcao == "3":
            iniciar_dashboard()
        elif opcao == "4":
            exibir_resumo()
        elif opcao == "5":
            executar_etl(anos=[2023, 2024, 2025, 2026])
            executar_modelagem()
            exibir_resumo()
        elif opcao == "6":
            testar_tatica()
        elif opcao == "0":
            print("\nEncerrando. Até logo!")
            break
        else:
            print("\n[!] Opção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
