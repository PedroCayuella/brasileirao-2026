"""
================================================================================
ENTRYPOINT PRINCIPAL DO DASHBOARD STREAMLIT (app.py)
================================================================================
Ponto de entrada padrão para execução do painel interativo via:
    streamlit run app.py
ou:
    streamlit run app_dashboard.py
================================================================================
"""

import os
import runpy

if __name__ == "__main__" or True:
    caminho_dashboard = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_dashboard.py")
    runpy.run_path(caminho_dashboard, run_name="__main__")

