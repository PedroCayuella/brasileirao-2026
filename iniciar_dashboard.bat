@echo off
chcp 65001 >nul
title Painel de Inteligência Esportiva - Futebol Brasileiro
cd /d "C:\Users\Pepe\Desktop\Python projetos\Projeto Fut"

echo =======================================================================
echo   ⚽ PAINEL DE INTELIGÊNCIA ESPORTIVA - FUTEBOL BRASILEIRO
echo =======================================================================
echo.
echo [1/2] Iniciando o servidor local...
echo.
echo O navegador será aberto automaticamente em:
echo 🌐 http://127.0.0.1:8501
echo.
echo (Para encerrar o painel no final, basta fechar esta janela)
echo =======================================================================
echo.

:: Abre o navegador automaticamente em 127.0.0.1 após 2 segundos
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:8501"

:: Inicia o Streamlit fixado no IPv4 127.0.0.1
python -m streamlit run app_dashboard.py --server.address 127.0.0.1 --server.port 8501

pause
