@echo off
REM Script para executar análise completa de roteirização

echo ========================================
echo   ANALISE DE ROTEIRIZACAO - BRASILIA
echo ========================================
echo.

echo [1/2] Executando algoritmos de roteirizacao...
python run.py
if errorlevel 1 (
    echo ERRO ao executar run.py
    pause
    exit /b 1
)

echo.
echo [2/2] Gerando graficos comparativos adicionais...
python visualizar_comparacao.py
if errorlevel 1 (
    echo ERRO ao executar visualizar_comparacao.py
    pause
    exit /b 1
)

echo.
echo ========================================
echo   ANALISE CONCLUIDA COM SUCESSO!
echo ========================================
echo.
echo Arquivos gerados na pasta 'output/':
echo   - comparacao_algoritmos.png
echo   - rotas_brasilia.png
echo   - rotas_brasilia.html
echo   - grafico_comparacao_barras.png
echo   - analise_detalhada_rotas.png
echo   - RESULTADOS_COMPARACAO.md
echo.
echo Abrindo pasta de resultados...
explorer output
echo.
pause
