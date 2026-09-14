@echo off
chcp 65001 > nul
echo.
echo ================================================
echo   SAR Thickness Predictor - Gradio App
echo ================================================
echo.
echo Starting app with Python 3.11...
echo The browser will open automatically at:
echo   http://localhost:7860
echo.
set PYTHONIOENCODING=utf-8
py -3.11 app.py
if %ERRORLEVEL% NEQ 0 (
    "C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\python.exe" app.py
)
pause
