@echo off
chcp 65001 > nul
echo.
echo ================================================
echo   SAR Thickness Predictor - Streamlit App
echo ================================================
echo.
echo Starting Streamlit app with Python 3.11...
echo The browser will open automatically at:
echo   http://localhost:8501
echo.
"C:\Users\SAMSUNG\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run streamlit_app.py
pause
