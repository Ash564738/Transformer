@echo off
cd /d "%~dp0backend"

python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python train_models.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Training complete. Run "python app.py" (from backend\) to start the Flask API.
