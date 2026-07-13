@echo off
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

python train_transformer_degradation.py
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

echo Training complete. Run python api.py to start the Flask API.
