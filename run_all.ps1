param(
    [switch]$StartApi
)

$ErrorActionPreference = 'Stop'
Write-Host 'Installing Python dependencies...'
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Dependency installation failed.'
    exit $LASTEXITCODE
}

Write-Host 'Training transformer degradation models...'
python train_transformer_degradation.py
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Model training failed.'
    exit $LASTEXITCODE
}

if ($StartApi) {
    Write-Host 'Starting Flask API at http://localhost:5000'
    python api.py
} else {
    Write-Host 'Training complete. Run "python api.py" to start the Flask API.'
}
