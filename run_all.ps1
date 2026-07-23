param(
    [switch]$StartApi
)

$ErrorActionPreference = 'Stop'
Set-Location -Path (Join-Path $PSScriptRoot 'backend')

Write-Host 'Installing Python dependencies...'
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Dependency installation failed.'
    exit $LASTEXITCODE
}

Write-Host 'Training transformer degradation models...'
python train_models.py
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Model training failed.'
    exit $LASTEXITCODE
}

if ($StartApi) {
    Write-Host 'Starting Flask API at http://localhost:5000'
    python app.py
} else {
    Write-Host 'Training complete. Run "python app.py" (from backend/) to start the Flask API.'
}
