# Transformer Degradation Ranking

A transformer degradation ranking framework built on DGA data.

This repository includes:

- data cleaning and feature engineering for dissolved gas analysis (DGA)
- severity scoring and class labels for transformer health
- LightGBM ranking, XGBoost regression, and PyTorch temporal modeling
- Flask dashboard and Streamlit app with CSV/XLSX upload support, transformer-level ranking, preview/validation panels, explainability, and a result-aware chatbot

## Prerequisites

- Python 3.10+ recommended
- `git` and command-line access to the repository root
- `dataset/DGA of Main Tank only KT 11022026_09062026.xlsx` must be present

## Install dependencies

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train models

Run training from the repository root:

```powershell
python train_transformer_degradation.py
```

This will:

- load the Excel dataset from `dataset/DGA of Main Tank only KT 11022026_09062026.xlsx`
- clean and impute missing DGA values
- build severity scores and target classes
- train a LightGBM ranking model, an XGBoost regression model, and a PyTorch temporal model
- save trained artifacts to `models/`

## Start the Flask dashboard

Run the product dashboard:

```powershell
.\.venv\Scripts\python.exe api.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Start the Streamlit app

Run the Streamlit app:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Run helper scripts

Windows PowerShell:

```powershell
.
un_all.ps1
```

Start the API after training:

```powershell
.
un_all.ps1 -StartApi
```

Windows Command Prompt:

```cmd
run_all.bat
```

macOS/Linux:

```bash
./run_all.sh
```

## Prediction output

The app uses the same inference service behind the scenes and returns a JSON-style payload with:

- `predictions`
- `transformer_summary`
- `top_risk_transformers`
- `severity_distribution`
- `fault_distribution`
- `dataset_summary`
- `transformer_timeseries`
- `chat_context_payload`
- `preview_rows`
- `rows`
- `total_rows`

## Saved artifacts

After training, the `models/` folder contains:

- `lightgbm_ranker_model.joblib`
- `xgboost_regressor_model.joblib`
- `temporal_model_state_dict.pt`
- `temporal_scaler.joblib`
- `validation_predictions.csv`

## Notes

- `api.py` uses Flask development server for local testing.
- The training pipeline may skip Cox survival model fitting if the dataset is ill-conditioned or convergence fails.
- For production, deploy with a WSGI server such as Gunicorn or Waitress.
- Ensure the dataset file exists before running training.

## Recommended checks

Verify the model artifacts after training:

```powershell
Get-ChildItem models\
```

Confirm the service is live:

```bash
curl http://127.0.0.1:5000/health
```
