# HeatGuard AI

**10-Day Wet-Bulb Heat Risk Forecasting & Action Intelligence**

HeatGuard AI is a deployable climate-intelligence dashboard built for the **IIIT Lucknow Climate Intelligence Challenge 2026**. It forecasts future wet-bulb temperature, classifies heat-risk levels, explains the model drivers, and generates audience-specific action plans for citizens, hospitals, schools, outdoor workers, and local authorities.

## Problem Statement

Extreme heat becomes especially dangerous when high air temperature combines with high humidity. Wet-bulb temperature captures that combined physiological stress better than temperature alone. HeatGuard AI predicts wet-bulb heat risk up to 10 days ahead and turns model output into practical decision support.

## Why Wet-Bulb Temperature Matters

Wet-bulb temperature estimates the lowest temperature achievable through evaporative cooling. When wet-bulb values rise, the human body has a harder time cooling itself through sweat. High wet-bulb conditions can raise the risk of heat exhaustion, heat stroke, hospital surges, school disruption, and unsafe outdoor work.

## Features

- Automatic CSV discovery from `data/`
- Synthetic demo dataset fallback when no real Kaggle data is present
- Automatic target, date, and station/location column detection
- Time features, cyclical calendar features, lag features, and rolling climate features
- Time-aware validation when a date column is available
- Candidate model training with Random Forest, HistGradientBoosting, and Extra Trees
- Optional XGBoost and LightGBM support if installed locally
- Best model selection by validation RMSE
- 10-day direct lead-time forecasting with risk categories
- Plotly forecast chart, risk gauge, historical trend, feature importance, and risk distribution
- Gemini-powered heat advisory via `GEMINI_API_KEY`
- Strong rule-based advisory fallback when Gemini is not configured
- Streamlit dark climate-tech dashboard
- Hugging Face Spaces compatible layout

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn
- Plotly
- joblib
- google-generativeai
- python-dotenv

## Folder Structure

```text
HeatGuard-AI/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .env.example
|
|-- data/
|   |-- README.md
|   |-- .gitkeep
|
|-- models/
|   |-- .gitkeep
|
|-- assets/
|   |-- heatguard_logo.svg
|   |-- climate_hero.svg
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- model.py
|   |-- train.py
|   |-- predict.py
|   |-- explainability.py
|   |-- advisory.py
|   |-- utils.py
|
|-- notebooks/
|   |-- experimentation_template.ipynb
|
|-- scripts/
    |-- train_model.py
    |-- make_submission.py
```

## Local Setup

```bash
python -m venv venv
source venv/bin/activate  # for Mac/Linux
venv\Scripts\activate     # for Windows
pip install -r requirements.txt
streamlit run app.py
```

## Add the Kaggle Dataset

Competition page:

https://www.kaggle.com/competitions/aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026

1. Download the challenge dataset from Kaggle.
2. Place CSV files in the `data/` folder.
3. If the main file is named `train.csv`, HeatGuard AI uses it automatically.
4. If multiple CSV files exist, select the desired file in the dashboard.
5. If the target column is not auto-detected, select it in the app or pass it to the training script.

The app does not use external datasets by default. When no real CSV is present, it uses a small synthetic demo dataset only for UI testing and clearly labels it as demo fallback data.

Optional Kaggle CLI download:

```bash
pip install kaggle
kaggle competitions download -c aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026 -p data
```

After download, unzip the archive inside `data/`, then run training again.

## Train the Model

```bash
python scripts/train_model.py
```

Optional arguments:

```bash
python scripts/train_model.py --csv data/train.csv --target-col wet_bulb_c --date-col date --location-col station --horizon 10
```

Training outputs:

- `models/heatguard_model.joblib`
- `models/metrics.json`
- `models/features.json`

## Run the Dashboard Locally

```bash
streamlit run app.py
```

The dashboard is now competition-demo focused. It automatically detects the Kaggle folder under `data/`, loads `models/heatguard_model.joblib`, lets judges select a Kaggle location/day index, and shows the 10-day wet-bulb forecast, risk gauge, explainability, advisory, validation, and submission readiness.

## Final Competition Workflow

1. Place Kaggle files under `data/aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026/`.
2. Place trained artifacts under `models/`:
   - `heatguard_model.joblib`
   - `metrics.json`
   - `features.json`
3. Generate Kaggle submission:

```bash
python scripts/make_submission.py --data-dir data/aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026 --output submission.csv --horizon 10 --target-col WBT
```

4. Run the dashboard:

```bash
streamlit run app.py
```

## Gemini API Configuration

Create `.env` from `.env.example` and add:

```bash
GEMINI_API_KEY=your_key_here
```

The app loads the key from the environment. If no key is present, HeatGuard AI uses a rule-based advisory system instead of failing.

## Hugging Face Spaces Deployment

1. Create new Hugging Face Space
2. Select Streamlit SDK
3. Connect GitHub repo or upload files
4. Add GEMINI_API_KEY in Space Secrets
5. Deploy

Notes:

- Keep `app.py` at the repository root.
- Keep `requirements.txt` at the repository root.
- Add Kaggle data only if competition rules allow redistribution.
- Otherwise, upload the dataset manually in the dashboard during the demo.

## GitHub Upload

```bash
git init
git add .
git commit -m "Initial commit: HeatGuard AI climate intelligence platform"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Model and Validation Approach

HeatGuard AI creates a supervised direct forecasting dataset by shifting the wet-bulb target into the future for the selected horizon. Lag and rolling features are computed from current and previous observations only, preventing future target leakage.

Validation:

- Uses chronological 80/20 split when a date column exists
- Uses random train/test split only when no date column exists
- Reports MAE, RMSE, and R2
- Selects the best candidate model by RMSE

Wet-bulb risk thresholds:

- Low: below 24 C
- Moderate: 24 to 27 C
- High: 27 to 30 C
- Extreme: 30 C and above

Thresholds are centralized in `src/config.py` and can be adjusted for domain guidance.

## Demo Screenshots

Add screenshots here after running the dashboard:

- Hero and KPI overview
- 10-day forecast chart
- Advisory section
- Model validation section

## Future Scope

- Add calibrated prediction intervals
- Add geospatial risk maps when station latitude and longitude are available
- Add SHAP explanations for tree models
- Add batch forecast export for all stations
- Add scenario simulation for humidity and temperature stress testing
- Integrate official local heat action plan documents if allowed by the challenge

## Team

Team name:

Members:

Contact:

## Responsible Use

HeatGuard AI is decision-support software. It should not be represented as an official government alerting system unless integrated, validated, and approved by the relevant authorities.
