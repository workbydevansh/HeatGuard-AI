# HeatGuard AI

**10-Day Wet-Bulb Heat Risk Forecasting and Action Intelligence**

HeatGuard AI is a climate-intelligence system built for the **IIIT Lucknow Climate Intelligence Challenge 2026**. It forecasts wet-bulb temperature up to 10 days ahead, classifies public-health heat risk, and converts model output into practical action guidance for citizens, schools, hospitals, outdoor workers, and local authorities.

## Live Links

- Live demo: https://heat-guard-ai.vercel.app
- GitHub repository: https://github.com/workbydevansh/HeatGuard-AI
- Kaggle public leaderboard score: **0.04347**
- Public leaderboard rank at submission time: **2**

The final Kaggle rank may differ because the public leaderboard uses about 30% of the test data and the final evaluation uses the hidden 70% split.

## Problem

Extreme heat becomes more dangerous when high temperature combines with high humidity. Wet-bulb temperature captures this combined physiological stress better than air temperature alone because it reflects the body's ability to cool through evaporation.

HeatGuard AI answers three operational questions:

- How risky will wet-bulb heat become over the next 10 days?
- Which days require the most attention?
- What should different audiences do before the peak risk arrives?

## What The Demo Does

The deployed Vercel demo is an interactive browser dashboard. Judges can adjust climate parameters such as air temperature, humidity, rainfall, wind speed, cloud cover, and heat trend. The app recalculates:

- 10-day wet-bulb temperature forecast
- Peak risk day and risk category
- Forecast curve with risk bands
- Day-wise forecast table
- Audience-specific action advisory
- Downloadable forecast CSV

The browser demo uses a lightweight client-side polynomial WBT estimator exported from the training workflow so it can run instantly without exposing Kaggle data or large model artifacts.

## Kaggle Approach

For the competition submission, the final selected Kaggle entries were:

- `submission_blend_poly95_extra5_offset1.zip` with public score **0.04347**
- `submission_poly3_offset1.zip` with public score **0.04421**

The best public submission blends a polynomial wet-bulb estimator with an ExtraTrees model and uses next-day aligned target columns for `target_day_1` through `target_day_10`.

## Modeling

HeatGuard AI uses weather and location variables from the challenge dataset, including:

- Air temperature, min/max temperature, and soil/surface temperature
- Relative and specific humidity
- Wind speed and direction
- Surface pressure
- Rainfall, cloud cover, and solar radiation
- Relative latitude and longitude

The Streamlit workflow in `app.py` supports trained model artifacts, feature engineering, risk visualization, explainability, and action advisory generation. The Vercel demo uses a compact browser-side estimator for live interaction.

## Risk Bands

| Wet-bulb temperature | Risk level |
| --- | --- |
| Below 24 C | Low |
| 24 C to 27 C | Moderate |
| 27 C to 30 C | High |
| 30 C and above | Extreme |

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- scikit-learn
- Plotly
- Vercel static deployment
- Browser-side JavaScript predictor

## Repository Structure

```text
HeatGuard-AI/
|-- index.html                  # Live Vercel dashboard
|-- app.py                      # Streamlit dashboard
|-- requirements.txt
|-- assets/
|   |-- browser_wbt_model.json  # Lightweight browser-side WBT model
|   |-- heatguard_logo.svg
|   |-- climate_hero.svg
|-- src/
|   |-- feature_engineering.py
|   |-- model.py
|   |-- train.py
|   |-- predict.py
|   |-- advisory.py
|   |-- explainability.py
|-- scripts/
|   |-- train_model.py
|   |-- make_submission.py
|-- deliverables/
|   |-- HeatGuard_AI_Judge_Ready_Deck.pptx
|-- data/
|   |-- README.md
|-- models/
|   |-- .gitkeep
```

Large Kaggle CSV files, trained `.joblib` artifacts, `.env`, and generated submission files are intentionally excluded from the public repository.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To train with the Kaggle data locally, place the competition CSV files under `data/` and run:

```bash
python scripts/train_model.py
```

To generate a Kaggle-format submission:

```bash
python scripts/make_submission.py --data-dir data/aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026 --output submission.csv --horizon 10 --target-col WBT
```

## Responsible Use

HeatGuard AI is decision-support software. It should support planning and awareness, not replace official weather alerts, medical judgment, or government heat-action protocols.
