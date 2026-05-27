# Data Directory

Place the Kaggle challenge CSV files here.

Competition:

https://www.kaggle.com/competitions/aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026

Optional Kaggle CLI download:

```bash
kaggle competitions download -c aether-iiit-lucknow-wet-bulb-forecasting-challenge-2026 -p data
```

Then unzip the downloaded archive inside `data/`.

Expected behavior:

- If `train.csv` exists, HeatGuard AI uses it by default.
- If multiple CSV files exist, the dashboard lets you choose one.
- If no CSV exists, the app generates a small synthetic demo dataset for UI testing only.
- Demo fallback data is clearly labeled in the dashboard and model metadata.

Do not commit private or large raw datasets to GitHub unless the competition rules allow it.
