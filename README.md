# Premier League Transfer Value Predictor

Predicts a Premier League player's market value from their stats (goals, assists, minutes, age and position) with linear regression. A machine learning portfolio project pairing football with a from-scratch regression workflow, built as a Jupyter notebook plus a small Gradio demo.

The full analysis lives in the notebook: [`notebook.ipynb`](notebook.ipynb). GitHub renders it with its charts and metrics, so you can read the whole thing without running anything.

## Features

- **Two models, compared**: model A (one row per player, career totals against current value) and model B (one row per player-season, each season's stats against the value at that time). Model B explains roughly twice the variance.
- **Full analysis**: distributions, a correlation heatmap, value-vs-stat scatter plots, coefficient interpretation and a predicted-vs-actual comparison that surfaces the over and undervalued players.
- **Evaluated against a baseline**: every score is measured against a predictor that always guesses the mean, not reported as a bare R².
- **Gradio demo**: a what-if predictor wrapping both fitted models, with a switcher to pick which one and input validation on the stats.

## Dataset

[Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) by David Cariboo (Kaggle), licensed CC0. The project uses three files from it: `players.csv`, `appearances.csv` and `player_valuations.csv`.

The CSVs are not committed (`appearances.csv` alone is ~149 MB, over GitHub's 100 MB limit). Download those three from the dataset page above and put them in a `data/` folder, where the notebook expects them.

## Tech Stack

- **Analysis**: pandas, numpy
- **Modelling**: scikit-learn (LinearRegression, GroupShuffleSplit, DummyRegressor)
- **Visualisation**: matplotlib, seaborn
- **Demo**: Gradio
- **Model persistence**: joblib
- **Dev tools**: black, ruff

## Setup and local run

1. Clone the repo:
   ```
   git clone <repository-url>
   cd pl-transfer-value-predictor
   ```
2. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the three CSVs into `data/` (see Dataset above).
5. Run the notebook top to bottom (`notebook.ipynb`). It trains both models and saves them to `model.joblib`.
6. Run the demo:
   ```
   python app.py
   ```
   then open the local URL it prints.

## Live demo

[Try the live demo on Render](https://pl-transfer-value-predictor.onrender.com/)

It runs on Render's free tier, so if it has been idle the first load takes 30 to 60 seconds to wake up, then it is instant.

## Design Choices

1. **Two models instead of one**:
   - Model A is a simple backbone: one row per player, career Premier League totals against current value. Model B is the real model: one row per player-season, so each season's stats sit next to the value from that same time. Keeping both makes the improvement concrete rather than asserted.
2. **Predicting the log of value**:
   - **Why**: market value is heavily right-skewed, so linear regression fit on the raw value gets pulled toward a few superstars. The log makes the target roughly symmetric, and back-transforming with a power of ten guarantees a positive prediction.
3. **GroupShuffleSplit for model B**:
   - **Why**: a player appears in many season-rows, so a plain random split could put their 2020 season in train and their 2021 in test. That leaks the player across the split and inflates the score. Grouping on the player keeps every one of their rows on the same side.
4. **A season feature in model B**:
   - **Why**: values have inflated hugely across eras. The season term lets the model absorb that inflation instead of blaming it on the stats.
5. **Input validation in `app.py`, not the notebook**:
   - **Why**: in the notebook a bad value should fail loudly, that is the point. But a stranger typing into the demo needs guarding, so `app.py` checks age, non-negative stats and a minutes cap and returns a clear message rather than a prediction from garbage.

## Model limits

This is a rough guide, not a valuation engine. Five stats cannot see reputation, potential, hype, contract length, injuries or a career built at other clubs, so both models break down at the extremes: they underrate elite stars and overrate raw prospects. That is a finding the notebook reports plainly rather than a flaw it hides.

## Project Structure

```
pl-transfer-value-predictor/
├── notebook.ipynb          # The analysis: data, EDA, both models, evaluation, interpretation
├── app.py                  # Gradio demo, loads the fitted models from model.joblib
├── model.joblib            # The two fitted models, saved by the notebook
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev dependencies (black, ruff)
├── data/                   # The three Kaggle CSVs (git-ignored, download separately)
└── README.md
```
