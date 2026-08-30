"""
Gradio demo for the Premier League transfer value predictor

Loads the two models fitted in notebook.ipynb and serves a small web form that
predicts a player's market value from their stats. The user chooses which model
to use: model A (career totals) or model B (a single season).
"""

import sys

import gradio as gr
import joblib
import pandas as pd

MODEL_FILE = "model.joblib"
POSITIONS = ["Attack", "Defender", "Goalkeeper", "Midfield"]
MODEL_A_LABEL = "Model A (career totals)"
MODEL_B_LABEL = "Model B (single season)"
MIN_AGE, MAX_AGE = 15, 45
MIN_SEASON, MAX_SEASON = 2012, 2025
MAX_SEASON_MINUTES = 4000  # 38 full games plus stoppage time
MAX_CAREER_MINUTES = 50000

try:
    bundle = joblib.load(MODEL_FILE)
except FileNotFoundError:
    sys.exit(
        f"{MODEL_FILE} not found. Run notebook.ipynb first to train and save the models."
    )


def predict(
    model_choice: str,
    position: str,
    age: float | None,
    goals: int | None,
    assists: int | None,
    minutes: int | None,
    season: int | None,
) -> str:
    """
    Predict a player's market value from their stats

    :param model_choice: Which model to use, MODEL_A_LABEL or MODEL_B_LABEL
    :param position: One of POSITIONS; 'Attack' is the model's baseline category
    :param age: Player age in years
    :param goals: Goals scored (career total for model A, one season for model B)
    :param assists: Assists (career total for model A, one season for model B)
    :param minutes: Minutes played (career total for model A, one season for model B)
    :param season: Starting year of the season, used by model B only
    :return: A formatted euro string with the predicted value, or a plain message if an input is missing or out of range
    """
    if None in (age, goals, assists, minutes):
        return "Please fill in age, goals, assists and minutes."
    if not MIN_AGE <= age <= MAX_AGE:
        return f"Age must be between {MIN_AGE} and {MAX_AGE}."
    if min(goals, assists, minutes) < 0:
        return "Goals, assists and minutes cannot be negative."

    if model_choice == MODEL_B_LABEL:
        if season is None:
            return "Season is required for model B."
        if not MIN_SEASON <= season <= MAX_SEASON:
            return f"Season must be between {MIN_SEASON} and {MAX_SEASON} for model B."

        model, feature_cols = bundle["model_b"], bundle["feature_cols_b"]
        max_minutes = MAX_SEASON_MINUTES
    else:
        model, feature_cols = bundle["model_a"], bundle["feature_cols_a"]
        max_minutes = MAX_CAREER_MINUTES

    if minutes > max_minutes:
        return f"Minutes cannot exceed {max_minutes:,} for {model_choice}."

    row = {col: 0.0 for col in feature_cols}
    row["age"], row["goals"], row["assists"], row["minutes_played"] = (
        age,
        goals,
        assists,
        minutes,
    )

    if "season" in feature_cols:
        row["season"] = season

    pos_col = f"position_{position}"

    if pos_col in feature_cols:
        row[pos_col] = 1.0

    X = pd.DataFrame([row], columns=feature_cols)
    value = 10 ** model.predict(X)[0]

    return f"€{value:,.0f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(
            choices=[MODEL_A_LABEL, MODEL_B_LABEL], value=MODEL_B_LABEL, label="Model"
        ),
        gr.Dropdown(choices=POSITIONS, value="Attack", label="Position"),
        gr.Number(label="Age", value=25),
        gr.Number(label="Goals", value=0, precision=0),
        gr.Number(label="Assists", value=0, precision=0),
        gr.Number(label="Minutes played", value=0, precision=0),
        gr.Number(label="Season (start year, model B only)", value=2025, precision=0),
    ],
    outputs=gr.Textbox(label="Predicted market value"),
    title="Premier League Transfer Value Predictor",
    description=(
        "Predicts a player's market value from their stats. Model A uses career "
        "totals, model B a single season, so set the season year for B. It is a "
        "rough guide only, since transfer value depends on much that these five "
        "stats cannot capture."
    ),
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
