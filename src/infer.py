import sys
from pathlib import Path

import joblib

MODEL_PATH = Path("models/best_model.joblib")

def predict(text: str):
    pipe = joblib.load(MODEL_PATH)
    pred = pipe.predict([text])[0]
    proba = None
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba([text])[0].max()
    return pred, proba

if __name__ == "__main__":
    predict("bu ürünü hiç beğenmedim, kalitesi çok kötü")
    predict("mükemmel bir alışverişti, herkese tavsiye ederim")


