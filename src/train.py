import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DATA = Path("data/processed/train_processed.csv")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

def train() -> None:
    df = pd.read_csv(DATA)
    X, y = df["text"], df["label"]

    # eğitim test böl
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # pipeline: TF-IDF + LR
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    # 5-fold CV (F1 macro)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro").mean()

    # eğit ve kaydet
    pipe.fit(X_tr, y_tr)
    joblib.dump(pipe, MODELS / "best_model.joblib")

    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"cv_f1_macro": float(cv_score)}, f, ensure_ascii=False, indent=2)

    print(f"✅ CV F1(macro): {cv_score:.4f} — model kaydedildi → models/best_model.joblib")

if __name__ == "__main__":
    train()

