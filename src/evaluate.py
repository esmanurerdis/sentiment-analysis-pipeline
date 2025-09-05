import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA = Path("data/processed/train_processed.csv")
MODELS = Path("models")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

def eval_holdout() -> None:
    df = pd.read_csv(DATA)
    X, y = df["text"], df["label"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = joblib.load(MODELS / "best_model.joblib")
    y_pred = model.predict(X_te)

    # rapor (hataları bastırmak için zero_division=0 eklenebilir)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    with open(REPORTS / "metrics.json", "r+", encoding="utf-8") as f:
        base = json.load(f)
    base["holdout"] = report
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    # confusion matrix
    labels = sorted(df["label"].unique().tolist())
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_png = REPORTS / "confusion_matrix.png"
    plt.savefig(out_png, dpi=150)
    print(f"✅ Değerlendirme tamam. Raporlar → {REPORTS}, görsel → {out_png}")

if __name__ == "__main__":
    eval_holdout()
