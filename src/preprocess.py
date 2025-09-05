import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import Word

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

# metin temizleme 
def clean_text(s: str) -> str:
    """
    Küçük harfe çevir, noktalama/sayıları kaldır, fazla boşlukları sıkıştır.
    """
    s = s.lower()
    s = re.sub(r"[\W_]+", " ", s)   # alfanümerik olmayanları kaldır
    s = re.sub(r"\d+", " ", s)      # sayıları kaldır
    s = re.sub(r"\s+", " ", s).strip()
    return s

# VADER ile zayıf etiket (weak label) üretimi 
def weak_label_vader(df, text_col="text"):
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    POS_T, NEG_T = 0.20, -0.20  # eşikler

    def lab(x):
        c = sia.polarity_scores(str(x))["compound"]
        return "olumlu" if c > POS_T else ("olumsuz" if c < NEG_T else "nötr")

    df["label"] = df[text_col].apply(lab)
    df = df[df["label"] != "nötr"]  # nötrleri at
    return df

def preprocess(input_path: str,
               text_col: str = "text",
               label_col: str | None = None,
               use_weak_labels: bool = False,
               out_name: str = "train_processed.csv") -> None:
    """
    1) Dosyayı oku (csv/xlsx)
    2) Metni temizle + stopwords/lemmatization
    3) Etiket yoksa VADER ile zayıf etiket üret (opsiyon)
    4) 'data/processed/' klasörüne yaz
    """
    # 1) oku
    if input_path.lower().endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    # güvenlik
    assert text_col in df.columns, f"{text_col} isimli bir kolon bulunamadı."
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)

    # 2) temizleme
    df[text_col] = df[text_col].map(clean_text)

    # stopwords + lemmatization
    nltk.download("stopwords", quiet=True)
    sw = set(stopwords.words("turkish"))
    df[text_col] = df[text_col].apply(
        lambda x: " ".join([w for w in x.split() if w not in sw and len(w) > 2])
    )
    # textblob wordnet gerektirir; kurulu değilse try/except ile atla
    try:
        nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
       
    except Exception:
        pass  # lemmatization başarısızsa temiz metinle devam et

    # 3) etiket
    if not label_col:
        if use_weak_labels:
            df = weak_label_vader(df, text_col=text_col)
            label_col = "label"
        else:
            raise ValueError("label_col verilmedi ve use_weak_labels=False. "
                             "Ya label_col belirt ya da --use-weak-labels kullan.")

    # 4) çıktı
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})
    out_path = PROCESSED / out_name
    df.to_csv(out_path, index=False)
    print(f"✅ İşlendi → {out_path} (satır={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ön işleme: temizle + (opsiyonel) VADER ile etiketle")
    ap.add_argument("--input", required=True, help="csv/xlsx dosya yolu")
    ap.add_argument("--text-col", default="text", help="metin kolon adı (ör: Text/Review)")
    ap.add_argument("--label-col", default=None, help="varsa etiket kolon adı")
    ap.add_argument("--use-weak-labels", action="store_true", help="VADER ile olumlu/olumsuz etiket üret")
    ap.add_argument("--out-name", default="train_processed.csv", help="çıktı dosya adı")
    args = ap.parse_args()

    preprocess(args.input, text_col=args.text_col, label_col=args.label_col,
               use_weak_labels=args.use_weak_labels, out_name=args.out_name)
