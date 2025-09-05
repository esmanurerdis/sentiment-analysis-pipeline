# Duygu Analizi Pipeline (TF-IDF + Lojistik Regresyon)

Bu projede kullanıcı yorumlarını **pozitif / negatif** olarak sınıflandırıyorum.  
Akış: **Ön işleme → TF-IDF → Logistic Regression → 5-fold CV → Hold-out değerlendirme → Tek cümle tahmini.**

## Hızlı Başlangıç
```bash
pip install -r requirements.txt

# 1) Ön işleme (örnek veri: data/raw/amazon_sample.csv, metin kolonu: Text)
python src/preprocess.py --input data/raw/amazon_sample.csv --text-col Text --use-weak-labels

# 2) Eğitim (model: models/best_model.joblib)
python src/train.py

# 3) Değerlendirme (rapor + confusion matrix)
python src/evaluate.py

# 4) Tek cümle tahmini
python src/infer.py "worst purchase ever, completely disappointed"
