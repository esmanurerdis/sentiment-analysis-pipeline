# Sentiment Analysis Pipeline (Turkish Product Reviews)

A complete NLP pipeline to classify Turkish product reviews as **positive** or **negative** using **TF-IDF + Logistic Regression**.

## Pipeline Overview
**Preprocessing → Model Training → Evaluation → Single Sentence Inference**

## Project Structure
```bash
sentiment-analysis-pipeline/
├── data/
│   ├── raw/amazon_sample.csv        # Sample Turkish reviews (18 rows)
│   └── processed/train_processed.csv # Preprocessed dataset
├── models/
│   └── best_model.joblib            # Trained model
├── reports/
│   ├── metrics.json                 # Model scores
│   └── confusion_matrix.png         # Confusion matrix
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```
## Usage

1️⃣ **Preprocessing**
```bash
python src/preprocess.py --input data/raw/amazon_sample.csv --text-col text --label-col label
python src/train.py
python src/evaluate.py
python src/infer.py
```
Results

CV F1-macro: ~0.70 (small sample dataset)

Holdout test set is tiny (4 sentences), metrics may fluctuate

Confusion matrix is provided for visualization

![Confusion Matrix](reports/confusion_matrix.png)


---


Technologies

Python 🐍

scikit-learn (TF-IDF + Logistic Regression)

pandas, nltk, textblob (preprocessing)

matplotlib, seaborn (visualization)

---

Future Improvements

Expand dataset to improve performance

Experiment with different algorithms (SVM, Random Forest, BERT)

Add web interface (Streamlit / Gradio)


---

 Türkçe Özet 

```markdown
## Türkçe Özet
Bu proje, Türkçe ürün yorumlarını olumlu veya olumsuz olarak sınıflandıran uçtan uca bir NLP pipeline’ıdır.  
- Ön işleme: metin temizleme, stopwords temizleme, lemmatization  
- Özellik çıkarımı: TF-IDF  
- Model: Logistic Regression  
- Değerlendirme: 5-fold cross-validation (F1-macro) ve holdout test seti  
- Tek cümle tahmini yapılabilir (inference)

