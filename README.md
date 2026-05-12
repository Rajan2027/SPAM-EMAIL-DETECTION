# 📧 Spam Email Detection using Python

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.0-orange?logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> A machine learning project to detect spam emails/SMS using **Naive Bayes** and **TF-IDF** vectorization.  
> Built as part of the **"Data Science and Machine Learning using Python"** course at **NIELIT Gorakhpur** (Session 2025–2026).

---

## 📌 Project Overview

This project builds an intelligent spam detection system trained on the **SMS Spam Collection Dataset** (5,572 messages). It classifies messages as either **spam** or **ham (legitimate)** with ~97% accuracy.

---

## 🗂️ Project Structure

```
spam-email-detection/
│
├── data/
│   └── spam.csv                    ← Dataset (download from Kaggle)
│
├── models/
│   ├── spam_model.pkl              ← Saved Naive Bayes model
│   └── tfidf_vectorizer.pkl        ← Saved TF-IDF vectorizer
│
├── notebooks/
│   └── Spam_Detection_Notebook.ipynb  ← Jupyter Notebook (step-by-step)
│
├── outputs/
│   ├── 1_class_distribution.png
│   ├── 2_message_length_distribution.png
│   ├── 3_top_words.png
│   └── 4_confusion_matrix.png
│
├── spam_detection.py               ← Main Python script
├── requirements.txt                ← Python dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-detection.git
cd spam-email-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
- Download **spam.csv** from [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Place it inside the `data/` folder

### 4. Run the Script
```bash
python spam_detection.py
```

### 5. OR Open the Jupyter Notebook
```bash
jupyter notebook notebooks/Spam_Detection_Notebook.ipynb
```

---

## 🔬 Machine Learning Pipeline

| Step | Description |
|------|-------------|
| 1️⃣ Data Loading | Load SMS Spam Collection Dataset using Pandas |
| 2️⃣ Data Cleaning | Remove duplicates, handle nulls, encode labels |
| 3️⃣ EDA | Visualize class distribution, message length, top words |
| 4️⃣ Text Preprocessing | Lowercase → Remove punctuation → Tokenize → Stopwords → Stemming |
| 5️⃣ Feature Extraction | TF-IDF Vectorization (3000 features, unigrams + bigrams) |
| 6️⃣ Model Training | Multinomial Naive Bayes classifier |
| 7️⃣ Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~97% |
| **Precision (Spam)** | ~96% |
| **Recall (Spam)** | ~93% |
| **F1-Score (Spam)** | ~94% |

---

## 🧪 Live Prediction Demo

```python
def predict_message(message):
    cleaned = preprocess_text(message)
    vectorized = tfidf.transform([cleaned]).toarray()
    result = model.predict(vectorized)[0]
    return "🚨 SPAM" if result == 1 else "✅ HAM"

predict_message("Congratulations! You have won a FREE iPhone!")
# Output: 🚨 SPAM

predict_message("Hey, are we still meeting at 5pm today?")
# Output: ✅ HAM
```

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Pandas** — Data manipulation
- **Matplotlib & Seaborn** — Visualization
- **NLTK** — Natural Language Processing
- **Scikit-learn** — ML model & evaluation
- **Jupyter Notebook** — Interactive development

---

## 📚 Dataset

- **Name:** SMS Spam Collection Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) / UCI ML Repository
- **Records:** 5,572 messages (4,825 Ham + 747 Spam)

---

## 👨‍💻 Author

**[Your Name]**  
Course: Data Science and Machine Learning using Python  
Institute: NIELIT, Gorakhpur  
Session: 2025–2026  
Guide: Mr. Prashant Gupta (Scientist)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
