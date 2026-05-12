# ============================================================
#   SPAM EMAIL DETECTION USING PYTHON
#   NIELIT Gorakhpur — Data Science & ML using Python
#   Session: 2025-2026
# ============================================================

# ──────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pickle
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

print("=" * 60)
print("   SPAM EMAIL DETECTION — NIELIT PROJECT")
print("=" * 60)

# ──────────────────────────────────────────
# STEP 2: LOAD THE DATASET
# ──────────────────────────────────────────
print("\n[1/7] Loading Dataset...")

# Dataset: SMS Spam Collection
# Download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# Place 'spam.csv' inside the 'data/' folder

df = pd.read_csv('data/spam.csv', encoding='latin-1')

# Keep only the relevant columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(f"     Dataset loaded successfully!")
print(f"     Total records : {df.shape[0]}")
print(f"     Total columns : {df.shape[1]}")
print(f"\n     First 5 rows:")
print(df.head())

# ──────────────────────────────────────────
# STEP 3: DATA CLEANING
# ──────────────────────────────────────────
print("\n[2/7] Data Cleaning...")

# Check for null values
print(f"     Null values:\n{df.isnull().sum()}")

# Check for duplicates
print(f"     Duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"     After removing duplicates: {df.shape[0]} records")

# Encode labels: ham = 0, spam = 1
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"\n     Class Distribution:")
print(df['label'].value_counts())
print(f"\n     Spam %  : {round(df['label'].value_counts(normalize=True)['spam']*100, 2)}%")
print(f"     Ham  %  : {round(df['label'].value_counts(normalize=True)['ham']*100, 2)}%")

# ──────────────────────────────────────────
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────
print("\n[3/7] Performing EDA & Generating Visualizations...")

# Add message length feature
df['msg_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

print(f"\n     Spam avg length  : {round(df[df['label']=='spam']['msg_length'].mean(), 1)} chars")
print(f"     Ham  avg length  : {round(df[df['label']=='ham']['msg_length'].mean(), 1)} chars")

# --- Plot 1: Class Distribution (Pie + Bar) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Class Distribution: Spam vs Ham', fontsize=16, fontweight='bold', color='#1F3864')

colors = ['#2196F3', '#F44336']

# Pie chart
axes[0].pie(df['label'].value_counts(), labels=['Ham', 'Spam'],
            autopct='%1.1f%%', colors=colors, startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[0].set_title('Proportion', fontweight='bold')

# Bar chart
df['label'].value_counts().plot(kind='bar', color=colors, ax=axes[1], edgecolor='white')
axes[1].set_title('Count', fontweight='bold')
axes[1].set_xlabel('Label', fontweight='bold')
axes[1].set_ylabel('Count', fontweight='bold')
axes[1].set_xticklabels(['Ham', 'Spam'], rotation=0)
for i, v in enumerate(df['label'].value_counts()):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: outputs/1_class_distribution.png")

# --- Plot 2: Message Length Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Message Length Distribution', fontsize=16, fontweight='bold', color='#1F3864')

for label, color in zip(['ham', 'spam'], ['#2196F3', '#F44336']):
    axes[0].hist(df[df['label'] == label]['msg_length'], bins=50,
                 alpha=0.6, label=label.upper(), color=color)
axes[0].set_title('Histogram', fontweight='bold')
axes[0].set_xlabel('Message Length (chars)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

sns.boxplot(data=df, x='label', y='msg_length', palette={'ham': '#2196F3', 'spam': '#F44336'},
            ax=axes[1])
axes[1].set_title('Box Plot', fontweight='bold')
axes[1].set_xlabel('Label')
axes[1].set_ylabel('Message Length (chars)')

plt.tight_layout()
plt.savefig('outputs/2_message_length_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: outputs/2_message_length_distribution.png")

# --- Plot 3: Top Words in Spam and Ham ---
from collections import Counter

def get_top_words(df, label, n=20):
    stop_words = set(stopwords.words('english'))
    words = []
    for msg in df[df['label'] == label]['message']:
        tokens = msg.lower().split()
        tokens = [w.strip(string.punctuation) for w in tokens
                  if w.strip(string.punctuation) not in stop_words and len(w) > 2]
        words.extend(tokens)
    return Counter(words).most_common(n)

spam_words = get_top_words(df, 'spam')
ham_words  = get_top_words(df, 'ham')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Top 20 Most Common Words', fontsize=16, fontweight='bold', color='#1F3864')

for ax, words, color, title in zip(
        axes,
        [spam_words, ham_words],
        ['#F44336', '#2196F3'],
        ['SPAM Messages', 'HAM Messages']):
    words_list, counts = zip(*words)
    ax.barh(words_list[::-1], counts[::-1], color=color, edgecolor='white')
    ax.set_title(title, fontweight='bold', color=color)
    ax.set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/3_top_words.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: outputs/3_top_words.png")

# ──────────────────────────────────────────
# STEP 5: TEXT PREPROCESSING
# ──────────────────────────────────────────
print("\n[4/7] Text Preprocessing...")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess a message:
    1. Lowercase
    2. Remove punctuation and numbers
    3. Tokenize
    4. Remove stop words
    5. Stem words
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = ''.join([c for c in text if c not in string.punctuation and not c.isdigit()])
    # Tokenize
    tokens = text.split()
    # Remove stop words and short words, then stem
    tokens = [stemmer.stem(word) for word in tokens
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

df['cleaned_message'] = df['message'].apply(preprocess_text)

print("     Sample cleaned messages:")
for i in range(3):
    print(f"\n     Original : {df['message'].iloc[i][:80]}...")
    print(f"     Cleaned  : {df['cleaned_message'].iloc[i][:80]}...")

# ──────────────────────────────────────────
# STEP 6: FEATURE EXTRACTION (TF-IDF)
# ──────────────────────────────────────────
print("\n[5/7] Feature Extraction using TF-IDF...")

X = df['cleaned_message']
y = df['label_encoded']

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_transformed = tfidf.fit_transform(X).toarray()

print(f"     Feature matrix shape : {X_transformed.shape}")
print(f"     Total features       : {X_transformed.shape[1]}")

# Train-Test Split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n     Training samples : {X_train.shape[0]}")
print(f"     Testing  samples : {X_test.shape[0]}")

# ──────────────────────────────────────────
# STEP 7: MODEL TRAINING
# ──────────────────────────────────────────
print("\n[6/7] Training Naive Bayes Model...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("     Model trained successfully!")

# ──────────────────────────────────────────
# STEP 8: MODEL EVALUATION
# ──────────────────────────────────────────
print("\n[7/7] Evaluating Model Performance...")

y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
report    = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
cm        = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print(f"   MODEL EVALUATION RESULTS")
print(f"{'='*60}")
print(f"   Accuracy  : {round(accuracy * 100, 2)}%")
print(f"\n   Classification Report:")
print(report)

# --- Plot 4: Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold', color='#1F3864', pad=15)
plt.tight_layout()
plt.savefig('outputs/4_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("     Saved: outputs/4_confusion_matrix.png")

# ──────────────────────────────────────────
# STEP 9: SAVE MODEL
# ──────────────────────────────────────────
print("\n[+] Saving Model and Vectorizer...")

with open('models/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("     Saved: models/spam_model.pkl")
print("     Saved: models/tfidf_vectorizer.pkl")

# ──────────────────────────────────────────
# STEP 10: PREDICT ON NEW MESSAGE
# ──────────────────────────────────────────
print("\n" + "="*60)
print("   LIVE PREDICTION DEMO")
print("="*60)

def predict_message(message):
    """Predict if a message is spam or ham."""
    cleaned = preprocess_text(message)
    vectorized = tfidf.transform([cleaned]).toarray()
    result = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    label = "🚨 SPAM" if result == 1 else "✅ HAM (Not Spam)"
    confidence = round(max(prob) * 100, 2)
    return label, confidence

test_messages = [
    "Congratulations! You have won a FREE iPhone. Click here to claim now!",
    "Hey, are we still meeting at 5pm today?",
    "URGENT: Your account has been suspended. Call 0800-FREE to restore access.",
    "Can you please send me the notes from yesterday's class?",
    "Win cash prizes! Free entry. Text WIN to 80800 now!"
]

for msg in test_messages:
    label, confidence = predict_message(msg)
    print(f"\n   Message   : {msg[:65]}...")
    print(f"   Result    : {label}  (Confidence: {confidence}%)")

print("\n" + "="*60)
print("   PROJECT COMPLETE! All outputs saved in 'outputs/' folder.")
print("="*60)
