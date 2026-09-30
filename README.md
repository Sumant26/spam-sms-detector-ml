# 📱 Spam SMS Detector (Machine Learning & NLP)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

A natural language processing (NLP) machine learning pipeline designed to automatically classify text messages as **Spam** (unwanted promotional/phishing messages) or **Ham** (legitimate personal communications).

---

## 📌 Project Overview

Spam text messages are not only disruptive but often carry malicious links, scams, and financial phishing attempts. This project demonstrates how to build, evaluate, and interpret machine learning classifiers capable of identifying spam messages in an **imbalanced real-world dataset**.

### Key Highlights
- **Text Vectorization:** Converts raw text strings into numerical representations using **Bag-of-Words** and **TF-IDF (Term Frequency-Inverse Document Frequency)** with n-gram support.
- **Handling Class Imbalance:** Addresses the natural ~87% ham / 13% spam distribution using **stratified sampling**, **cost-sensitive learning (class weighting)**, and spam-centric evaluation metrics.
- **Model Comparison:** Evaluates multiple algorithms including **Multinomial Naive Bayes**, **Regularized Logistic Regression**, and **Linear SVM**.
- **Model Interpretability:** Extracts feature log-odds / weights to visualize top words and phrases that trigger spam and ham classifications.
- **Error Analysis:** Categorizes misclassifications (*False Alarms* vs. *Missed Spam*) to understand failure modes.
- **Interactive Inference & Export:** Demonstrates single-message probability scoring and saves trained pipelines using `joblib`.

---

## 📊 Dataset

The model is trained on the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI Machine Learning Repository):
- **Total Messages:** 5,572 SMS messages tagged as either `ham` (legitimate) or `spam`.
- **Distribution:**
  - `ham`: 4,825 messages (~86.6%)
  - `spam`: 747 messages (~13.4%)
- **Data Attributes:** Contains raw SMS text, label, and derived features such as character length.

---

## ⚙️ NLP & ML Pipeline

```
Raw SMS Text ───► Text Preprocessing (Tokenization & Cleaning)
                      │
                      ▼
               TF-IDF Vectorizer
         (Unigrams + Bigrams, Sublinear TF, Min DF)
                      │
                      ▼
        Classification Models (with Class Weighting)
     ┌────────────────┬───────────────────┬──────────────┐
     │  Naive Bayes   │   Logistic Reg.   │  Linear SVM  │
     └────────────────┴───────────────────┴──────────────┘
                      │
                      ▼
            Evaluation & Explanations
     (Confusion Matrix, Precision/Recall/F1, Top Words)
```

### Feature Engineering & Modeling Details
1. **TF-IDF Vectorization (`ngram_range=(1, 2)`):** Captures single words (*free*, *claim*) as well as key 2-word phrases (*free entry*, *call now*).
2. **Sublinear TF Scaling:** Replaces term frequency $tf$ with $1 + \log(tf)$ to dampen the impact of repetitive tokens.
3. **Cost-Sensitive Learning:** Configures `class_weight="balanced"` to penalize errors on minority spam instances proportionally.

---

## 📈 Model Performance & Comparison

Because accuracy is misleading on an 87/13 imbalanced dataset (a dummy classifier always predicting "ham" achieves 87% accuracy but 0% recall on spam), we focus on **Precision**, **Recall**, and **F1-Score for the Spam class**.

### Test Set Performance (20% Stratified Holdout)

| Model Pipeline | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | Overall Accuracy |
|---|---|---|---|---|
| **Majority Baseline (Always Ham)** | 0.00 | 0.00 | 0.00 | 86.6% |
| **Multinomial Naive Bayes (Standard TF-IDF)** | ~1.00 | ~0.70 | ~0.83 | ~96.0% |
| **Tuned Naive Bayes ($\alpha=0.05$, 1-2 grams)** | ~0.97 | ~0.88 | ~0.92 | ~98.0% |
| **Logistic Regression (`class_weight='balanced'`, $C=10$)** | **~0.95+** | **~0.91+** | **~0.93+** | **~98.5%** |
| **Linear SVM (Balanced)** | ~0.96 | ~0.90 | ~0.93 | ~98.3% |

### 5-Fold Stratified Cross-Validation (Spam F1-Score)
- **Naive Bayes:** `~0.838`
- **Naive Bayes (Tuned):** `~0.925`
- **Logistic Regression (Balanced):** `~0.934`
- **Linear SVM (Balanced):** `~0.931`

---

## 🔍 Model Insights: Top Indicators

Feature coefficients from the logistic regression model highlight key terms driving predictions:

- **🚨 Strongest SPAM Indicators:** `claim`, `prize`, `txt`, `free`, `won`, `urgent`, `call`, `guaranteed`, `service`, `reply`, `contact`, `150p`
- **💬 Strongest HAM Indicators:** `my`, `ok`, `me`, `later`, `ll`, `come`, `home`, `going`, `sorry`, `good`, `cant`

---

## 📁 Repository Structure

```
.
├── spam_detector.ipynb    # Complete end-to-end Jupyter Notebook pipeline
└── README.md              # Project documentation and specifications
```

---

## 🚀 Getting Started

### Option 1: Run in Google Colab (Easiest)
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File → Upload notebook** and select [`spam_detector.ipynb`](spam_detector.ipynb).
3. Run all cells sequentially (**Runtime → Run all** or `Shift + Enter`). Dataset is downloaded automatically within the notebook.

---

### Option 2: Run Locally

#### 1. Clone the repository
```bash
git clone https://github.com/Sumant26/spam-sms-detector-ml.git
cd spam-sms-detector-ml
```

#### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib jupyter joblib
```

#### 4. Launch Jupyter Notebook
```bash
jupyter notebook spam_detector.ipynb
```

---

## 🧪 Quick Code Example: Making Predictions

```python
import joblib

# Load the trained model pipeline
model = joblib.load("spam_model.joblib")

def check_message(message: str):
    prob_spam = model.predict_proba([message])[0, 1]
    label = "🚫 SPAM" if prob_spam >= 0.5 else "✅ HAM"
    print(f"[{label}] (Confidence: {prob_spam:.1%}) -> {message}")

# Test examples
check_message("Congratulations! You've won a FREE iPhone. Call 09061701461 to claim now!")
check_message("Hey, are we still on for dinner tonight?")
check_message("URGENT: Your account has been suspended. Reply with your PIN to reactivate.")
```

---

## 💡 Future Enhancements

- [ ] **Character N-Grams:** Use `analyzer="char_wb"` (2–5 grams) to resist intentional obfuscations (e.g., `FR33`, `w!n`, `p-r-i-z-e`).
- [ ] **Multi-Feature Fusion (`ColumnTransformer`):** Combine TF-IDF text features with message metadata (length, punctuation count, uppercase character ratio).
- [ ] **Interactive Web UI:** Deploy the model with a [Streamlit](https://streamlit.io/) or [Gradio](https://gradio.app/) dashboard for live SMS classification.
- [ ] **Deep Learning / Transformer Embeddings:** Benchmark against lightweight transformer models (e.g., DistilBERT / RoBERTa).

---

## 📜 License & Citation

- **Dataset:** [SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) created by Tiago A. Almeida and José María Gómez Hidalgo.
- **License:** Open for academic and personal learning projects under standard educational licensing.
