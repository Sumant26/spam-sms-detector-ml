# 📱 Spam SMS Detector

Classify text messages as spam or ham using TF-IDF features and Naive Bayes / logistic regression.

**Skills:** text preprocessing · bag-of-words · TF-IDF · n-grams · Naive Bayes · imbalanced classes · class weights · precision/recall/F1 · error analysis

**Time:** one weekend · **Level:** beginner

## Run it
Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → `spam_detector.ipynb`, then run top to bottom. The data downloads automatically.

## Expected results (spam class, test set)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Always "ham" | – | 0% | 0 |
| Naive Bayes (TF-IDF) | ~100% | ~70% | ~0.83 |
| Logistic regression (balanced, 1–2 grams) | ~95%+ | ~90%+ | ~0.93+ |

Data: [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI).
# spam-sms-detector-ml
