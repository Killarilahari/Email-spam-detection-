# 📧 Email Spam Detection Using Machine Learning and python

A machine learning system that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing and TF-IDF feature extraction. Three classifiers are trained and compared to identify the best-performing model automatically.

---

## 📌 Overview

Spam messages cost businesses and individuals billions annually in lost productivity and security risks. This project builds an end-to-end spam detection pipeline — from raw text preprocessing through model training, evaluation, and interactive prediction — using classical NLP techniques and multiple ML classifiers.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 98%+ | 98%+ | 100% | 99%+ |
| Logistic Regression | 98%+ | 98%+ | 98%+ | 98%+ |
| Random Forest | 98%+ | 98%+ | 98%+ | 98%+ |

> Best model selected automatically at runtime based on highest F1-Score.

---

## 🧠 How It Works

### Text Preprocessing Pipeline
- **Lowercasing** — normalizes all text
- **URL replacement** — replaces all links with token `url`
- **Number replacement** — replaces digits with token `num`
- **Punctuation removal** — strips all punctuation characters
- **Stopword filtering** — removes common words that carry no spam signal
- **Short token removal** — drops single-character tokens

### TF-IDF Feature Extraction
- `max_features = 5000` — top 5,000 most informative words
- `ngram_range = (1, 2)` — single words AND two-word phrases like "free prize" and "click now"
- `sublinear_tf = True` — log scaling reduces dominance of very frequent words
- `min_df = 2` — ignores words appearing in fewer than 2 messages

### Models Trained
- **Multinomial Naive Bayes** — fast probabilistic baseline for TF-IDF features
- **Logistic Regression** — strong linear classifier for sparse text data
- **Random Forest** — ensemble method for comparison

---

## 📂 Project Structure

```
Email-spam-detection/
│
├── spam_detection.py        # Main script — preprocessing, training, evaluation, prediction
├── spam.csv                 # Dataset (label, text columns)
├── spam_distribution.png    # Class balance chart
├── model_comparison.png     # Bar chart comparing all 3 models
├── confusion_matrix.png     # Heatmap for best model
└── README.md
```

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Killarilahari/Email-spam-detection.git
cd Email-spam-detection
```

**2. Install dependencies**
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

**3. Run the script**
```bash
python spam_detection.py
```

> By default the script generates a synthetic dataset for demonstration.
> To use the real SMS Spam Collection dataset, set `DATA_PATH = "spam.csv"` at the top.
> Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## 📈 Output Visualizations

**spam_distribution.png** — bar chart showing Ham vs Spam class balance

**model_comparison.png** — side-by-side comparison of all three models across all four metrics

**confusion_matrix.png** — heatmap for the best model showing correct vs incorrect predictions

**Sample Predictions:**
```
[🔴 SPAM]  (confidence: 100%)
  Message: Congratulations! You have won a free cash prize. Click now!

[🟢 HAM (Not Spam)]  (confidence: 100%)
  Message: Hey, are you coming to the meeting tomorrow?
```

---

## 🔧 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)

---

## 🔁 Reusable Prediction Function

```python
label, confidence = predict_spam("You have won a free prize! Click now.")
print(label, confidence)
# Output: Spam  100%
```

---

## 👩‍💻 Author

**Killari Lahari**

B.Tech – Computer Science and Engineering (AI and ML)

📧 laharikillari007@gmail.com

🔗 [LinkedIn](https://linkedin.com/in/lahari-killari-375587324)
