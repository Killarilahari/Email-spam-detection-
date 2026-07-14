

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string, re

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_PATH = None  # e.g. "spam.csv"

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def generate_synthetic_spam_data(n=5000):
    spam_msgs = [
        "Congratulations! You have won a free prize. Click here to claim now!",
        "URGENT: Your account will be suspended. Verify immediately.",
        "FREE entry to win cash prize. Text WIN to 80088 now.",
        "You are selected for a $1000 gift card. Claim before it expires!",
        "Your mobile number has been selected as a winner. Reply YES to claim.",
        "Get rich quick! Work from home and earn $5000 a week guaranteed.",
        "LIMITED OFFER: Buy now and get 90% discount. Hurry offer ends today!",
        "Dear customer your loan is approved. Call us now to get money fast.",
        "Hot singles in your area want to meet you tonight. Click link below.",
        "Final warning: your subscription will expire. Renew immediately.",
    ]
    ham_msgs = [
        "Hey, are you coming to the meeting tomorrow?",
        "Can you please send me the report by end of day?",
        "I will be late today. Please start without me.",
        "Happy birthday! Hope you have a wonderful day.",
        "Did you watch the match last night? It was amazing!",
        "Let's catch up for lunch this week. Are you free on Thursday?",
        "The project deadline has been moved to next Friday.",
        "I am on my way. Will reach in about 20 minutes.",
        "Can you pick up some groceries on your way home?",
        "The presentation went really well. They loved our proposal.",
    ]
    spam = [spam_msgs[i % len(spam_msgs)] for i in range(int(n * 0.3))]
    ham  = [ham_msgs[i  % len(ham_msgs)]  for i in range(int(n * 0.7))]
    df = pd.DataFrame({"label": ["spam"]*len(spam)+["ham"]*len(ham),
                       "text":  spam + ham})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

if DATA_PATH:
    df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df[["v1","v2"]].rename(columns={"v1":"label","v2":"text"})
else:
    print("No DATA_PATH set -> generating synthetic spam dataset.\n")
    df = generate_synthetic_spam_data()

print(f"Dataset loaded: {len(df)} messages")
print(df["label"].value_counts().to_string())

# ── 2. EDA ───────────────────────────────────────────────────────────────────
df["message_length"] = df["text"].apply(len)
df["word_count"]     = df["text"].apply(lambda x: len(x.split()))
print("\n" + "="*60)
print("MESSAGE STATISTICS BY CLASS")
print("="*60)
print(df.groupby("label")[["message_length","word_count"]].mean().round(1).to_string())

# ── 3. PREPROCESSING ─────────────────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","them",
    "what","which","who","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "a","an","the","and","but","or","for","in","of","on","to","as","if",
    "with","at","by","from","that","this","just","not","no","can","get"
}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

# ── 4. TF-IDF ────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                             min_df=2, sublinear_tf=True)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"].map({"ham":0,"spam":1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f"\nTrain: {X_train.shape[0]}  |  Test: {X_test.shape[0]}  |  Features: {X.shape[1]}")

# ── 5. TRAIN 3 MODELS ────────────────────────────────────────────────────────
models = {
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
}
results = {}

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cv = cross_val_score(clf, X, y, cv=5, scoring="f1")
    results[name] = {
        "model": clf, "y_pred": y_pred,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "cv_f1": cv.mean(), "cv_std": cv.std(),
    }
    r = results[name]
    print(f"\n{name}")
    print(f"  Accuracy  : {r['accuracy']:.2%}")
    print(f"  Precision : {r['precision']:.2%}")
    print(f"  Recall    : {r['recall']:.2%}")
    print(f"  F1-Score  : {r['f1']:.2%}")
    print(f"  CV F1     : {r['cv_f1']:.2%} (+/- {r['cv_std']:.2%})")

# ── 6. BEST MODEL ────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1"])
best      = results[best_name]
best_clf  = best["model"]

print(f"\n{'='*60}")
print(f"BEST MODEL : {best_name}  (F1: {best['f1']:.2%})")
print(f"{'='*60}")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Ham (Not Spam)","Spam"]))

# ── 7. VISUALIZATIONS ────────────────────────────────────────────────────────
# Plot 1 — Class distribution
counts = df["label"].value_counts()
plt.figure(figsize=(6,4))
bars = plt.bar(["Ham (Not Spam)","Spam"], counts.values,
               color=["#2ecc71","#e74c3c"], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
             str(val), ha="center", fontsize=12, fontweight="bold")
plt.title("Dataset Class Distribution", fontsize=14, fontweight="bold")
plt.ylabel("Number of Messages")
plt.tight_layout()
plt.savefig("spam_distribution.png", dpi=150)
plt.show(); plt.close()

# Plot 2 — Model comparison
metric_labels = ["Accuracy","Precision","Recall","F1-Score"]
model_names   = list(results.keys())
x = np.arange(len(metric_labels))
width = 0.25
colors = ["#3498db","#e67e22","#2ecc71"]
fig, ax = plt.subplots(figsize=(10,5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name]["accuracy"], results[name]["precision"],
            results[name]["recall"],   results[name]["f1"]]
    bars = ax.bar(x + i*width, vals, width, label=name, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title("Model Comparison - Accuracy, Precision, Recall, F1", fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show(); plt.close()

# Plot 3 — Confusion matrix (best model)
cm = confusion_matrix(y_test, best["y_pred"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Ham","Predicted Spam"],
            yticklabels=["Actual Ham","Actual Spam"])
plt.title(f"Confusion Matrix - {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show(); plt.close()

print("Saved: spam_distribution.png, model_comparison.png, confusion_matrix.png")

# ── 8. SPAM DETECTION SUMMARY ────────────────────────────────────────────────
spam_count = int(y.sum())
ham_count  = int((y==0).sum())
spam_pct   = spam_count / len(y) * 100

print(f"\n{'='*60}")
print("SPAM DETECTION SUMMARY")
print(f"{'='*60}")
print(f"Total Messages    : {len(y)}")
print(f"Ham (Legitimate)  : {ham_count}  ({100-spam_pct:.1f}%)")
print(f"Spam (Junk)       : {spam_count}  ({spam_pct:.1f}%)")
print(f"Best Model        : {best_name}")
print(f"Spam Caught       : {best['recall']:.2%}")
print(f"False Alarm Rate  : {1-best['precision']:.2%}")

# ── 9. SAMPLE PREDICTIONS ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS")
print(f"{'='*60}")

test_messages = [
    "Congratulations! You have won a free cash prize. Click now!",
    "Hey, are you coming to the meeting tomorrow?",
    "URGENT: Your account will be suspended. Verify immediately.",
    "Let's catch up for lunch this week.",
    "You have been selected for a special reward. Claim today!",
    "Can you send me the notes from today's lecture?",
]
for msg in test_messages:
    cleaned  = preprocess(msg)
    features = vectorizer.transform([cleaned])
    pred     = best_clf.predict(features)[0]
    prob     = max(best_clf.predict_proba(features)[0]) if hasattr(best_clf,"predict_proba") else None
    label    = "SPAM" if pred == 1 else "HAM (Not Spam)"
    icon     = "🔴" if pred == 1 else "🟢"
    conf_str = f"  (confidence: {prob:.0%})" if prob else ""
    print(f"\n  [{icon} {label}]{conf_str}")
    print(f"  Message : {msg[:70]}")

# ── 10. REUSABLE FUNCTION ────────────────────────────────────────────────────
def predict_spam(message):
    """
    Pass any string -> returns ('Spam' or 'Ham (Not Spam)', confidence%)
    Example: label, conf = predict_spam("You won a prize!")
    """
    cleaned  = preprocess(message)
    features = vectorizer.transform([cleaned])
    pred     = best_clf.predict(features)[0]
    label    = "Spam" if pred == 1 else "Ham (Not Spam)"
    if hasattr(best_clf, "predict_proba"):
        conf = max(best_clf.predict_proba(features)[0])
        return label, f"{conf:.0%}"
    return label, "N/A"
