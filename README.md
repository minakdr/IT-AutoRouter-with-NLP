# IT Support Auto-Router

An NLP-powered classifier that reads raw IT support ticket text and automatically routes it to the correct department, wrapped in a Streamlit web app.

Built by **Yousra Amina Kadri**.

---

## Overview

IT helpdesks receive a high volume of tickets that are typically triaged manually before being sent to the right team. This project trains a text classification model on ~48K historical tickets to predict the target department automatically, then serves that model through an interactive Streamlit interface.

## How It Works

1. **Data cleaning** — lowercasing and removal of noise not relevant to technical content.
2. **Text preprocessing** — tokenization, stopword removal, and lemmatization (NLTK).
3. **Feature extraction** — two approaches were compared:
   - **TF-IDF** (top 5,000 features, unigrams + bigrams)
   - **spaCy word embeddings** (300-dimensional semantic vectors, `en_core_web_md`)
4. **Modeling** — Logistic Regression (`class_weight='balanced'` to handle class imbalance) was trained and evaluated on both feature sets.
5. **Final pipeline** — the best-performing setup (TF-IDF) was rebuilt as a single `scikit-learn` `Pipeline` (`TfidfVectorizer` + `LinearSVC`) and saved with `joblib` for deployment.
6. **App** — `app.py` loads the saved pipeline and lets a user paste a ticket description to get an instant routing prediction.

## Dataset

- **File:** `all_tickets_processed_improved_v3.csv`
- **Rows:** 47,837 tickets
- **Columns:** `Document` (raw ticket text), `Topic_group` (target label)
- **Classes (8 departments):**

| Department | Ticket Count |
|---|---|
| Hardware | 13,617 |
| HR Support | 10,915 |
| Access | 7,125 |
| Miscellaneous | 7,060 |
| Storage | 2,777 |
| Purchase | 2,464 |
| Internal Project | 2,119 |
| Administrative rights | 1,760 |

The dataset is imbalanced, which is why `class_weight='balanced'` and stratified train/test splitting were used throughout.

## Model Performance

Two approaches were benchmarked on a held-out 20% test set (9,568 tickets):

| Approach | Accuracy | Weighted F1 |
|---|---|---|
| **TF-IDF + Logistic Regression** | **84.53%** | 0.85 |
| spaCy Word Embeddings + Logistic Regression | 69.66% | 0.70 |

TF-IDF clearly outperformed dense embeddings for this task — likely because ticket text relies heavily on specific keywords (e.g. "password", "laptop", "invoice") that TF-IDF captures directly, whereas averaged word vectors dilute that signal.

The **deployed model** uses TF-IDF features with a **LinearSVC** classifier inside a single `scikit-learn` pipeline for simplicity and speed at inference time.

## Tech Stack

- **Python**, **pandas**, **NumPy**
- **NLTK** (tokenization, stopwords, lemmatization)
- **spaCy** (`en_core_web_md` word embeddings)
- **scikit-learn** (`TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `Pipeline`)
- **joblib** (model persistence)
- **Streamlit** (web app / demo UI)

