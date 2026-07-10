# Fake News Detection — NLP + TF-IDF + Random Forest
### Natural Language Processing · Text Classification · Python · Scikit-learn

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-NLP-orange?style=flat&logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/CV%20Accuracy-99.79%25-brightgreen?style=flat)
![Dataset](https://img.shields.io/badge/Articles-44%2C898-blue?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## Overview

This project builds a **fake news detection system** using Natural Language Processing (NLP) techniques on 44,898 real-world news articles.

A Scikit-learn **Pipeline** combines TF-IDF vectorisation of article text and titles with engineered word-count features, feeding into a **Random Forest classifier**. The final model achieves **99.79% accuracy** on 5-fold stratified cross-validation — meaning it correctly identifies fake vs real news with near-perfect consistency across unseen data.

---

## Problem Statement

Misinformation spreads faster than corrections. Can a machine learn to distinguish fake news from real news purely by reading the text — without any human fact-checking?

This project answers that with a pipeline that reads article titles and full text, converts them into numerical representations, and classifies each article as `fake` or `real`.

---

## Dataset

**Files:** `Fake.csv` + `True.csv` — combined into a single shuffled dataset

| Source | Articles | Label |
|---|---|---|
| `Fake.csv` | 23,481 | `fake` |
| `True.csv` | 21,417 | `real` |
| **Combined** | **44,898** | balanced |

**Columns used:**

| Column | Description |
|---|---|
| `title` | Article headline |
| `text` | Full article body |
| `label` | Target — `fake` or `real` |
| `title_len` | Engineered — character length of title |
| `text_len` | Engineered — character length of text body |
| `title_word_count` | Engineered — word count of title |
| `text_word_count` | Engineered — word count of text body |

> `subject` and `date` columns are present in raw data but excluded from modelling features.

---

## Project Workflow

```
Load CSVs → Label → Merge → Shuffle → Feature Engineering → Pipeline → Cross-Validation
```

**1. Data Loading & Labelling**
- `Fake.csv` labelled `fake`, `True.csv` labelled `real`
- Concatenated and shuffled with `random_state=42` for reproducibility

**2. Feature Engineering**
- `title_len`: character count of headline
- `text_len`: character count of body
- `title_word_count`: word count of headline
- `text_word_count`: word count of body text

**3. Pipeline Architecture**

```python
Pipeline([
    ('vectorizer', ColumnTransformer([
        ('tfidf_text',  TfidfVectorizer(), 'text'),
        ('tfidf_title', TfidfVectorizer(), 'title'),
        ('scaled_TWC',  StandardScaler(), ['title_word_count', 'text_word_count'])
    ])),
    ('RF', RandomForestClassifier(n_estimators=100, verbose=3))
])
```

- **TF-IDF on text**: converts full article body into term-frequency/inverse-document-frequency vectors
- **TF-IDF on title**: separately vectorises headlines (titles carry different linguistic signals)
- **StandardScaler on word counts**: normalises the engineered numeric features
- **Random Forest (100 trees)**: ensemble classifier trained on the combined feature matrix

**4. Cross-Validation**
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Ensures balanced fake/real ratio in each fold
- `scoring='accuracy'` with `verbose=3` for full training transparency

---

## Results

### Cross-Validation Accuracy (5-fold)

| Fold | Accuracy |
|---|---|
| Fold 1 | 99.81% |
| Fold 2 | 99.82% |
| Fold 3 | 99.74% |
| Fold 4 | 99.78% |
| Fold 5 | 99.79% |
| **Mean** | **99.79%** |

### Key Result

> **99.79% average accuracy across 5 stratified folds** — not a single train/test split, but robust cross-validated performance on 44,898 articles.
>
> The model correctly classifies approximately **44,808 out of 44,898 articles** on average — leaving only ~90 misclassifications across the full dataset.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas | Data loading, merging, feature engineering |
| NumPy | Numerical operations |
| Scikit-learn | TfidfVectorizer, ColumnTransformer, Pipeline, RandomForestClassifier, StratifiedKFold |
| Jupyter Notebook | Development environment |

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/SouravDey1/fake-news-detection-nlp.git
cd fake-news-detection-nlp
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn jupyter
```

**3. Add the datasets**

Place `Fake.csv` and `True.csv` in the project root. Available on [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

**4. Open the notebook**
```bash
jupyter notebook project_natural_language_processing.ipynb
```

> ⚠️ Training takes ~40 minutes on CPU due to 5-fold cross-validation with 100 trees × TF-IDF on 44,898 articles.

---

## Repository Structure

```
fake-news-detection-nlp/
│
├── project_natural_language_processing.ipynb   # Full notebook
└── README.md                                   # This file
```

---

## What I Learned

- How **TF-IDF** converts raw text into numerical vectors that capture term importance across a corpus — and why it outperforms simple word counts for text classification
- Why **separate TF-IDF on titles and text** matters — headlines and body text carry different linguistic fingerprints for fake vs real news
- How **Scikit-learn Pipelines** cleanly chain preprocessing and modelling — preventing data leakage and making the code reproducible
- The difference between a **single train/test split** and **stratified cross-validation** — and why CV scores are far more trustworthy for model evaluation
- That **99.79% accuracy is real here** — the dataset is well-balanced (~52% fake, ~48% real), so high accuracy genuinely reflects model quality, not class imbalance
- The practical reality of NLP at scale: TF-IDF on 44,898 full articles takes significant memory and compute time — understanding computational tradeoffs is part of the skill

---

## Author

**Sourav Dey**
Final-Year EEE Student · Ahsanullah University of Science and Technology (AUST)
📧 sourav41dey@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/sourav-dey-754298251/)
🐙 [GitHub](https://github.com/SouravDey1)

---

*Part of my Data Science & Machine Learning portfolio.*

*Other projects:*
- [University Clustering — K-Means & PCA](https://github.com/SouravDey1/university-kmeans-clustering)
- [California Housing — Random Forest](https://github.com/SouravDey1/california-housing-random-forest)
- [Diabetes Prediction — KNN](https://github.com/SouravDey1/diabetes-knn-classification)
- [Ad Click Prediction — Logistic Regression](https://github.com/SouravDey1/ad-click-logistic-regression)
- [E-Commerce Spending — Linear Regression](https://github.com/SouravDey1/ecommerce-linear-regression)
