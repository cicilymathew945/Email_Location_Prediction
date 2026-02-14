# 📧 Email Location Prediction

*Predicting email folder locations using NLP and Machine Learning on the Enron Dataset*

---

## 📘 Project Overview

This project aims to **predict the target folder location** associated with emails from the **Enron dataset**.
By leveraging **Natural Language Processing (NLP)** and **Machine Learning classification** techniques, we analyze both **email metadata** and **content** to infer the destination or origin folder.

📂 **Dataset Used:**
🔗 [Enron Email Dataset (Original) — Kaggle](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
Contains approximately **500,000 emails** from **150+ Enron employees**, primarily senior management.

---

## 🩹 Data Preprocessing & EDA

* **Exploratory Data Analysis (EDA):** Investigated email distribution across different locations.
* **Data Pruning:** Applied a filter to remove locations with ≤5 emails to ensure statistical reliability and reduce noise.
* **Text Cleaning:** Removed stopwords, HTML tags, and special characters for cleaner text input.

---

## 🧠 Modeling Approaches

### 🔹 Attempt 1: Baseline Modeling (TF-IDF + Logistic Regression)

* **Feature Engineering:** Extracted numerical features (e.g., timestamp, recipients).
* **Correlation Analysis:** Identified key predictive relationships.
* **Text Vectorization:** Implemented **TF-IDF** on *Subject* and *Body*.
* **Model:** Logistic Regression (baseline).

---

### 🔹 Attempt 2: Hybrid Feature Integration

* **Approach:** Combined **TF-IDF** text features with **engineered metadata**.
* **Model:** Logistic Regression.
* **Goal:** Evaluate whether metadata adds predictive value beyond text.

---

### 🔹 Attempt 3: Deep Learning with BERT

* **Approach:** Used **BERT (Bidirectional Encoder Representations from Transformers)** embeddings to capture contextual semantics.
* **Feature Extraction:** Generated dense embeddings from email content using the HuggingFace *Transformers* library.
* **Classifier:** Logistic Regression on BERT embeddings.

---

## ⚙️ Requirements

| Library                       | Purpose                 |
| ----------------------------- | ----------------------- |
| Python 3.x                 | Core environment        |
| Pandas / NumPy             | Data manipulation       |
| Scikit-learn               | Modeling and evaluation |
| Transformers (HuggingFace) | Text embeddings         |
| Matplotlib / Seaborn       | Visualization           |

Install all dependencies:

```bash
pip install pandas numpy scikit-learn transformers matplotlib seaborn
```

---

## 📈 Results & Findings

* Baseline TF-IDF model provided a strong starting point.
* Hybrid approach showed **marginal improvement** with metadata.
* **BERT embeddings** significantly enhanced accuracy by capturing semantic nuance.
* Demonstrated how deep contextual understanding improves real-world NLP tasks.

---

## 🚀 Future Work

* Experiment with **fine-tuned BERT models** (e.g., *DistilBERT*, *RoBERTa*).
* Implement **multi-class neural classifiers** for scalability.
* Integrate **temporal patterns** and **email network features** for richer insights.

---

