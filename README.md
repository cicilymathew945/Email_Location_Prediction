# Email_Location_Prediction

Email Location Prediction using the Enron Dataset

Project Overview

This project aims to predict the target folder location associated with emails from the famous Enron dataset. By leveraging natural language processing (NLP) and machine learning classification techniques, we analyze email metadata and content to determine the origin or destination locale.

Dataset

We utilized the Enron Email Dataset sourced from Kaggle. This dataset contains approximately 500,000 emails from over 150 users, primarily Enron senior management.

Data Preprocessing & EDA

Exploratory Data Analysis (EDA): Performed initial investigations to understand the distribution of emails across various locations.

Data Pruning: To ensure statistical significance and model robustness, we applied a threshold filter. Locations with 5 or fewer emails were dropped from the analysis to prevent noise and overfitting on sparse classes.

Modeling Approaches

The project followed an iterative approach to improve prediction accuracy:

Attempt 1: Baseline Modeling & NLP

Feature Engineering: Created initial numerical features from the email metadata.

Correlation Analysis: Studied the relationships between numerical features to identify predictors.

Text Vectorization: Implemented TF-IDF Vectorizer on the Subject and Email Body.

Model: Logistic Regression.

Attempt 2: Hybrid Feature Integration

Approach: Combined the text-based features (TF-IDF) with the engineered numerical features from Attempt 1.

Model: Logistic Regression.

Goal: To determine if metadata (like timestamp or recipient count) provided additive value to the textual content.

Attempt 3: Deep Learning Embeddings

Approach: Moved beyond bag-of-words (TF-IDF) to capture semantic meaning using BERT (Bidirectional Encoder Representations from Transformers).

Feature Extraction: Used BERT to generate high-dimensional embeddings of the email text.

Classification: Fed BERT embeddings into a Logistic Regression classifier.

Requirements

Python 3.x

Pandas / Numpy

Scikit-learn

Transformers (HuggingFace)

Matplotlib / Seaborn

Results and Findings
