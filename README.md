# Customer Segmentation and Purchase Classification Dashboard

An interactive Streamlit dashboard that groups online-retail customers by buying behaviour, classifies them (frequent / infrequent, high-value / low-value) and estimates who is likely to buy again in the next 30 days.

**Live dashboard:** [Open the app](PASTE-YOUR-STREAMLIT-LINK-HERE)

> Free Streamlit apps go to sleep when unused. If you see a "wake up" button, click it and wait a minute.

## What the problem asked for

| Requirement | How this project covers it |
|---|---|
| Clustering | K-Means on RFM features, compared across raw RFM, PCA and Kernel PCA |
| Feature extraction | PCA and Kernel PCA (RBF kernel) |
| Classification | Frequent vs Infrequent and High-value vs Low-value, five models each |
| Prediction | Will the customer buy in the next 30 days? |

## Dashboard pages

1. **How to read this**: plain-English glossary
2. **Overview**: revenue, orders, customers, monthly trend, top products and countries (with date and country filters)
3. **Customer segments**: four named segments, profile table, charts, CSV download
4. **Classification**: model comparison and confusion matrices for both experiments
5. **Future purchase**: next-30-day prediction results and feature importance
6. **Predict a customer**: look up a customer ID or enter Recency, Frequency and Monetary values

## Dataset

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii): transactions of a UK online gift retailer, 1 Dec 2009 to 9 Dec 2010.

After cleaning (missing customer IDs, duplicates, cancelled invoices, zero prices and manual adjustments removed): **400,496 rows, 4,298 customers, 19,098 orders, 4,016 products, 37 countries, about £8.7M revenue.**

## Method

1. **Cleaning**: drop rows without a CustomerID, remove duplicates, fix mixed date formats, remove cancelled invoices, zero-price rows and manual entries.
2. **RFM features** per customer: Recency (days since last order), Frequency (number of orders), Monetary (total spend).
3. **Preparation**: log transform, then standardise.
4. **Feature extraction**: PCA and Kernel PCA (RBF, gamma = 0.1) to two dimensions.
5. **Clustering**: K-Means with K = 4, chosen using the elbow method and silhouette score.
6. **Classification**: Logistic Regression, Decision Tree, Random Forest, KNN and SVM, trained on an 80/20 stratified split.
7. **Future purchase**: features from history before 10 Nov 2010, target is any purchase between 10 Nov and 10 Dec 2010.

### The four segments

| Segment | Typical behaviour |
|---|---|
| High-Value Loyal | Recent, frequent, highest spend |
| Moderate-Value | Decent spend, less recently active |
| Recent Occasional | Bought recently but only a few times, low spend |
| Inactive / At-Risk | Long gap since last order, usually one order, lowest spend |

## Key results (from the project notebook)

| Task | Best model | Result |
|---|---|---|
| Frequent vs Infrequent | SVM | Accuracy 0.854, ROC-AUC 0.931 |
| High-value vs Low-value | Decision Tree (accuracy), SVM (ROC-AUC) | Accuracy 0.851, ROC-AUC 0.927 |
| Next-30-day purchase | Random Forest (accuracy), Logistic Regression (ROC-AUC) | Accuracy 0.749, ROC-AUC 0.733 |

Segmentation quality (silhouette score at K = 4): Kernel PCA 0.431, PCA 0.373, raw RFM 0.331. Cluster stability (Adjusted Rand Index across two random seeds): 0.998.

**Limitations:** the two classification tasks use labels created from the same RFM data, so their high scores are partly built in. The next-30-day model finds fewer than half of the real buyers (low recall), so treat its output as a guide only.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. It reads `online_retail_cleaned.csv.gz` from the same folder (a plain `online_retail_cleaned.csv` also works).

## Repository contents

```
.
├── app.py                        # the Streamlit dashboard
├── requirements.txt              # Python libraries needed
├── online_retail_cleaned.csv.gz  # cleaned data exported from the notebook
├── .streamlit/config.toml        # dashboard colour theme
├── .gitignore
└── README.md
```

## Team

- Add the names of your team members here.
- Dashboard: add your name here.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, Plotly.
