# AI-Powered Customer Retention Decision Support System for E-Commerce

A practical, end-to-end system for predicting customer inactivity, understanding risk drivers, segmenting customers, and recommending targeted marketing actions — all delivered through an interactive Streamlit dashboard with an optional AI advisor.

### What this project helps you answer

* Which customers are at risk of becoming inactive?
* Why are they at risk?
* What marketing action should be taken next?

---

## How to Use This Project

### 1. Clone the Repository

```bash
git clone https://github.com/haleypgit/ai-retention-decision-support.git
cd ai-retention-decision-support
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

Dataset source:
https://archive.ics.uci.edu/dataset/502/online+retail+ii

Place the raw dataset here:

```text
data/raw/online_retail_II.csv
```

Note: The dataset is not included in this repository due to size and reproducibility considerations.

### 4. Run the Notebooks (in order)

```text
01_data_cleaning.ipynb
02_customer_features.ipynb
03_churn_label.ipynb
04_eda.ipynb
05_modeling_mlflow.ipynb
06_segmentation.ipynb
07_explainability.ipynb
08_retention_recommendations.ipynb
```

Some notebooks include run/save flags to avoid regenerating outputs unless explicitly enabled.

### 5. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

If needed:

```bash
python -m streamlit run dashboard/app.py
```

Open in browser:

```text
http://localhost:8501
```

---

## Dashboard Features

A business-focused interface for customer retention analysis.

### Executive Overview

High-level KPIs:

* Total customers
* Predicted at-risk customers
* High-risk customers
* Average inactivity risk

Includes filters for segment, risk tier, and probability threshold.

### Customer Lookup

Search by customer ID to view:

* Segment and risk tier
* Inactivity probability
* Recency, frequency, monetary value
* Risk drivers
* Recommended marketing action
* Campaign details (subject line, CTA, cadence, KPI)

### Manual Prediction

Input customer attributes to generate a new inactivity prediction.
Useful for testing scenarios or scoring new customers.

### Marketing Actions

Summarizes recommended campaigns and target audiences.
Customers are ranked using a priority score combining risk and value.

### Segments

Displays segment size and inactivity rate.

Segments include:

* Loyal Customers
* Regular Customers
* New Customers
* Low-Value Occasional Customers
* High-Value At-Risk Customers
* Dormant Customers

### Model & Explainability

Shows model performance and feature importance to explain predictions.

### AI Advisor

Ask business questions such as:

* Why is this customer at risk?
* What offer should we send?
* Write a marketing message
* Which audience should we prioritize?

Supports both:

* Local rule-based responses
* Optional OpenAI-powered responses

---

## Optional: OpenAI Advisor Setup

The dashboard works without an API key (local mode).

To enable OpenAI:

**PowerShell**

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

**Mac/Linux**

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Do not commit API keys or `.env` files.

---

## Project Overview

Customer retention is a key challenge in e-commerce. While companies collect transaction data, they often lack clear insights into:

* Which customers are becoming inactive
* What behaviors drive risk
* How marketing teams should respond

This project converts raw transaction data into:

* Customer-level features
* 90-day inactivity predictions
* Explainable risk drivers
* Customer segments
* Marketing recommendations
* Interactive decision support

---

## Dataset

Based on the **Online Retail II** dataset.

Includes:

* Invoice data
* Product details
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

### Target Definition

Since no churn label exists, inactivity is defined behaviorally:

```text
inactive_90d = 1 → no purchase in next 90 days
inactive_90d = 0 → purchase within next 90 days
```

---

## Key Results

### Data Summary

* Cleaned transactions: **779,425**
* Unique customers: **5,878**
* Modeling customers: **5,281**
* Inactive customers: **2,989**
* Active customers: **2,292**
* Inactive rate: **56.6%**

### Model Performance

* Best model: **Gradient Boosting**
* ROC-AUC: **0.8154**
* Threshold: **0.40**
* Recall: **90.8%**
* Precision: **70.5%**
* F1-score: **0.7937**

### Top Drivers

1. Recency
2. Frequency
3. Monetary value
4. Product diversity
5. Total quantity
6. Customer lifetime

---

## Project Workflow

```text
Raw Transactions
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Inactivity Labeling
      ↓
EDA
      ↓
Modeling + MLflow
      ↓
Threshold Tuning
      ↓
Explainability
      ↓
Segmentation
      ↓
Recommendations
      ↓
Dashboard + AI Advisor
```

---

## Repository Structure

```text
ai-retention-decision-support/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── .gitignore
├── requirements.txt
└── README.md
```

Generated files (data, models, MLflow logs) are excluded.

---

## Notebook Overview

### 01_data_cleaning.ipynb

* Removes duplicates
* Handles missing IDs
* Filters invalid transactions

### 02_customer_features.ipynb

* Builds RFM features
* Adds behavioral metrics

### 03_churn_label.ipynb

* Creates 90-day inactivity label

### 04_eda.ipynb

* Compares active vs inactive customers

### 05_modeling_mlflow.ipynb

* Trains multiple models
* Tracks experiments with MLflow
* Tunes threshold for recall

### 06_segmentation.ipynb

* Creates business-friendly segments

### 07_explainability.ipynb

* Feature importance
* SHAP workflow
* Risk driver logic

### 08_retention_recommendations.ipynb

* Maps predictions to marketing actions

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri models/mlruns
```

Open:

```text
http://127.0.0.1:5000
```

---

## Reproducibility Notes

Excluded from GitHub:

* Raw data
* Processed data
* MLflow logs
* Model artifacts
* API keys

All outputs can be regenerated by running the notebooks.

---

## Tools Used

* Python
* pandas, numpy
* scikit-learn, XGBoost
* SHAP
* MLflow
* Streamlit
* Plotly, Matplotlib
* OpenAI API (optional)

---

## Author

**Haley Phan**
MS Business Analytics Capstone Project
University of Cincinnati

Project repository:
https://github.com/haleypgit/ai-retention-decision-support
