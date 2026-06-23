# AI-Powered Customer Retention Decision Support System for E-Commerce

This project is a customer retention decision-support system that predicts customer inactivity, explains risk drivers, segments customers, recommends marketing actions, and provides an interactive Streamlit dashboard with an optional AI advisor.

The goal is to help business users answer:

* Which customers are at risk of becoming inactive?
* Why are they at risk?
* What marketing action should be taken next?

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

The raw dataset is not included in this repository because of file size and reproducibility considerations.

### 4. Run the Notebooks in Order

Run the notebooks in this sequence:

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

Some notebooks include run/save control flags so generated outputs are not recreated unless the flag is changed to `True`.

### 5. Launch the Dashboard

After running the notebooks and generating the processed files, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

If Streamlit is not recognized, use:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard usually opens at:

```text
http://localhost:8501
```

## Dashboard Features

The dashboard is designed as a marketing analytics tool for customer retention.

### Executive Overview

Shows high-level retention KPIs:

* Total customers
* Predicted at-risk customers
* High-risk customers
* Average inactivity risk

Users can filter by customer segment, risk tier, and minimum inactivity probability.

### Customer Lookup

Allows users to search for a specific customer ID and view:

* Customer segment
* Risk tier
* Predicted inactivity probability
* Recency
* Frequency
* Monetary value
* Behavioral risk drivers
* Recommended marketing action
* Subject line, preheader, CTA, cadence, and KPI

### Manual Prediction

Allows users to enter customer attributes and generate a new inactivity prediction.

This is useful for testing customer scenarios or scoring customers that are not already in the dataset.

### Marketing Actions

Summarizes recommended campaigns and target audiences.

Customers are ranked using a campaign priority score that combines predicted inactivity risk and customer value.

### Segments

Shows customer segment size and inactivity rate by segment.

Segments include:

* Loyal Customers
* Regular Customers
* New Customers
* Low-Value Occasional Customers
* High-Value At-Risk Customers
* Dormant Customers

### Model & Explainability

Displays model performance and feature importance to explain what drives inactivity predictions.

### AI Advisor

The AI Advisor allows users to ask business questions such as:

* Why is this customer at risk?
* What offer should we send?
* Write a marketing message for this customer.
* Which campaign audience should be prioritized?

The advisor uses model scores, customer segment, risk drivers, and retention recommendation context. It includes a local rule-based fallback and can optionally use the OpenAI API if an API key is provided.

## Optional: OpenAI Advisor Setup

The dashboard can run without an OpenAI API key using the local advisor mode.

To enable the OpenAI-powered advisor, set your API key as an environment variable.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Mac/Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Do not commit API keys, `.env` files, or Streamlit secrets to GitHub.

## Project Overview

Customer retention is a major challenge for e-commerce businesses. Many companies have historical transaction data, but they may not know which customers are becoming inactive, what behaviors are driving the risk, or how marketing teams should respond.

This project uses the Online Retail II dataset to build a decision-support system that turns transaction data into:

* Customer-level features
* 90-day inactivity predictions
* Explainable churn-risk drivers
* Customer segments
* Marketing recommendations
* Dashboard-based decision support

## Dataset

The project uses transaction-level e-commerce data from the Online Retail II dataset.

The raw data includes:

* Invoice information
* Product details
* Quantity
* Invoice date
* Unit price
* Customer ID
* Country

Because the dataset does not include a churn label, inactivity was defined behaviorally.

A customer is labeled as inactive if they do not make another purchase within the next 90 days after the cutoff date.

```text
inactive_90d = 1 → no repeat purchase in next 90 days
inactive_90d = 0 → repeat purchase in next 90 days
```

## Key Results

After cleaning and preparing the data:

* Cleaned sales transactions: **779,425**
* Unique customers: **5,878**
* Modeling customers: **5,281**
* 90-day inactive customers: **2,989**
* 90-day active customers: **2,292**
* Inactive rate: **56.6%**
* Active rate: **43.4%**

Modeling results:

* Best model: **Gradient Boosting**
* ROC-AUC: **0.8154**
* Selected threshold: **0.40**
* Tuned recall: **90.8%**
* Tuned precision: **70.5%**
* Tuned F1-score: **0.7937**

Top model drivers:

1. Recency
2. Frequency
3. Monetary value
4. Product diversity / description diversity
5. Total quantity
6. Customer lifetime

## Project Workflow

```text
Raw Transactions
      ↓
Data Cleaning
      ↓
Customer-Level Feature Engineering
      ↓
90-Day Inactivity Label Creation
      ↓
Exploratory Data Analysis
      ↓
Predictive Modeling + MLflow Tracking
      ↓
Threshold Tuning
      ↓
Explainability
      ↓
Customer Segmentation
      ↓
Retention Recommendations
      ↓
Streamlit Dashboard + AI Advisor
```

## Repository Structure

```text
ai-retention-decision-support/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── models/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_customer_features.ipynb
│   ├── 03_churn_label.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_modeling_mlflow.ipynb
│   ├── 06_segmentation.ipynb
│   ├── 07_explainability.ipynb
│   └── 08_retention_recommendations.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

Generated datasets, raw data files, MLflow logs, model artifacts, and API keys are intentionally excluded from GitHub.

## Notebook Descriptions

### 01_data_cleaning.ipynb

Cleans the raw Online Retail II transaction data.

Main tasks include:

* Removing duplicate records
* Handling missing customer IDs
* Separating cancellations and returns
* Removing invalid transactions
* Creating cleaned transaction outputs

### 02_customer_features.ipynb

Transforms transaction-level data into customer-level features, including:

* Recency
* Frequency
* Monetary value
* Average order value
* Product diversity
* Customer lifetime
* Return and cancellation behavior

### 03_churn_label.ipynb

Creates the 90-day inactivity target label.

Historical transactions are used to build customer features, while future transactions are used only to define whether the customer became inactive.

### 04_eda.ipynb

Explores behavioral differences between active and inactive customers.

Key findings:

* Inactive customers generally have higher recency.
* Inactive customers tend to have lower frequency.
* Inactive customers tend to have lower monetary value.
* Inactive customers often have lower product diversity and shorter customer lifetime.

### 05_modeling_mlflow.ipynb

Trains and compares multiple classification models:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* Hist Gradient Boosting
* XGBoost

The notebook also logs model runs and metrics using MLflow.

Because customer retention prioritizes identifying at-risk customers, the final model threshold was tuned from the default 0.50 to 0.40 to improve recall.

### 06_segmentation.ipynb

Creates business-friendly customer segments using RFM-style rules.

### 07_explainability.ipynb

Explains the final model using:

* Feature importance
* SHAP workflow
* Customer-level risk driver logic

The strongest driver of inactivity was recency, followed by frequency and monetary value.

### 08_retention_recommendations.ipynb

Maps model predictions and customer segments into marketing actions.

Outputs include:

* Recommended action
* Offer level
* Campaign objective
* Subject line
* Preheader
* CTA
* Cadence
* Primary KPI
* Campaign priority score

## MLflow Tracking

Model training runs are logged locally with MLflow.

To view the MLflow UI:

```bash
mlflow ui --backend-store-uri models/mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow logs are excluded from GitHub because they are generated artifacts.

## Reproducibility Notes

The repository excludes:

* Raw data files
* Processed CSV files
* MLflow logs
* Model artifacts
* API keys and secrets

This keeps the repository lightweight and avoids uploading large or sensitive files. All generated files can be recreated by running the notebooks in order.

## Tools Used

* Python
* pandas
* numpy
* scikit-learn
* XGBoost
* SHAP
* MLflow
* Streamlit
* Plotly
* Matplotlib
* OpenAI API optional
* GitHub

## Author

**Haley Phan**
MS Business Analytics Capstone Project
University of Cincinnati

Project repository:
https://github.com/haleypgit/ai-retention-decision-support
