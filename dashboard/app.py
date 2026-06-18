from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import os
import textwrap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =========================
# App config
# =========================

st.set_page_config(
    page_title="Customer Retention Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

SELECTED_THRESHOLD = 0.40


# =========================
# Styling
# =========================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f8fb;
    }

    h1, h2, h3 {
        font-family: "Inter", sans-serif;
    }

    .hero-card {
        background: linear-gradient(135deg, #111827 0%, #312e81 48%, #7c3aed 100%);
        padding: 32px 36px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 18px 40px rgba(17, 24, 39, 0.18);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
        line-height: 1.15;
    }

    .hero-subtitle {
        font-size: 17px;
        color: rgba(255, 255, 255, 0.82);
        max-width: 980px;
    }

    .metric-card {
        background: white;
        padding: 22px 24px;
        border-radius: 20px;
        border: 1px solid #edf0f7;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        min-height: 128px;
    }

    .metric-label {
        font-size: 13px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 34px;
        color: #111827;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .metric-note {
        font-size: 13px;
        color: #6b7280;
    }

    .section-card {
        background: white;
        padding: 24px;
        border-radius: 22px;
        border: 1px solid #edf0f7;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin-bottom: 20px;
    }

    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }

    .pill {
        background-color: #eef2ff;
        color: #3730a3;
        padding: 6px 12px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
        font-size: 13px;
    }

    .driver-box {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .driver-title {
        font-weight: 800;
        color: #111827;
        margin-bottom: 4px;
    }

    .driver-text {
        color: #4b5563;
        font-size: 14px;
    }

    .message-box {
        background: linear-gradient(135deg, #fdf2f8 0%, #eef2ff 100%);
        border: 1px solid #e9d5ff;
        border-radius: 20px;
        padding: 20px 22px;
        margin-top: 10px;
    }

    .small-muted {
        color: #6b7280;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Helper functions
# =========================

def read_csv_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_data():
    customer_model = read_csv_if_exists(PROCESSED_PATH / "customer_modeling_90d.csv")
    customer_segments = read_csv_if_exists(PROCESSED_PATH / "customer_segments.csv")
    segment_summary = read_csv_if_exists(PROCESSED_PATH / "segment_summary.csv")
    recommendations = read_csv_if_exists(PROCESSED_PATH / "customer_retention_recommendations.csv")
    action_summary = read_csv_if_exists(PROCESSED_PATH / "retention_action_summary.csv")
    model_results = read_csv_if_exists(PROCESSED_PATH / "model_results_90d.csv")
    feature_importance = read_csv_if_exists(PROCESSED_PATH / "feature_importance_90d.csv")

    return {
        "customer_model": customer_model,
        "customer_segments": customer_segments,
        "segment_summary": segment_summary,
        "recommendations": recommendations,
        "action_summary": action_summary,
        "model_results": model_results,
        "feature_importance": feature_importance,
    }


def format_pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.1%}"


def format_num(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


def format_money(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


def risk_badge(risk_tier):
    risk_tier = str(risk_tier)
    if risk_tier == "High Risk":
        return '<span class="risk-high">High Risk</span>'
    elif risk_tier == "Medium Risk":
        return '<span class="risk-medium">Medium Risk</span>'
    else:
        return '<span class="risk-low">Low Risk</span>'


def assign_risk_tier(probability):
    if probability >= 0.75:
        return "High Risk"
    elif probability >= SELECTED_THRESHOLD:
        return "Medium Risk"
    else:
        return "Low Risk"


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def recommend_marketing_strategy(row):
    segment = row.get("customer_segment", "Unknown")
    risk_tier = row.get("risk_tier", "Low Risk")

    strategy = {
        "recommended_action": "Monitor / no immediate action",
        "offer_level": "No offer",
        "campaign_objective": "Monitor engagement",
        "message_angle": "Keep customer warm without over-discounting",
        "subject_line": "New picks are waiting for you",
        "preheader": "Explore recommendations based on what customers are loving now.",
        "cta": "Explore New Arrivals",
        "cadence": "No immediate campaign; include in regular newsletter",
        "primary_kpi": "Email engagement rate"
    }

    if segment == "High-Value At-Risk Customers" and risk_tier == "High Risk":
        strategy.update({
            "recommended_action": "VIP win-back campaign",
            "offer_level": "Premium offer",
            "campaign_objective": "Recover high-value customers before they become fully dormant",
            "message_angle": "Exclusive, personal, high-touch reactivation",
            "subject_line": "A special offer, just for you",
            "preheader": "We saved something extra for valued customers like you.",
            "cta": "Redeem Your Exclusive Offer",
            "cadence": "Email 1 now, reminder in 5 days, final reminder in 10 days",
            "primary_kpi": "Reactivation revenue"
        })

    elif segment == "High-Value At-Risk Customers":
        strategy.update({
            "recommended_action": "Personalized win-back offer",
            "offer_level": "Moderate offer",
            "campaign_objective": "Re-engage valuable customers with curated product recommendations",
            "message_angle": "Value reminder plus personalized recommendations",
            "subject_line": "We picked these with your past favorites in mind",
            "preheader": "Come back to products that match your shopping history.",
            "cta": "View Your Picks",
            "cadence": "Email now, product reminder in 7 days",
            "primary_kpi": "Repeat purchase rate"
        })

    elif segment == "Dormant Customers":
        strategy.update({
            "recommended_action": "Low-cost reactivation campaign",
            "offer_level": "Low-cost offer",
            "campaign_objective": "Reactivate dormant customers while controlling promotion cost",
            "message_angle": "Newness and rediscovery",
            "subject_line": "A lot has changed since your last order",
            "preheader": "See new arrivals and limited-time picks you may have missed.",
            "cta": "See What’s New",
            "cadence": "Email now, final reminder in 7 days",
            "primary_kpi": "Reactivation rate"
        })

    elif segment == "Low-Value Occasional Customers":
        strategy.update({
            "recommended_action": "Automated bundle or value campaign",
            "offer_level": "Low-cost offer",
            "campaign_objective": "Increase repeat purchase without expensive incentives",
            "message_angle": "Affordable bundles and easy add-ons",
            "subject_line": "Small finds you may love",
            "preheader": "Affordable picks based on your past shopping behavior.",
            "cta": "Shop Value Picks",
            "cadence": "Automated email now, bundle reminder in 7 days",
            "primary_kpi": "Conversion rate"
        })

    elif segment == "New Customers":
        strategy.update({
            "recommended_action": "Second-purchase nurture campaign",
            "offer_level": "Welcome / onboarding offer",
            "campaign_objective": "Convert first-time buyers into repeat customers",
            "message_angle": "Welcome journey plus next best purchase",
            "subject_line": "Ready for your next favorite find?",
            "preheader": "Here are a few picks to continue your shopping experience.",
            "cta": "Shop Your Recommendations",
            "cadence": "Email now, follow-up in 5 days, educational content in 10 days",
            "primary_kpi": "Second purchase rate"
        })

    elif segment == "Regular Customers":
        strategy.update({
            "recommended_action": "Personalized replenishment / cross-sell reminder",
            "offer_level": "Standard offer",
            "campaign_objective": "Maintain repeat purchase behavior",
            "message_angle": "Helpful reminder based on prior purchases",
            "subject_line": "You may be due for a restock",
            "preheader": "We found a few items that pair well with your past orders.",
            "cta": "Shop Recommended Items",
            "cadence": "Email now, cross-sell reminder in 7 days",
            "primary_kpi": "Repeat purchase rate"
        })

    elif segment == "Loyal Customers":
        strategy.update({
            "recommended_action": "Loyalty protection campaign",
            "offer_level": "Relationship-building offer",
            "campaign_objective": "Protect loyal customer relationship and prevent disengagement",
            "message_angle": "Appreciation, exclusivity, and early access",
            "subject_line": "A thank-you for being one of our best customers",
            "preheader": "Enjoy early access and loyalty benefits before everyone else.",
            "cta": "Unlock Your Loyalty Benefit",
            "cadence": "Email now, loyalty reminder in 10 days",
            "primary_kpi": "Loyal customer retention rate"
        })

    return pd.Series(strategy)


def create_marketing_message(row):
    action = row.get("recommended_action", "Monitor / no immediate action")
    recency = int(row.get("recency", 0)) if not pd.isna(row.get("recency", 0)) else 0
    frequency = int(row.get("frequency", 0)) if not pd.isna(row.get("frequency", 0)) else 0

    if action == "VIP win-back campaign":
        return (
            f"We noticed it has been {recency} days since your last order. "
            f"As one of our higher-value customers, we wanted to give you early access "
            f"to a personalized offer selected around your past shopping behavior."
        )

    if action == "Personalized win-back offer":
        return (
            "It has been a little while since your last purchase, so we curated a few "
            "recommendations based on your past orders. Come back to products that match "
            "your shopping history."
        )

    if action == "Low-cost reactivation campaign":
        return (
            "A lot has changed since your last visit. Explore new arrivals, customer "
            "favorites, and limited-time picks that may be worth another look."
        )

    if action == "Automated bundle or value campaign":
        return (
            "Based on your past shopping behavior, we found a few value-friendly bundles "
            "and add-ons that may fit your style."
        )

    if action == "Second-purchase nurture campaign":
        return (
            "Thanks for your first order. To help you continue your shopping experience, "
            "we selected a few next-step recommendations that pair well with what customers often buy next."
        )

    if action == "Personalized replenishment / cross-sell reminder":
        return (
            f"You have ordered from us {frequency} times before, so we selected a few items "
            f"that may pair well with your past purchases. Take a look before your next restock."
        )

    if action == "Loyalty protection campaign":
        return (
            "Thank you for being a loyal customer. You have built a strong shopping history "
            "with us, so we wanted to share early access and loyalty benefits before everyone else."
        )

    return (
        "No immediate outreach is recommended for this customer. Continue monitoring behavior "
        "and include them in standard marketing communication."
    )


def explain_customer(row, medians):
    explanations = []

    rules = [
        {
            "feature": "recency",
            "label": "Days since last purchase",
            "risk_condition": row.get("recency", 0) > medians.get("recency", 0),
            "high_text": "Customer has not purchased recently, which increases inactivity risk.",
            "low_text": "Customer purchased relatively recently, which lowers inactivity risk.",
            "gap": abs(row.get("recency", 0) - medians.get("recency", 0))
        },
        {
            "feature": "frequency",
            "label": "Purchase frequency",
            "risk_condition": row.get("frequency", 0) < medians.get("frequency", 0),
            "high_text": "Customer has fewer orders than the typical customer, indicating weaker engagement.",
            "low_text": "Customer orders more frequently than the typical customer, indicating stronger engagement.",
            "gap": abs(row.get("frequency", 0) - medians.get("frequency", 0))
        },
        {
            "feature": "monetary_value",
            "label": "Total customer value",
            "risk_condition": row.get("monetary_value", 0) < medians.get("monetary_value", 0),
            "high_text": "Customer has lower total spend than the typical customer.",
            "low_text": "Customer has higher total spend, which suggests stronger value and relationship depth.",
            "gap": abs(row.get("monetary_value", 0) - medians.get("monetary_value", 0))
        },
        {
            "feature": "product_diversity",
            "label": "Product diversity",
            "risk_condition": row.get("product_diversity", 0) < medians.get("product_diversity", 0),
            "high_text": "Customer purchased from a narrower product range, which may indicate weaker brand engagement.",
            "low_text": "Customer has explored a wider range of products, which may indicate stronger engagement.",
            "gap": abs(row.get("product_diversity", 0) - medians.get("product_diversity", 0))
        },
        {
            "feature": "customer_lifetime_days",
            "label": "Customer lifetime",
            "risk_condition": row.get("customer_lifetime_days", 0) < medians.get("customer_lifetime_days", 0),
            "high_text": "Customer has a shorter relationship history with the business.",
            "low_text": "Customer has a longer relationship history, which usually supports retention.",
            "gap": abs(row.get("customer_lifetime_days", 0) - medians.get("customer_lifetime_days", 0))
        },
        {
            "feature": "return_cancel_rate",
            "label": "Return / cancellation rate",
            "risk_condition": row.get("return_cancel_rate", 0) > medians.get("return_cancel_rate", 0),
            "high_text": "Customer has a higher return or cancellation rate than typical.",
            "low_text": "Customer does not show unusually high return or cancellation behavior.",
            "gap": abs(row.get("return_cancel_rate", 0) - medians.get("return_cancel_rate", 0))
        }
    ]

    for rule in rules:
        direction = "Increases risk" if rule["risk_condition"] else "Decreases risk"
        insight = rule["high_text"] if rule["risk_condition"] else rule["low_text"]
        explanations.append({
            "driver": rule["label"],
            "impact": direction,
            "insight": insight,
            "gap": rule["gap"]
        })

    explanations = sorted(explanations, key=lambda x: x["gap"], reverse=True)
    return explanations[:4]

# =========================
# GenAI Advisor Helpers
# =========================

def get_openai_api_key():
    """
    Get OpenAI API key from Streamlit secrets or environment variable.
    This keeps the key out of the codebase.
    """
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")


def build_customer_context(row, medians):
    """
    Build business-friendly context for one customer.
    This context is passed to the AI advisor.
    """
    drivers = explain_customer(row, medians)

    driver_text = "\n".join([
        f"- {d['driver']}: {d['impact']}. {d['insight']}"
        for d in drivers
    ])

    context = f"""
Customer Profile
----------------
Customer ID: {row.get("CustomerID", "N/A")}
Segment: {row.get("customer_segment", "N/A")}
Risk tier: {row.get("risk_tier", "N/A")}
Predicted inactivity probability: {row.get("predicted_inactive_probability", 0):.1%}
Predicted inactive in next 90 days: {row.get("predicted_inactive_90d", "N/A")}

Behavioral Metrics
------------------
Recency: {row.get("recency", "N/A")} days since last purchase
Frequency: {row.get("frequency", "N/A")} orders
Monetary value: {row.get("monetary_value", "N/A")}
Average order value: {row.get("avg_order_value", "N/A")}
Product diversity: {row.get("product_diversity", "N/A")}
Customer lifetime: {row.get("customer_lifetime_days", "N/A")} days
Return/cancellation rate: {row.get("return_cancel_rate", "N/A")}

Current Marketing Recommendation
--------------------------------
Recommended action: {row.get("recommended_action", "N/A")}
Offer level: {row.get("offer_level", "N/A")}
Campaign objective: {row.get("campaign_objective", "N/A")}
Message angle: {row.get("message_angle", "N/A")}
Subject line: {row.get("subject_line", "N/A")}
Preheader: {row.get("preheader", "N/A")}
CTA: {row.get("cta", "N/A")}
Cadence: {row.get("cadence", "N/A")}
Primary KPI: {row.get("primary_kpi", "N/A")}

Top Risk Drivers
----------------
{driver_text}

Suggested Message Template
--------------------------
{row.get("message_template", "N/A")}
"""

    return context.strip()


def build_portfolio_context(filtered_df):
    """
    Build portfolio-level context for campaign strategy questions.
    """
    total_customers = filtered_df["CustomerID"].nunique()
    avg_risk = filtered_df["predicted_inactive_probability"].mean()
    risk_mix = filtered_df["risk_tier"].value_counts().to_dict()
    segment_mix = filtered_df["customer_segment"].value_counts().head(8).to_dict()
    action_mix = filtered_df["recommended_action"].value_counts().head(8).to_dict()

    context = f"""
Portfolio Summary
-----------------
Customers in current filtered view: {total_customers}
Average predicted inactivity probability: {avg_risk:.1%}

Risk tier mix:
{risk_mix}

Top customer segments:
{segment_mix}

Recommended action mix:
{action_mix}
"""

    return context.strip()


def local_ai_advisor_response(question, customer_row=None, filtered_df=None, medians=None):
    """
    Local rule-based fallback advisor.
    This works even without an LLM API key.
    """
    question_lower = question.lower()

    if customer_row is not None:
        drivers = explain_customer(customer_row, medians)

        if "why" in question_lower or "risk" in question_lower or "churn" in question_lower:
            driver_lines = "\n".join([
                f"- **{d['driver']}**: {d['insight']}"
                for d in drivers
            ])

            return f"""
### Customer Risk Explanation

This customer is classified as **{customer_row.get("risk_tier", "N/A")}** with a predicted inactivity probability of **{customer_row.get("predicted_inactive_probability", 0):.1%}**.

The main drivers are:

{driver_lines}

From a marketing perspective, this customer should be handled as a **{customer_row.get("customer_segment", "N/A")}**.
"""

        if "offer" in question_lower or "send" in question_lower or "campaign" in question_lower:
            return f"""
### Recommended Campaign

**Recommended action:** {customer_row.get("recommended_action", "N/A")}  
**Offer level:** {customer_row.get("offer_level", "N/A")}  
**Campaign objective:** {customer_row.get("campaign_objective", "N/A")}  
**Primary KPI:** {customer_row.get("primary_kpi", "N/A")}  
**Cadence:** {customer_row.get("cadence", "N/A")}

This recommendation balances the customer’s predicted inactivity risk with their customer value and segment behavior.
"""

        if "message" in question_lower or "email" in question_lower or "copy" in question_lower:
            return f"""
### Suggested Marketing Message

**Subject:** {customer_row.get("subject_line", "N/A")}  
**Preheader:** {customer_row.get("preheader", "N/A")}  

{customer_row.get("message_template", "N/A")}

**CTA:** {customer_row.get("cta", "N/A")}
"""

        if "compare" in question_lower:
            return f"""
### Customer Comparison

Compared with the typical customer:

- Recency: **{customer_row.get("recency", "N/A")} days**
- Frequency: **{customer_row.get("frequency", "N/A")} orders**
- Monetary value: **{customer_row.get("monetary_value", "N/A")}**
- Product diversity: **{customer_row.get("product_diversity", "N/A")}**
- Customer lifetime: **{customer_row.get("customer_lifetime_days", "N/A")} days**

The customer’s segment is **{customer_row.get("customer_segment", "N/A")}**, and their current risk tier is **{customer_row.get("risk_tier", "N/A")}**.
"""

        return f"""
### Customer Advisor Summary

This customer belongs to **{customer_row.get("customer_segment", "N/A")}** and has a predicted inactivity probability of **{customer_row.get("predicted_inactive_probability", 0):.1%}**.

Recommended action: **{customer_row.get("recommended_action", "N/A")}**  
Offer level: **{customer_row.get("offer_level", "N/A")}**  
Primary KPI: **{customer_row.get("primary_kpi", "N/A")}**
"""

    if filtered_df is not None:
        high_risk = (filtered_df["risk_tier"] == "High Risk").sum()
        medium_risk = (filtered_df["risk_tier"] == "Medium Risk").sum()
        top_action = filtered_df["recommended_action"].value_counts().idxmax()

        return f"""
### Portfolio Advisor Summary

The current filtered audience contains **{filtered_df["CustomerID"].nunique():,} customers**.

- High-risk customers: **{high_risk:,}**
- Medium-risk customers: **{medium_risk:,}**
- Most common recommended action: **{top_action}**

A strong marketing strategy would prioritize high-risk, high-value customers first, then use lower-cost automated campaigns for dormant or low-value occasional customers.
"""

    return "Please select a customer or ask a portfolio-level question."


def call_openai_advisor(question, customer_context=None, portfolio_context=None, model_name="gpt-4.1-mini"):
    """
    Optional LLM-powered advisor.
    Falls back to local advisor if no API key is available.
    """
    api_key = get_openai_api_key()

    if not HAS_OPENAI or not api_key:
        return None

    client = OpenAI(api_key=api_key)

    system_prompt = """
You are a senior retention marketing analyst and CRM strategist.
You explain customer churn risk in business-friendly language.

Use only the context provided. Do not invent customer facts.
Be specific, concise, and marketing-savvy.

When relevant, include:
1. Risk interpretation
2. Key behavioral drivers
3. Recommended campaign strategy
4. Offer level
5. Suggested message angle
6. KPI to track

Avoid overly generic advice. Make the answer useful for a marketing manager.
"""

    user_prompt = f"""
User question:
{question}

Customer context:
{customer_context if customer_context else "No selected customer context provided."}

Portfolio context:
{portfolio_context if portfolio_context else "No portfolio context provided."}
"""

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
        )

        return response.output_text

    except Exception as e:
        return f"OpenAI advisor failed, using local advisor instead. Error: {e}"
    
@st.cache_resource
def train_final_model(customer_model_csv_path):
    customer_model = pd.read_csv(customer_model_csv_path)

    target = "inactive_90d"

    drop_cols = [
        "CustomerID",
        "first_purchase_date",
        "last_purchase_date",
        "active_90d",
        "inactive_90d"
    ]

    X = customer_model.drop(columns=[c for c in drop_cols if c in customer_model.columns])
    y = customer_model[target]

    categorical_cols = ["primary_country"] if "primary_country" in X.columns else []
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", onehot)
        ])
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    model.fit(X, y)

    medians = customer_model[numeric_cols].median(numeric_only=True).to_dict()

    countries = (
        sorted(customer_model["primary_country"].dropna().unique().tolist())
        if "primary_country" in customer_model.columns
        else []
    )

    return model, X.columns.tolist(), numeric_cols, categorical_cols, medians, countries


def build_dashboard_dataset(data, model, feature_cols):
    customer_model = data["customer_model"]
    recommendations = data["recommendations"]
    customer_segments = data["customer_segments"]

    if recommendations is not None:
        dashboard_df = recommendations.copy()
    else:
        dashboard_df = customer_model.copy()
        X_all = dashboard_df[feature_cols]
        dashboard_df["predicted_inactive_probability"] = model.predict_proba(X_all)[:, 1]
        dashboard_df["predicted_inactive_90d"] = (
            dashboard_df["predicted_inactive_probability"] >= SELECTED_THRESHOLD
        ).astype(int)
        dashboard_df["risk_tier"] = dashboard_df["predicted_inactive_probability"].apply(assign_risk_tier)

    if customer_segments is not None and "customer_segment" not in dashboard_df.columns:
        segment_cols = [
            col for col in [
                "CustomerID",
                "customer_segment",
                "R_score",
                "F_score",
                "M_score",
                "RFM_score"
            ]
            if col in customer_segments.columns
        ]

        dashboard_df = dashboard_df.merge(
            customer_segments[segment_cols],
            on="CustomerID",
            how="left"
        )

    if "risk_tier" not in dashboard_df.columns:
        dashboard_df["risk_tier"] = dashboard_df["predicted_inactive_probability"].apply(assign_risk_tier)

    if "campaign_priority_score" not in dashboard_df.columns:
        dashboard_df["campaign_priority_score"] = (
            dashboard_df["predicted_inactive_probability"] * np.log1p(dashboard_df["monetary_value"])
        )

    # Add marketing strategy fields if missing
    needed_strategy_cols = [
        "recommended_action",
        "offer_level",
        "campaign_objective",
        "message_angle",
        "subject_line",
        "preheader",
        "cta",
        "cadence",
        "primary_kpi"
    ]

    if any(col not in dashboard_df.columns for col in needed_strategy_cols):
        strategy_cols = dashboard_df.apply(recommend_marketing_strategy, axis=1)
        for col in strategy_cols.columns:
            dashboard_df[col] = strategy_cols[col]

    if "message_template" not in dashboard_df.columns:
        dashboard_df["message_template"] = dashboard_df.apply(create_marketing_message, axis=1)

    dashboard_df["CustomerID"] = dashboard_df["CustomerID"].astype(str)

    return dashboard_df


# =========================
# Load data
# =========================

data = load_data()

if data["customer_model"] is None:
    st.error("Missing data/processed/customer_modeling_90d.csv. Please run the modeling notebook first.")
    st.stop()

customer_model_path = PROCESSED_PATH / "customer_modeling_90d.csv"

model, feature_cols, numeric_cols, categorical_cols, medians, countries = train_final_model(
    str(customer_model_path)
)

dashboard_df = build_dashboard_dataset(data, model, feature_cols)

# Make CustomerID consistent
data["customer_model"]["CustomerID"] = data["customer_model"]["CustomerID"].astype(str)


# =========================
# Sidebar filters
# =========================

st.sidebar.title("Retention Command Center")
st.sidebar.caption("Filter the dashboard view")

available_segments = sorted(dashboard_df["customer_segment"].dropna().unique().tolist()) if "customer_segment" in dashboard_df.columns else []
available_risks = ["High Risk", "Medium Risk", "Low Risk"]

selected_segments = st.sidebar.multiselect(
    "Customer segments",
    options=available_segments,
    default=available_segments
)

selected_risks = st.sidebar.multiselect(
    "Risk tiers",
    options=available_risks,
    default=available_risks
)

min_probability = st.sidebar.slider(
    "Minimum inactivity probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05
)

filtered_df = dashboard_df.copy()

if selected_segments and "customer_segment" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["customer_segment"].isin(selected_segments)]

if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_tier"].isin(selected_risks)]

filtered_df = filtered_df[
    filtered_df["predicted_inactive_probability"] >= min_probability
]


# =========================
# Hero
# =========================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Customer Retention Command Center</div>
        <div class="hero-subtitle">
            A marketing analytics dashboard for identifying at-risk customers, explaining churn drivers,
            segmenting customer behavior, and recommending personalized retention actions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_customers = dashboard_df["CustomerID"].nunique()
at_risk_customers = int((dashboard_df["predicted_inactive_probability"] >= SELECTED_THRESHOLD).sum())
high_risk_customers = int((dashboard_df["risk_tier"] == "High Risk").sum())
avg_risk = dashboard_df["predicted_inactive_probability"].mean()

with kpi1:
    metric_card("Total Customers", format_num(total_customers), "Customers available for scoring")

with kpi2:
    metric_card("At-Risk Customers", format_num(at_risk_customers), f"Threshold = {SELECTED_THRESHOLD:.2f}")

with kpi3:
    metric_card("High-Risk Customers", format_num(high_risk_customers), "Priority campaign audience")

with kpi4:
    metric_card("Average Risk", format_pct(avg_risk), "Predicted inactivity probability")


# =========================
# Tabs
# =========================

tabs = st.tabs([
    "Executive Overview",
    "Customer Lookup",
    "Predict New Customer",
    "Marketing Actions",
    "Segments",
    "Model & Explainability",
    "AI Advisor"
])


# =========================
# Executive Overview
# =========================

with tabs[0]:
    st.subheader("Executive Overview")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Risk Distribution")

        risk_counts = filtered_df["risk_tier"].value_counts().reindex(available_risks).fillna(0).reset_index()
        risk_counts.columns = ["risk_tier", "customers"]

        if HAS_PLOTLY:
            fig = px.bar(
                risk_counts,
                x="risk_tier",
                y="customers",
                text="customers",
                color="risk_tier",
                color_discrete_map={
                    "High Risk": "#ef4444",
                    "Medium Risk": "#f59e0b",
                    "Low Risk": "#22c55e"
                }
            )
            fig.update_layout(
                height=360,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis_title="Risk Tier",
                yaxis_title="Customers"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(risk_counts.set_index("risk_tier")["customers"])

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Campaign Action Mix")

        action_counts = filtered_df["recommended_action"].value_counts().head(8).reset_index()
        action_counts.columns = ["recommended_action", "customers"]

        if HAS_PLOTLY:
            fig = px.pie(
                action_counts,
                values="customers",
                names="recommended_action",
                hole=0.45
            )
            fig.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(action_counts, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Top Campaign Opportunities")

    top_cols = [
        "CustomerID",
        "customer_segment",
        "risk_tier",
        "predicted_inactive_probability",
        "monetary_value",
        "campaign_priority_score",
        "recommended_action",
        "offer_level",
        "primary_kpi"
    ]

    top_cols = [c for c in top_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[top_cols]
        .sort_values("campaign_priority_score", ascending=False)
        .head(50),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Customer Lookup
# =========================

with tabs[1]:
    st.subheader("Customer Lookup")

    lookup_col1, lookup_col2 = st.columns([1, 2])

    with lookup_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Search Customer")

        customer_query = st.text_input("Enter Customer ID", placeholder="Example: 12346")

        customer_ids = dashboard_df["CustomerID"].sort_values().unique().tolist()

        if customer_query:
            matching_ids = [cid for cid in customer_ids if customer_query in cid]
        else:
            matching_ids = customer_ids[:100]

        selected_customer = st.selectbox(
            "Select matching customer",
            options=matching_ids if matching_ids else ["No match found"]
        )

        st.markdown("</div>", unsafe_allow_html=True)

    if selected_customer != "No match found":
        customer_row = dashboard_df[dashboard_df["CustomerID"] == selected_customer].iloc[0]

        with lookup_col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f"### Customer {selected_customer}")

            st.markdown(risk_badge(customer_row["risk_tier"]), unsafe_allow_html=True)
            st.markdown(
                f'<span class="pill">{customer_row.get("customer_segment", "Unknown Segment")}</span>',
                unsafe_allow_html=True
            )

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Inactivity Probability", format_pct(customer_row["predicted_inactive_probability"]))
            p2.metric("Recency", f'{format_num(customer_row.get("recency", np.nan))} days')
            p3.metric("Frequency", format_num(customer_row.get("frequency", np.nan)))
            p4.metric("Monetary Value", format_money(customer_row.get("monetary_value", np.nan)))

            if HAS_PLOTLY:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(customer_row["predicted_inactive_probability"]) * 100,
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#7c3aed"},
                        "steps": [
                            {"range": [0, 40], "color": "#dcfce7"},
                            {"range": [40, 75], "color": "#fef3c7"},
                            {"range": [75, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "#111827", "width": 4},
                            "thickness": 0.75,
                            "value": SELECTED_THRESHOLD * 100
                        }
                    },
                    title={"text": "Inactivity Risk Score"}
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Why Is This Customer At Risk?")

            drivers = explain_customer(customer_row, medians)

            for driver in drivers:
                st.markdown(
                    f"""
                    <div class="driver-box">
                        <div class="driver-title">{driver["driver"]} — {driver["impact"]}</div>
                        <div class="driver-text">{driver["insight"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with detail_col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Recommended Marketing Action")

            st.markdown(f"**Action:** {customer_row.get('recommended_action', 'N/A')}")
            st.markdown(f"**Offer Level:** {customer_row.get('offer_level', 'N/A')}")
            st.markdown(f"**Campaign Objective:** {customer_row.get('campaign_objective', 'N/A')}")
            st.markdown(f"**Primary KPI:** {customer_row.get('primary_kpi', 'N/A')}")
            st.markdown(f"**Cadence:** {customer_row.get('cadence', 'N/A')}")

            st.markdown(
                f"""
                <div class="message-box">
                    <b>Subject:</b> {customer_row.get("subject_line", "N/A")}<br>
                    <b>Preheader:</b> {customer_row.get("preheader", "N/A")}<br><br>
                    <b>Message:</b><br>
                    {customer_row.get("message_template", "N/A")}<br><br>
                    <b>CTA:</b> {customer_row.get("cta", "N/A")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Predict New Customer
# =========================

with tabs[2]:
    st.subheader("Predict New Customer Risk")

    st.markdown(
        """
        Enter customer attributes below to simulate a new customer profile. 
        The model will estimate whether the customer is likely to become inactive in the next 90 days.
        """
    )

    with st.form("manual_prediction_form"):
        input_values = {}

        st.markdown("### Customer Behavior Inputs")

        cols = st.columns(3)

        for i, col in enumerate(numeric_cols):
            default_value = float(medians.get(col, 0))
            with cols[i % 3]:
                input_values[col] = st.number_input(
                    label=col.replace("_", " ").title(),
                    min_value=0.0,
                    value=max(default_value, 0.0),
                    step=1.0
                )

        if "primary_country" in feature_cols:
            selected_country = st.selectbox(
                "Primary Country",
                options=countries if countries else ["United Kingdom"],
                index=0
            )
            input_values["primary_country"] = selected_country

        submitted = st.form_submit_button("Predict Inactivity Risk")

    if submitted:
        input_row = {}

        for col in feature_cols:
            if col in input_values:
                input_row[col] = input_values[col]
            elif col in medians:
                input_row[col] = medians[col]
            else:
                input_row[col] = countries[0] if countries else "United Kingdom"

        input_df = pd.DataFrame([input_row])[feature_cols]

        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = int(probability >= SELECTED_THRESHOLD)
        risk_tier = assign_risk_tier(probability)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Prediction Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Inactivity Probability", format_pct(probability))
        r2.metric("Prediction", "Inactive / At Risk" if prediction == 1 else "Active / Lower Risk")
        r3.markdown(risk_badge(risk_tier), unsafe_allow_html=True)

        simulated_row = pd.Series(input_row)
        simulated_row["predicted_inactive_probability"] = probability
        simulated_row["risk_tier"] = risk_tier
        simulated_row["customer_segment"] = "Manual Input"

        st.markdown("### Risk Driver Explanation")

        drivers = explain_customer(simulated_row, medians)

        for driver in drivers:
            st.markdown(
                f"""
                <div class="driver-box">
                    <div class="driver-title">{driver["driver"]} — {driver["impact"]}</div>
                    <div class="driver-text">{driver["insight"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Marketing Actions
# =========================

with tabs[3]:
    st.subheader("Marketing Recommendations")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Recommended Action Summary")

    action_summary = data["action_summary"]

    if action_summary is not None:
        st.dataframe(action_summary, use_container_width=True)
    else:
        summary = filtered_df.groupby(["recommended_action", "offer_level", "primary_kpi"]).agg(
            customers=("CustomerID", "count"),
            avg_inactivity_probability=("predicted_inactive_probability", "mean"),
            avg_monetary_value=("monetary_value", "mean"),
            total_monetary_value=("monetary_value", "sum"),
            avg_campaign_priority_score=("campaign_priority_score", "mean")
        ).reset_index().sort_values("avg_campaign_priority_score", ascending=False)

        st.dataframe(summary, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Campaign Target List")

    campaign_cols = [
        "CustomerID",
        "customer_segment",
        "risk_tier",
        "predicted_inactive_probability",
        "monetary_value",
        "campaign_priority_score",
        "recommended_action",
        "offer_level",
        "subject_line",
        "preheader",
        "cta",
        "cadence",
        "primary_kpi"
    ]

    campaign_cols = [c for c in campaign_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[campaign_cols]
        .sort_values("campaign_priority_score", ascending=False)
        .head(200),
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Segments
# =========================

with tabs[4]:
    st.subheader("Customer Segments")

    segment_summary = data["segment_summary"]

    if segment_summary is not None:
        s1, s2 = st.columns([1, 1])

        with s1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Segment Size")

            if HAS_PLOTLY:
                fig = px.bar(
                    segment_summary,
                    x="customers",
                    y="customer_segment",
                    orientation="h",
                    color="inactive_rate",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(segment_summary.set_index("customer_segment")["customers"])

            st.markdown("</div>", unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Inactivity Rate by Segment")

            if HAS_PLOTLY:
                fig = px.bar(
                    segment_summary.sort_values("inactive_rate"),
                    x="inactive_rate",
                    y="customer_segment",
                    orientation="h",
                    text=segment_summary.sort_values("inactive_rate")["inactive_rate"].map(lambda x: f"{x:.1%}"),
                    color="inactive_rate",
                    color_continuous_scale="Reds"
                )
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(segment_summary.set_index("customer_segment")["inactive_rate"])

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Segment Summary Table")
        st.dataframe(segment_summary, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("segment_summary.csv not found. Please run the segmentation notebook first.")


# =========================
# Model & Explainability
# =========================

with tabs[5]:
    st.subheader("Model Performance & Explainability")

    m1, m2 = st.columns([1, 1])

    with m1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Model Comparison")

        model_results = data["model_results"]

        if model_results is not None:
            st.dataframe(model_results, use_container_width=True)

            if HAS_PLOTLY and "roc_auc" in model_results.columns:
                fig = px.bar(
                    model_results.sort_values("roc_auc"),
                    x="roc_auc",
                    y="model_name",
                    orientation="h",
                    text="roc_auc",
                    color="roc_auc",
                    color_continuous_scale="Purples"
                )
                fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("model_results_90d.csv not found.")

        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Feature Importance")

        feature_importance = data["feature_importance"]

        if feature_importance is not None:
            top_features = feature_importance.head(15)

            if HAS_PLOTLY:
                fig = px.bar(
                    top_features.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Purples"
                )
                fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(top_features, use_container_width=True)
        else:
            st.info("feature_importance_90d.csv not found. Run the explainability notebook to add this.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Model Decision Threshold")

    st.markdown(
        f"""
        The model uses a selected decision threshold of **{SELECTED_THRESHOLD:.2f}**.
        This threshold was chosen for the retention use case because it prioritizes catching more at-risk customers.
        A lower threshold increases recall, which is useful when missing inactive customers may be more costly than sending additional outreach.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# AI Advisor
# =========================

with tabs[6]:
    st.subheader("Generative AI Retention Advisor")

    st.markdown(
        """
        Ask business questions about customer inactivity, marketing actions, campaign targeting, 
        or message strategy. The advisor uses model outputs, customer segments, risk scores, 
        and retention recommendations as context.
        """
    )

    advisor_col1, advisor_col2 = st.columns([1, 1.4])

    with advisor_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Advisor Settings")

        advisor_mode = st.radio(
            "Advisor mode",
            options=["Local advisor", "OpenAI advisor if API key is available"],
            index=0
        )

        model_name = st.text_input(
            "OpenAI model name",
            value="gpt-4.1-mini",
            help="Only used if OpenAI advisor mode is selected and an API key is available."
        )

        advisor_scope = st.radio(
            "Question scope",
            options=["Selected customer", "Portfolio / campaign strategy"],
            index=0
        )

        selected_ai_customer = None

        if advisor_scope == "Selected customer":
            customer_ids = dashboard_df["CustomerID"].sort_values().unique().tolist()

            customer_search = st.text_input(
                "Search Customer ID for AI Advisor",
                placeholder="Example: 12346"
            )

            if customer_search:
                advisor_matches = [cid for cid in customer_ids if customer_search in cid]
            else:
                advisor_matches = customer_ids[:100]

            selected_ai_customer = st.selectbox(
                "Select customer",
                options=advisor_matches if advisor_matches else ["No match found"],
                key="ai_customer_select"
            )

        st.markdown("### Try asking")

        example_question = st.selectbox(
            "Example questions",
            options=[
                "Why is this customer at risk?",
                "What offer should we send this customer?",
                "Write a marketing message for this customer.",
                "How does this customer compare to loyal customers?",
                "Which campaign audience should we prioritize?",
                "Summarize the best retention strategy for this segment."
            ]
        )

        user_question = st.text_area(
            "Your question",
            value=example_question,
            height=120
        )

        ask_button = st.button("Ask AI Advisor", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

    with advisor_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Advisor Response")

        if ask_button:
            customer_row = None
            customer_context = None

            if advisor_scope == "Selected customer":
                if selected_ai_customer == "No match found":
                    st.warning("No matching customer found.")
                    st.stop()

                customer_row = dashboard_df[
                    dashboard_df["CustomerID"] == selected_ai_customer
                ].iloc[0]

                customer_context = build_customer_context(customer_row, medians)

            portfolio_context = build_portfolio_context(filtered_df)

            if advisor_mode == "OpenAI advisor if API key is available":
                ai_response = call_openai_advisor(
                    question=user_question,
                    customer_context=customer_context,
                    portfolio_context=portfolio_context,
                    model_name=model_name
                )

                if ai_response is None:
                    ai_response = local_ai_advisor_response(
                        question=user_question,
                        customer_row=customer_row,
                        filtered_df=filtered_df,
                        medians=medians
                    )
                    st.info("No OpenAI API key found, so the dashboard used the local advisor.")
            else:
                ai_response = local_ai_advisor_response(
                    question=user_question,
                    customer_row=customer_row,
                    filtered_df=filtered_df,
                    medians=medians
                )

            st.markdown(ai_response)

            st.markdown("---")
            st.caption(
                "Advisor output should be reviewed by a marketer before use. "
                "The recommendation is based on model outputs, segment rules, and available customer features."
            )
        else:
            st.info("Choose a question and click **Ask AI Advisor**.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### What the Advisor Uses as Context")

    st.markdown(
        """
        The advisor is grounded in:

        - Customer inactivity prediction score
        - Risk tier
        - RFM-based customer segment
        - Recency, frequency, monetary value, product diversity, and lifetime
        - Model explanation rules
        - Recommended retention action
        - Offer level, subject line, CTA, cadence, and KPI
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)