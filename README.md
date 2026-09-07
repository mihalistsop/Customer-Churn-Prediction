# 📞 Telco Customer Churn Prediction & Deployment

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest%20%2F%20XGBoost-green.svg)

## 📌 Business Problem
Customer churn is one of the biggest challenges in the telecom industry. The goal of this project is to build a machine learning pipeline that predicts which customers are likely to cancel their service. By identifying "at-risk" customers, the company can take proactive measures (discounts, loyalty programs) to retain them.

## 🚀 Live Demo
You can try the interactive predictor here:  
👉 **[https://churn-prediction-michaeltsop.streamlit.app/](https://churn-prediction-michaeltsop.streamlit.app/)**

---

## 📊 Key Insights from EDA
* **Contract Type:** Customers on "Month-to-month" contracts have a significantly higher churn rate compared to one or two-year contracts.
* **Monthly Charges:** Higher monthly bills (especially between $70-$100) are strong indicators of churn.
* **Tenure:** New customers are much more likely to leave; the risk drops significantly after the first 12-24 months.



## 🛠️ Tech Stack & Methodology
* **Data Cleaning:** Handled missing values in `TotalCharges` and performed feature encoding.
* **Class Imbalance:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset (Churned vs. Retained).
* **Modeling:** Trained and compared **Random Forest** and **XGBoost** classifiers.
* **Hyperparameter Tuning:** Optimized models using `RandomizedSearchCV` to maximize **Recall**.
* **Deployment:** Created an interactive web dashboard using **Streamlit**.

## 📈 Model Performance
| Metric (Class: Churn) | Initial Model | Tuned Model (SMOTE) |
| :--- | :--- | :--- |
| **Accuracy** | 79% | 76% |
| **Recall** | 48% | **59%** |
| **F1-Score** | 0.54 | 0.57 |

*Note: We prioritized **Recall** over Accuracy to ensure we capture as many potential leavers as possible.*



## 📂 Project Structure
* `streamlit_app.py`: The web application code.
* `rf_churn_model.joblib`: The trained Random Forest model.
* `model_columns.pkl`: The saved feature schema for consistent predictions.
* `requirements.txt`: Necessary libraries for deployment.
