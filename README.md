# 📊 Telecom Customer Churn Prediction & Analytics

An end-to-end machine learning and business analytics application built with Python, Scikit-learn, Pandas, Plotly, and Streamlit to analyze telecom customer churn and predict customers who are likely to leave.

The application combines exploratory data analysis, machine learning, interactive visualizations, customer-level churn prediction, model evaluation, and business insights in one interactive dashboard.

## 🚀 Live Demo
 
[View the Telecom Churn Analytics App](https://telecom-customer-churn-analytics.onrender.com/)

---

##  Project Overview

Customer churn is an important business problem for telecom companies because losing existing customers can significantly affect revenue and long-term growth.

This project uses historical telecom customer data to:

- Analyze customer churn patterns
- Identify factors associated with customer churn
- Build a machine learning model to predict churn
- Estimate the probability that a customer will churn
- Evaluate machine learning model performance
- Identify important features influencing predictions
- Provide business recommendations for customer retention

The final solution is deployed as an interactive Streamlit web application on Render.

---

##  Project Objectives

The main objectives of this project are to:

1. Understand customer churn patterns.
2. Explore customer behavior and service usage.
3. Identify factors associated with customer churn.
4. Build a machine learning model for churn prediction.
5. Evaluate the performance of the model.
6. Provide an interactive customer churn prediction tool.
7. Translate analytical findings into actionable business insights.

---

## 📂 Dataset

The project uses the Telecom Customer Churn dataset.

### Dataset Size

- Rows: 2,666
- Columns: 20
- Target Variable: Churn

The target variable indicates whether a customer churned:

- False → Customer retained
- True → Customer churned

### Main Features

The dataset contains information about:

- Customer state
- Account length
- Area code
- International plan
- Voice mail plan
- Number of voicemail messages
- Daytime usage
- Evening usage
- Night usage
- International usage
- Customer service calls
- Customer churn status

---

##  Data Analysis

Before building the machine learning model, the dataset was examined for:

- Missing values
- Duplicate records
- Data types
- Categorical variables
- Numerical variables
- Target class distribution
- Outliers
- Negative values
- Feature relationships
- Potential data quality issues

The dataset contains no missing values and no duplicate records.

The churn target is imbalanced, with significantly more retained customers than churned customers.

---

## 🤖 Machine Learning

A Random Forest Classifier was used to predict customer churn.

### Why Random Forest?

Random Forest was selected because it:

- Works well with structured/tabular data
- Can capture nonlinear relationships
- Handles multiple features effectively
- Provides feature importance
- Performs well without requiring complex mathematical assumptions

### Model Pipeline

The project uses a Scikit-learn Pipeline containing:

1. Data preprocessing
2. Missing-value handling
3. Categorical feature encoding
4. Random Forest classification

Categorical variables are processed using:

- Most-frequent imputation
- One-hot encoding

Numerical variables are processed using:

- Median imputation

The model also uses class balancing to help address the imbalance between churned and retained customers.

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Why multiple metrics?

Accuracy alone can be misleading when the target classes are imbalanced.

For a churn prediction problem, Recall, Precision, F1 Score, and ROC-AUC provide additional information about how effectively the model identifies customers who are likely to churn.

The model performance can be viewed directly from the Model Performance page of the application.

---

##  Application Features
### 🏠 Dashboard

Provides an overview of customer retention and churn.

The dashboard includes:

- Total Customers
- Churned Customers
- Retained Customers
- Overall Churn Rate
- Churn Distribution

---

### 📈 Churn Analysis

Interactive visualizations are used to explore relationships between customer characteristics and churn.

Examples include:

- International Plan vs Churn
- Customer Service Calls vs Churn

These visualizations help identify customer groups that may require additional attention.

---

### 🤖 Predict Churn

Users can enter individual customer information and receive a churn prediction.

The application provides:

- Churn prediction
- Churn probability
- Customer risk indication

The prediction model uses the same preprocessing pipeline used during model training.

---

### 📊 Model Performance

The application provides an interactive model evaluation page containing:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

###  Feature Importance

The application displays the features that contribute most strongly to the Random Forest model's predictions.

This helps users understand which variables the model considers important when predicting churn.

> Feature importance indicates the features used by the model to make predictions. It does not necessarily mean that a feature directly causes customer churn.

---

### 💡 Business Insights

The application translates the analytical findings into potential business actions.

Examples include:

- Identifying high-risk customers
- Monitoring customers with frequent customer service interactions
- Investigating churn patterns among customers with international plans
- Using churn probability to support customer retention strategies

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Plotly | Interactive data visualization |
| Streamlit | Web application |
| Joblib | Model serialization |
| Git & GitHub | Version control |
| Render | Application deployment |

---

## 📁 Project Structure

```text
telecom-customer-churn-analytics/
│
├── app.py
├── churn-bigml-80.csv
├── churn_model.pkl
├── churn_analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── ...
