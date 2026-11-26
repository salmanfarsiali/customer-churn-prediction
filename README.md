# Customer Churn Prediction using Machine Learning

## Overview

Customer churn is a major challenge for telecommunications companies. When customers discontinue their services, businesses lose potential revenue and customer relationships.

This project uses Machine Learning classification algorithms to predict whether a customer is likely to leave a telecommunications service.

The project includes Exploratory Data Analysis (EDA), data cleaning, feature encoding, feature scaling, model training, hyperparameter tuning, and model evaluation.

## Objectives

- Understand and explore the Telco Customer Churn dataset.
- Identify important factors influencing customer churn.
- Clean and preprocess the dataset.
- Apply different Machine Learning classification algorithms.
- Compare model performance using classification metrics.
- Identify a suitable model for customer churn prediction.

## Dataset

**Dataset:** Telco Customer Churn Dataset

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information, including:

- Customer demographics
- Account information
- Services subscribed to
- Contract type
- Monthly and total charges
- Customer churn status

The target variable is `Churn`:

- `0` — Customer did not churn
- `1` — Customer churned

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Jupyter Notebook

## Project Structure

```text
customer-churn-prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── customer_churn_prediction.ipynb
```

## Project Workflow

### 1. Data Understanding and Exploration
- Load and inspect the dataset.
- Check data types and missing values.
- Generate summary statistics.
- Explore customer churn distribution.
- Visualize relationships between customer features and churn.

### 2. Data Cleaning and Preprocessing
- Convert `TotalCharges` into numeric values.
- Handle missing values.
- Remove the `customerID` column.
- Encode categorical variables.
- Apply feature scaling to numerical features.
- Split the dataset into training and testing sets.

### 3. Model Implementation

Four classification algorithms were implemented:

**Logistic Regression**

A classification algorithm used to estimate the probability of customer churn.

**K-Nearest Neighbors (KNN)**

A distance-based algorithm that predicts churn using the nearest customer observations.

**Decision Tree**

A tree-based algorithm that makes predictions through a sequence of decision rules.

**Random Forest**

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

### 4. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

Hyperparameter tuning was also performed for KNN, Decision Tree, and Random Forest.

## Key Findings

The analysis highlighted several factors associated with customer churn:

- Customers with month-to-month contracts have higher churn rates.
- Customers with shorter tenure are more likely to leave.
- Higher monthly charges are associated with increased churn.
- Internet service type and additional services can influence churn.
- Customers using electronic check payments show higher churn rates.

Random Forest was recommended as the deployment candidate based on the notebook's model comparison discussion. Logistic Regression is also a useful alternative when model interpretability is important.

## Business Insights

Predicting customer churn can help telecommunications companies identify customers who may leave their services.

Businesses can use these predictions to:

- Improve customer retention strategies.
- Identify customers who need additional support.
- Develop targeted offers and incentives.
- Reduce avoidable customer losses.
- Improve long-term customer relationships.

## Limitations and Future Improvements

- Apply k-fold cross-validation for more reliable model evaluation.
- Explore advanced models such as XGBoost, LightGBM, and CatBoost.
- Perform decision-threshold tuning.
- Explore SHAP values for better model interpretability.
- Include customer behavior and support-ticket data for richer predictions.
