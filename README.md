# Customer Churn Prediction using Machine Learning

## Overview

Customer churn is a major challenge for telecommunications companies. When customers discontinue their services, businesses lose potential revenue and valuable customer relationships.

This project focuses on predicting whether a telecommunications customer is likely to leave a service using Machine Learning classification algorithms.

The project includes Exploratory Data Analysis (EDA), data cleaning, preprocessing, model training, hyperparameter tuning, and performance evaluation to understand customer churn patterns and identify a suitable predictive model.

## Objectives

- Understand and explore the Telco Customer Churn dataset.
- Identify important factors influencing customer churn.
- Clean and preprocess customer data.
- Apply and compare different Machine Learning classification algorithms.
- Evaluate model performance using classification metrics.
- Identify a suitable model for customer churn prediction.

## Dataset

**Dataset:** Telco Customer Churn Dataset

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains customer information related to telecommunications services, including:

- Customer demographics
- Account information
- Services subscribed to
- Contract type
- Tenure
- Monthly charges
- Total charges
- Customer churn status

**Target Variable:** `Churn`

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

### 1. Exploratory Data Analysis (EDA)

- Explore dataset structure, columns, and data types.
- Check missing values and blank entries.
- Generate descriptive statistics.
- Analyze customer churn distribution.
- Visualize relationships between customer features and churn.

### 2. Data Cleaning and Preprocessing

- Convert `TotalCharges` into numeric format.
- Handle missing values.
- Remove the unnecessary `customerID` column.
- Encode categorical variables using Label Encoding and One-Hot Encoding.
- Apply Standard Scaling to numerical features.
- Split the dataset into training and testing sets using stratified sampling.

### 3. Model Implementation

Four supervised Machine Learning classification algorithms were implemented.

**1. Logistic Regression**

A classification algorithm that estimates the probability of a customer churning using a logistic function.

**2. K-Nearest Neighbors (KNN)**

A distance-based algorithm that predicts customer churn by considering the nearest data points.

**3. Decision Tree**

A tree-based algorithm that makes predictions through a sequence of decision rules based on customer features.

**4. Random Forest**

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance and generalization.

### 4. Hyperparameter Tuning

Hyperparameter tuning was performed to improve model performance and compare different configurations.

- Experimented with different K values for KNN.
- Tuned Decision Tree parameters such as `max_depth` and `min_samples_leaf`.
- Tuned Random Forest parameters such as `n_estimators`, `max_depth`, and `max_features`.

### 5. Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Additional evaluation techniques included:

- Confusion Matrix
- True Positive (TP)
- False Positive (FP)
- True Negative (TN)
- False Negative (FN)
- ROC Curve
- Model Performance Comparison

## Key Findings

The exploratory analysis and feature importance results highlighted several factors associated with customer churn:

- **Contract Type:** Customers with month-to-month contracts are more likely to leave.
- **Tenure:** Customers with shorter service periods show higher churn rates.
- **Monthly Charges:** Higher monthly charges are associated with increased churn.
- **Internet Services:** Internet service type and the availability of additional services can influence churn.
- **Payment Method:** Customers using electronic check payments tend to have higher churn rates.

## Model Comparison and Recommendation

Based on the model comparison and evaluation discussion:

**Random Forest** is the recommended candidate for deployment because it generally provides strong predictive performance and useful feature importance information.

**Logistic Regression** is a strong alternative when model interpretability is important. Its coefficients make it easier to understand how different customer features influence churn predictions.

The final model selection should also consider the business cost of false positives and false negatives.

## Business Insights

Customer churn prediction can help telecommunications companies identify customers who are at risk of leaving their services.

Businesses can use these predictions to:

- Develop targeted customer retention strategies.
- Identify customers who may need additional support.
- Offer suitable incentives and promotional campaigns.
- Reduce avoidable customer losses.
- Improve customer satisfaction and long-term relationships.

For this business problem, recall is particularly important because missing an actual churner can result in lost revenue. However, precision also matters when retention offers involve significant costs.

## Limitations and Future Improvements

- Apply k-fold cross-validation for more reliable model evaluation.
- Explore advanced boosting algorithms such as XGBoost, LightGBM, and CatBoost.
- Perform classification threshold tuning based on business requirements.
- Use SHAP values for better model interpretability.
- Incorporate customer behavior, usage trends, and support interaction data.
- Explore additional techniques to handle class imbalance.
