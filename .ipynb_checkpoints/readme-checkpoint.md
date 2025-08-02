Internship Tasks – Machine Learning & Data Analysis

This repository contains internship work focused on solving real-world problems using data science techniques. Each task involves data preprocessing, exploratory analysis, model training, evaluation, and insightful reporting.

---

Task Overview

Task 1: Exploring and Visualizing a Simple Dataset

Objective:  
Understand and visualize the structure and relationships in the famous Iris Dataset.

Highlights:
- Loaded data using `pandas` and `seaborn`
- Explored shape, columns, and basic statistics
- Visualized data distributions using histograms and box plots
- Analyzed feature relationships with scatter plots

Task 2: Credit Risk Prediction

Objective:  
Predict whether a loan applicant is likely to default using classification models.

Dataset:  
Loan Default Dataset (from Kaggle)

Approach:
- Cleaned missing values
- Encoded categorical features
- Visualized income, education, and loan amount
- Trained Logistic Regression and Decision Tree classifiers
- Evaluated models using accuracy and confusion matrix

Insight:  
Logistic Regression provided better accuracy in this task.

Task 3: Customer Churn Prediction (Bank Customers)

Objective:  
Identify customers likely to leave the bank using machine learning.

Dataset:  
Churn Modelling Dataset

Steps Taken:
- Label encoded binary features; One-hot encoded multi-category columns
- Scaled numerical features using `StandardScaler`
- Trained three models: Logistic Regression, Decision Tree, Random Forest
- Evaluated models using Accuracy, Precision, Recall, and F1 Score

Insight:  
Though Random Forest had the best accuracy, the Decision Tree performed relatively better in identifying churned customers due to slight improvement in recall.

Task 4: Predicting Insurance Claim Amounts

Objective: 
Estimate medical insurance charges using regression modeling.

Dataset:  
Medical Cost Personal Dataset

Steps:
- Encoded categorical variables (`sex`, `smoker`, `region`)
- Detected and removed outliers using the IQR method
- Visualized correlations and relationships using heatmaps and pairplots
- Trained a Linear Regression model
- Evaluated performance using MAE, RMSE, and MAPE

Results:
-MAE: ~1380.80  
- RMSE: ~2581.54  
- MAPE:21.22%

Insight:  
Linear Regression offers a basic estimation, which has a significant amount to error but can be improved using advanced models (e.g., Gradient Boosting or SVR).


## 🛠️ Tools & Libraries

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

