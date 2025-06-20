# 💳 Credit Card Approval Prediction using Ensemble Learning

This project uses ensemble machine learning techniques to predict whether a credit card application should be approved or not based on applicant demographics and credit behavior.

## 📌 Problem Statement

Banks receive numerous credit card applications, many of which need to be filtered based on customer profiles and credit history. The goal is to automate the approval process using a machine learning model that minimizes risk and boosts approval accuracy.

## 📂 Dataset

- **application_record.csv**: Contains applicant information such as income, employment, age, education, family, etc.
- **credit_record.csv**: Monthly repayment status and delinquency information.

**Target Variable:** Binary label (1: Bad credit history, 0: Good credit history)

## 🔍 Steps Performed

1. **Data Merging**: Merged application and credit data based on `ID`
2. **Feature Engineering**:
   - Converted days to age/employment years
   - Created target label from credit status
   - One-hot encoded categorical features
3. **Modeling**:
   - Used Random Forest, Gradient Boosting, XGBoost
   - Combined via VotingClassifier (soft voting)
4. **Evaluation**:
   - Accuracy, Confusion Matrix, Classification Report

## 🧠 ML Models Used

- Random Forest
- XGBoost
- Gradient Boosting
- Voting Classifier (Ensemble)

## 📈 Evaluation Metrics

- Accuracy
- Precision/Recall/F1
- Confusion Matrix

## 📁 Files

- `Credit_Card_Approval_Ensemble_Model.ipynb`: Full notebook
- `application_record.csv`: Input dataset
- `credit_record.csv`: Input dataset
- `README.md`: Project description

## ✅ Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
