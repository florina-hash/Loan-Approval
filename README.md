# Loan Approval Prediction System

A predictive Machine Learning model built to automate the loan eligibility process based on customer details provided during the application process. This project was developed as a hands-on project during my **Certification Course on Python with Machine Learning**.

## 📌 Project Overview
The goal of this project is to classify whether a loan application will be **Approved (Y)** or **Rejected (N)** based on various applicant parameters such as income, credit history, education, marital status, and more. 

Using historical customer data, we perform Exploratory Data Analysis (EDA), handle data preprocessing, and train a **Logistic Regression** model to make accurate predictions.

## 🗂️ Dataset Features
The model utilizes the `loan.csv` dataset, which contains the following applicant attributes:
*   `Loan_ID`: Unique applicant ID
*   `Gender`: Male / Female
*   `Married`: Applicant marital status (Yes / No)
*   `Dependents`: Number of dependents
*   `Education`: Applicant education (Graduate / Under Graduate)
*   `Self_Employed`: Self-employed (Yes / No)
*   `ApplicantIncome`: Applicant income
*   `CoapplicantIncome`: Coapplicant income
*   `LoanAmount`: Loan amount in thousands
*   `Loan_Amount_Term`: Term of loan in months
*   `Credit_History`: Credit history meets guidelines (1.0 = Good, 0.0 = Bad)
*   `Property_Area`: Urban / Semi-Urban / Rural
*   `Loan_Status`: Target Variable (Y = Approved, N = Rejected)

## 🛠️ Tech Stack & Libraries
*   **Language:** Python
*   **Data Manipulation:** `pandas`
*   **Data Visualization:** `seaborn`, `matplotlib`
*   **Machine Learning:** `scikit-learn` (Logistic Regression, LabelEncoder, train_test_split)

## 📈 Workflow & Implementation
1. **Data Cleaning:** Handled missing values using dropna techniques and checked for duplicates.
2. **Exploratory Data Analysis (EDA):** Visualized feature distributions, credit history impacts, and correlation heatmaps.
3. **Data Preprocessing:** Categorical features encoded into numerical elements using `LabelEncoder`.
4. **Model Training:** Split the dataset (80% Training, 20% Testing) and trained a `LogisticRegression` model.
5. **Interactive Interface:** Created a command-line interface block allowing real-time data inputs to test custom applicant scenarios.

## 📊 Model Performance
The trained Logistic Regression model achieves an impressive baseline performance:
*   **Accuracy Score:** ~82.3%
*   **Precision (Class 1 - Approved):** 80%
*   **Recall (Class 1 - Approved):** 100%

### Confusion Matrix:
```text
[[11 17]
 [ 0 68]]
