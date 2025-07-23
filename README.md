# 🚢 Titanic Survival Prediction — Logistic Regression from Scratch

This project implements **Logistic Regression from scratch using NumPy** to predict passenger survival in the Titanic dataset. It walks through data preprocessing, encoding, visualization, model training, and performance evaluation — all without using external machine learning libraries.

---

## 🧰 **Technologies Used**

| **Tool**      | **Purpose**                           |
|---------------|----------------------------------------|
| Python        | Core programming language              |
| NumPy         | Numerical operations                   |
| Pandas        | Data handling                          |
| Matplotlib    | Data visualization                     |
| Seaborn       | Enhanced data visualization            |
| No sklearn    | Model implemented fully from scratch   |

---

## 🗂️ **Workflow**

1. **Data Preprocessing**
   - Dropped unnecessary columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
   - Filled missing values:
     - `Age` → median
     - `Embarked` → mode
   - One-hot encoded categorical variables
   - Standardized features manually

2. **Exploratory Data Analysis**
   - Boxplots for outliers grouped by `Sex` and `Pclass`
   - Heatmap of correlation matrix
   - Feature selection based on correlation with `Survived`

3. **Model Implementation**
   - Implemented **Logistic Regression** from scratch with:
     - Sigmoid activation
     - Binary cross-entropy loss
     - Gradient descent optimization

4. **Evaluation**
   - Tracked loss over iterations
   - Manually calculated:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
     - Confusion Matrix

---

## 📊 **Model Performance**

| **Metric**   | **Score** |
|--------------|-----------|
| Accuracy     | 0.8324    |
| Precision    | 0.7742    |
| Recall       | 0.7500    |
| F1 Score     | 0.7619    |

---

## 📉 **Loss Over Iterations**

Loss was calculated and plotted at each iteration (2000 in total), showing convergence and model learning over time.

---

## 🧪 **Final Features Used**

After encoding and correlation filtering, these features were used in model training:

| **Feature**     | **Description**                          |
|------------------|------------------------------------------|
| Pclass           | Passenger class (1st, 2nd, 3rd)          |
| Age              | Age of passenger                         |
| SibSp            | Number of siblings/spouses aboard        |
| Fare             | Ticket fare                              |
| Sex_male         | Binary encoded gender                    |
| Embarked_S       | Binary encoded port of embarkation (Southampton) |

---

## 🔍 **Evaluation Metrics (Manual Calculation)**

| **Metric**   | **Definition**                            |
|--------------|-------------------------------------------|
| Accuracy     | Correct predictions / Total samples       |
| Precision    | True Positives / Predicted Positives      |
| Recall       | True Positives / Actual Positives         |
| F1 Score     | Harmonic mean of Precision and Recall     |

---

## 🔎 **Confusion Matrix**

A confusion matrix was created manually and visualized using Seaborn's heatmap.

|              | **Predicted 0** | **Predicted 1** |
|--------------|------------------|------------------|
| **Actual 0** | True Negative     | False Positive   |
| **Actual 1** | False Negative    | True Positive    |

---

## 📦 **How to Run**

1. Download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data).
2. Save the CSV to your local directory.
3. Update the path in the script:
   ```python
   pd.read_csv('C:\\Users\\HP\\Downloads\\titanic.csv')
