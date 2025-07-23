TITANIC SURVIVAL PERDICTION USING LOGISTIC REGRESSION
This repository contains a Python notebook that demonstrates step-by-step application of logistic regression to predict survival on the Titanic dataset. Using pandas, Seaborn, and Matplotlib, this project explores, visualizes, and models the classic Titanic dataset to provide an introduction to machine learning and binary classification.

Table of Contents

Overview
Dataset
Libraries Used
Exploratory Data Analysis
Data Preprocessing
Model Training
How to Run
Result
Directory Structure

1)Overview
    This project aims to predict which passengers survived the Titanic disaster using logistic regression. We:
    
    Clean and preprocess the raw data,
    
    Explore and visualize feature importance,
    
    Fit a logistic regression model,
    
    Evaluate model performance.

2)Dataset
    Source: Titanic dataset from Kaggle
    
    Columns included:
    
    PassengerId
    
    Survived (target variable)
    
    Pclass
    
    Name
    
    Sex
    
    Age
    
    SibSp

    Parch
    
    Ticket
    
    Fare
    
    Cabin
    
    Embarked

3)Libraries Used
    numpy
    
    pandas
    
    matplotlib
    
    seaborn
    
    scikit-learn (for model training)

4)Exploratory Data Analysis
    Visual distribution of Age, Fare, and Survival rate by class and gender.
    
    Assessment of missing values and correlations.

5)Data Preprocessing
    Handling missing values (e.g., Age, Cabin).
    
    Encoding categorical variables (Sex, Embarked) using one-hot encoding.
    
    Selecting features for modeling.

6)Model Training
    Logistic regression is used as the core model for binary classification (Survived/Not Survived).
    
    Model fitting, validation, and prediction steps are shown in the notebook.

7)How to Run
    Clone this repository.
    
    Place the titanic.csv file in the project folder.
    
    Run the logistic_regression-checkpoint.ipynb notebook using Jupyter Notebook or JupyterLab.
    
    bash
    pip install numpy pandas matplotlib seaborn scikit-learn
8)Results
    Model performance is displayed using accuracy and confusion matrix.
    
    Feature impacts are discussed based on model coefficients and exploratory analysis.

9)Directory Structure
    text
    ├── logistic_regression-checkpoint.ipynb
    ├── titanic.csv
    ├── README.md

10)Acknowledgments
    Kaggle for making the Titanic dataset publicly available.
    
    Python open-source community for the libraries and documentation.
