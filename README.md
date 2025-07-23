🚢 TITANIC SURVIVAL PREDICTION - Logistic Regression from Scratch

This project aims to understand model fundamentals without relying on external ML libraries.

This project demonstrates a complete machine learning pipeline to predict survival on the Titanic dataset using Logistic Regression implemented from scratch in NumPy. The workflow includes data preprocessing, visualization, model training, and evaluation.

1) 📁DATASET
The dataset used is the classic Titanic dataset from Kaggle. It contains information on the passengers aboard the Titanic, such as age, sex, ticket fare, class, etc., along with their survival outcome.

2) 🧰 TECHNOLOGIES USED
Python
NumPy for numerical operations
Pandas for data handling
Matplotlib & Seaborn for data visualization
No external ML libraries like scikit-learn were used for modeling

3)🧪 FEATURES USED
After data cleaning and feature selection using correlation thresholding, the final features used for model training were:
Pclass
Age
SibSp
Fare
Sex_male (encoded from Sex)
Embarked_S (encoded from Embarked)

4)⚙️ WORKFLOW
1. Data Preprocessing
Dropped unnecessary columns: PassengerId, Name, Ticket, Cabin
Handled missing values:
Filled Age with the median
Filled Embarked with the mode
One-hot encoded categorical variables
Normalized features using standardization

2. Exploratory Data Analysis
Used boxplots to visualize outliers grouped by Sex and Pclass
Plotted a heatmap of the correlation matrix

3. Model Implementation
Logistic Regression implemented manually using NumPy
Trained using gradient descent
Tracked loss function over 2000 iterations


### 📊 MODEL PERFORMANCE

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 0.8324  |
| Precision | 0.7742  |
| Recall    | 0.7500  |
| F1 Score  | 0.7619  |


6)📉 Training Loss
A plot of the loss over iterations shows smooth convergence, indicating stable training.

7)📦 How to Run
Download the Titanic dataset CSV from Kaggle and place it in your working directory.
Update the path in pd.read_csv() accordingly.

Run the script in a Jupyter Notebook or any Python IDE.


Feature selection was based on a correlation threshold (0.01) relative to the Survived column.

