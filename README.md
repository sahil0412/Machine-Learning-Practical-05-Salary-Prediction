# Adult Census Income prediction

## Installation Guide
1. Clone or Fork the Project
2. Create a Virtual Enviroment
3. go to same virtual enviroment and write below cmd
4. pip install -r requirements.txt


### 1. Project Description
#### A. Problem Statement

This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.

#### B. Tools and Libraries
Tools<br><br>
a.Python<br>
b.Jupyter Notebook<br>
c. Flask<br>
d. HTML<br>
e. Render<br>
f. GitHub

Libraries<br><br>
a.Pandas<br>
b.Scikit Learn<br>
c.Numpy<br>
d.Seaborn<br>
e.Matpoltlib<br>

### 2. Data Collection
There are a total of 48,842 rows of data, and 3,620 with missing values, leaving 45,222 complete rows.
The dataset provides 14 input variables that are a mixture of categorical, ordinal, and numerical data types. The complete list of variables is as follows:

1. Age.
2. Workclass.
3. Final Weight.
4. Education.
5. Education Number of Years.
6. Marital-status.
7. Occupation.
8. Relationship.
9. Race.
10. Sex.
11. Capital-gain.
12. Capital-loss.
13. Hours-per-week.
14. Native-country.

There are two class values ‘>50K‘ and ‘<=50K‘, meaning it is a binary classification task. The classes are imbalanced, with a skew toward the ‘<=50K‘ class label.<br>
The dataset contains `?` instead of Null values, we need to replace them with NaN values

### 3. EDA
#### A.Data Cleaning
Null values for numerical columns can be handled via median stargegy and for categorical columns mode can be applied<br>
All null values are part of categorical columns so we used `mode` as medium to impute them.

#### B. Feature Engineering
No outliers are present in the data

#### C. Data Normalization
Normalization (min-max Normalization)<br>
In this approach we scale down the feature in between 0 to 1<br>
There are different ways by which we can convert categorical cols to numerical cols such as ->
1. Label Encoding -> Label Encoding converts categorical values to numerical values by assigning a unique integer to each category. This is useful for ordinal categorical variables.
2. One-Hot Encoding -> One-Hot Encoding creates binary columns for each category. This is useful for nominal categorical variables.
3. Ordinal Encoding -> Ordinal Encoding is used when the categorical variable has a natural order but no fixed spacing between the categories.
4. Frequency Encoding -> Frequency Encoding replaces categories with their frequency in the dataset.
5. Target Encoding -> Target Encoding replaces categories with the mean of the target variable for each category. This is more advanced and should be used carefully to avoid data leakage.

We use Label encoding to convert categorical cols to numerical cols in our use-case

We have numerical column where we can apply min-max Normalization.<br>

### 4. Choosing Best ML Model
List of the model that we can use for our problem<br>
a. XGBoost model<br>

### 5. Model Creation
So,using a XGBoost Classifier, we got good accuracy of 87%, we can Hyperparameter tuning for best accuracy.

Algorithm that can be used for Hyperparameter tuning are :-

a. GridSearchCV<br>
b. RandomizedSearchCV<br>

Main parameters used by XGBoostClassifier Algorithm are :-
a. n_estimators -> Number of trees in the forest.<br>
b. learning_rate -> Controls the step size shrinkage during each boosting iteration. Lower values make the model more robust but require more boosting rounds.<br>
c. max_depth -> The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain fewer than min_samples_split samples.<br>
d. gamma -> Minimum loss reduction required to make a further partition on a leaf node of the tree.<br>
e. lambda (or reg_lambda) -> The minimuL2 regularization term on weights. Increases model's robustness to noise.<br>
f. alpha (or reg_alpha) -> L1 regularization term on weights. Adds penalty for large coefficients.<br>

### 6. Model Deployment
After creating model ,we integrate that model with beautiful UI. for the UI part we used HTML and Flask. We have added extra validation check also so that user doesn't enter Incorrect data. Then the model is deployed on render

### 7. Model Conclusion

Model predict 86% accurately on test data.

### 8. Project Innovation
a. Easy to use<br>
b. Open source<br>
c. Best accuracy<br>
d. GUI Based Application

### 9. Limitation And Next Step
Limitation are :-<br>
a. Mobile Application<br>
b. Accuracy can be improved more<br>
d. Feature is limited

Next Step are :-<br>
a. we can work on mobile application<br>

## Deployable Link
https://machine-learning-practical-04-boston.onrender.com/predict
git remote set-url https://github.com/sahil0412/Machine-Learning-Practical-05-Salary-Prediction.git