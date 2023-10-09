"""
    Machine learning classification for a cradit card fraud detection.
    Training a XGBoost Model
"""

# Data manipulation and analysis
import pandas as pd

# Numerical operations
import numpy as np

# Used to split the dataset into training and testing subsets
from sklearn.model_selection import train_test_split

# XGBoost is a popular and efficient open-source implementation of the gradient boosted trees algorithm.
# Gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target
# variable by combining the estimates of a set of simpler, weaker models.
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from collections import Counter

# Performance Metrics
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# reads a CSV file named 'creditcard.csv' into a Pandas DataFrame named df
df = pd.read_csv("creditcard.csv")

# assigned the 'Class' column from the DataFrame df, which typically represents the target variable
# (0 for non-fraudulent transactions and 1 for fraudulent transactions).
y = df["Class"]
# X is assigned the DataFrame df with three columns dropped: 'Class', 'Amount', and 'Time'.
# This step separates the features used for training the machine learning model.
X = df.drop(["Class", "Amount", "Time"], axis=1)

"""
Spliting the dataset into training and testing subsets.

test_size=0.1 indicates that 10% of the data will be used as the testing set
while the remaining 90% will be the training set.

random_state=42 sets a random seed for reproducibility.

stratify=y ensures that the class distribution in the training and testing sets is similar to 
the original dataset, which is important for imbalanced datasets like credit card fraud detection.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

X_train

"""
The code uses NumPy to count the number of occurrences of '1' in y_train and y_test, 
which corresponds to the number of fraudulent transactions in each set.
"""
print("Fraud in y_train:", len(np.where(y_train == 1)[0]))
print("Fraud in y_test", len(np.where(y_test == 1)[0]))

"""'
Create a XGBoost classifier model for binary classification tasks. 

max_depth and scale_pos_weight, are two of the many hyperparameters 
to customize the behavior of the XGBoost classifier.

max_depth:
This parameter controls the maximum depth of each tree in the boosting process. 
It is an important hyperparameter for controlling the complexity of the individual trees in the ensemble.

scale_pos_weight:
This parameter is used to address class imbalance in binary classification problems. 
It is typically set to the ratio of the number of negative class samples to the number of 
positive class samples in the training data.
"""
model = xgb.XGBClassifier(max_depth=5, scale_pos_weight=100)
model.fit(X_train, y_train)

# Uses the trained model to make predictions
# The predicted values are stored in the y_pred
y_pred = model.predict(X_test)
y_pred

# Precision is the proportion of correctly predicted fraudulent instances among all instances
# predicted as fraud
# TP / TP + FP

precision_score(y_test, y_pred)

# Recall is the proportion of the fraudulent instances that are successfully predicted
# TP / TP + FN
recall_score(y_test, y_pred)

# F1-score is the harmonic balance of precision and recall (can be weighted more towards P or R if need be)
# F = 2 * (Precision * Recall)/(Precision + Recall)
f1_score(y_test, y_pred)

# AUROC/AUC = Area under the Receiver Operating Characteristic curve
# plot the TPR (Recall) and FPR at various classification thresholds
# FPR = FP / FP + TN
# Good measure of overall performance
roc_auc_score(y_test, y_pred)

# AUPRC = Area under the Precision-Recall curve
# Better alternative to AUC as doesn't include TN which influences the scores significantly in highly imbalanced data
# calculates the area under the curve at various classification thresholds
average_precision_score(y_test, y_pred)

# Classification report summarizes the classification metrics at the class and overall level

print(classification_report(y_test, y_pred))

print("Original dataset shape %s" % Counter(y_train))
sm = SMOTE(sampling_strategy=1, random_state=42, k_neighbors=5)
# sampling_strategy = ratio of minority to majority after resampling
# k_neighbors = defines neighborhood of samples to use to generate synthetic samples. Decrease to reduce false positives.
X_res, y_res = sm.fit_resample(X_train, y_train)
print("Resampled dataset shape %s" % Counter(y_res))

# confusion matrix
LABELS = ["Normal", "Fraud"]
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()
