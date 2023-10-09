"""
    Machine learning classification for a cradit card fraud detection.
    Performs the logistic regression for binary classification.
    
    Logistic regression is a statistical and supervised machine learning algorithm used 
    for binary classification.
"""

# Data manipulation and analysis
import pandas as pd

# Numerical operations
import numpy as np

# Used to split the dataset into training and testing subsets
from sklearn.model_selection import train_test_split

# Imports the LogisticRegression class, a statistical method used for binary classification tasks
from sklearn.linear_model import LogisticRegression

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

"""
Training the logistic regression model on a training dataset
The class_weight parameter allows specify weights for the classes in your dataset.
Class 0 is assigned a weight of 1, and class 1 is assigned a weight of 50
"""
model = LogisticRegression(class_weight={0: 1, 1: 50})
model.fit(X_train, y_train)

# Uses the trained model to make predictions
# The predicted values are stored in the y_pred
y_pred = model.predict(X_test)
y_pred

# Confusion matrix
LABELS = ["Normal", "Fraud"]
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion matrix")
plt.ylabel("True class")
plt.xlabel("Predicted class")
plt.show()
