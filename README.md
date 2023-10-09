# Machine learning - Fraud Detection
This project contains some machine learning models for a credit card fraud detection study.

Requirements:

* Python 3;
* pandas;
* numpy
* scikit-learn

## Development environment
Create a virtual environment

```shell
# Linux
sudo apt-get install python3-venv    # If needed
python3 -m venv .venv
source .venv/bin/activate

# macOS
python3 -m venv .venv
source .venv/bin/activate
```

Update the pip
```shell
python3 -m pip install --upgrade pip
```

Install python dependencies
```shell
pip3 install -r requirements.txt
```

## Data set

The machine learning models created in thos project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) available on [Kaggle](https://www.kaggle.com/).

## Machine learning models used

* Logistic Regression;
* XGBoost

## Confusion Matrix

Matrix that summarizes the performance of a machine learning model on a set of test data.
Used to measure the performance of classification models, which aim to predict a categorical label for each input instance. The matrix displays the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) produced by the model on the test data.
