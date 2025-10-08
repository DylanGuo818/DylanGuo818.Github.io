---
title: Regression Practice (Titanic)
subtitle: Beginner-friendly logistic regression walkthrough with pandas, seaborn, and scikit-learn
image: https://via.placeholder.com/1200x800?text=Regression+Practice
alt: Logistic regression on the Titanic dataset

caption:
  title: Logistic Regression Practice
  subtitle: Titanic dataset walkthrough
  thumbnail: https://via.placeholder.com/600x400?text=Thumbnail
---

Use this area to describe your project. **Markdown** supported. This entry converts the provided PDF into a GitHub/Jekyll-friendly Markdown page with YAML front matter, a clear narrative, and step-by-step, beginner-oriented code that mirrors the original content.

{:.list-inline}

- Date: October 2025
- Client: UTS 22577 (Regression Practice)
- Category: Machine Learning (Classification)

---

## Goal
Predict the **Survived** outcome on the Titanic dataset using **logistic regression**, while demonstrating data cleaning, imputation, feature engineering, model training, evaluation, and inference on new data.

> **Note on data leakage**: Do **not** use the target `Survived` when imputing features like `Age`. Using the label to fill feature values leaks future information into training and leads to overly optimistic metrics and poor generalization.

---

## 1) Setup & Imports
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
```

---

## 2) Load Data & Backup
```python
# Load training data (expects train.csv in the working directory)
data = pd.read_csv('train.csv')

# Quick peek
data.head()

# Keep a backup copy
data_original = data.copy()
```

---

## 3) Impute `Age` (without using `Survived`)
Fill missing `Age` by the **mean age within groups** of `Sex` and `Pclass`.

```python
# Impute Age using group means of Sex x Pclass (no target leakage)
data['Age'] = (
    data.groupby(['Sex', 'Pclass'], group_keys=False)['Age']
        .apply(lambda s: s.fillna(s.mean()))
)
```

### Before vs After Imputation (Histogram)
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.histplot(data_original['Age'], ax=axes[0], kde=True, color='red')
axes[0].set_title('Age (before imputation)')

sns.histplot(data['Age'], ax=axes[1], kde=True, color='green')
axes[1].set_title('Age (after imputation)')

plt.tight_layout()
plt.show()
```

> **Chart questions**  
> • What does `plt.subplots` do? *Hint: check the Matplotlib docs or run `help(plt.subplots)` in a notebook.*  

---

## 4) One-hot encode categoricals
Convert categorical variables to numbers so the model can learn from them.

```python
# One-hot encode Sex and Embarked (drop_first to avoid dummy trap)
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Columns created include: Sex_male, Embarked_Q, Embarked_S (depending on data)
```

> **Practical question**  
> *What did `pd.get_dummies` do? How would you find out?*  
> Try: `help(pd.get_dummies)` or `pd.get_dummies.__doc__`.

---

## 5) Features & Target
```python
# Select features for the first model
feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                'Sex_male', 'Embarked_Q', 'Embarked_S']

X = data[feature_cols]
y = data['Survived']
```

---

## 6) Train / Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## 7) Train Logistic Regression
```python
model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)
```

> **A bit of theory**  
> `max_iter` is the cap on optimization steps. If the solver converges early, it stops; otherwise it stops at this cap even if not fully converged.

> **Practical question**  
> *Why use logistic regression here?* Because the outcome (`Survived`) is **binary**, and logistic regression is a simple, interpretable baseline for binary classification.

---

## 8) Evaluate (Accuracy & Confusion Matrix)
```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:
', conf_mat)
```

> **Reminder**  
> • A **confusion matrix** summarizes correct vs. incorrect predictions for classification.  
> • **Accuracy** is the fraction of correct predictions.

---

## 9) Feature Importance (Coefficients)
```python
coef = model.coef_[0]
importance_df = (
    pd.DataFrame({'Feature': X.columns, 'Importance': coef})
      .sort_values(by='Importance', ascending=False)
)

plt.figure(figsize=(7, 4))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Logistic Coefficients)')
plt.tight_layout()
plt.show()
```

> **Interpreting signs**  
> • **Positive** coefficient → higher values increase probability of the positive class (`Survived = 1`).  
> • **Negative** coefficient → higher values decrease that probability.

---

## 10) Inference on New (Unlabeled) Test Data
Prepare `test.csv` in the same way as training data, line up its columns with `X`, then predict.

```python
# Load test data
test_data = pd.read_csv('test.csv')

# Impute Age using Sex x Pclass means
test_data['Age'] = (
    test_data.groupby(['Sex', 'Pclass'], group_keys=False)['Age']
             .apply(lambda s: s.fillna(s.mean()))
)

# Fill any missing Fare using Sex x Pclass means
test_data['Fare'] = (
    test_data.groupby(['Sex', 'Pclass'], group_keys=False)['Fare']
             .apply(lambda s: s.fillna(s.mean()))
)

# One-hot encode
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Ensure same columns/order as training features
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Predict
test_pred = model.predict(test_data)

# Attach predictions
pred_df = test_data.copy()
pred_df['Survived_predicted'] = test_pred

# Quick count of predicted classes
pred_df['Survived_predicted'].value_counts()
```

---

## 11) Practical Exercise: Add `FamilySize`
Create a new feature and rebuild the model.

```python
# Feature engineering
data = data.copy()
data['FamilySize'] = data['SibSp'] + data['Parch']

# Features for the second model
feat2 = ['Pclass', 'Age', 'FamilySize', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
X2 = data[feat2]
y2 = data['Survived']

# Split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

# Train
model2 = LogisticRegression(max_iter=200)
model2.fit(X2_train, y2_train)

# Evaluate
y2_pred = model2.predict(X2_test)
acc2 = accuracy_score(y2_test, y2_pred)
cm2 = confusion_matrix(y2_test, y2_pred)

print(f'Accuracy (with FamilySize): {acc2:.4f}')
print('Confusion Matrix (with FamilySize):
', cm2)
```

> **Stretch**: Repeat step 10 using `model2` and the same preprocessing for `test.csv` (remember to compute `FamilySize` there too) to compare predictions.

---

## FAQ-style prompts from the original worksheet
- *What does `plt.subplots` do? How would you find out more about it?* → Use `help(plt.subplots)` or consult the Matplotlib docs.  
- *What did `pd.get_dummies` do?* → It converted categorical columns into one-hot encoded indicator columns so the model can ingest them.  
- *Why are we using a logistic regression model?* → Binary target, interpretable baseline, fast to train, good starting point.

---

## Attributions
Converted to Markdown and restructured for GitHub/Jekyll from the provided worksheet.
