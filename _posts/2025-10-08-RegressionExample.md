---
layout: home
title: Titanic Survival Prediction
subtitle: Logistic Regression with Feature Engineering and Evaluation
image: https://raw.githubusercontent.com/BlackrockDigital/startbootstrap-agency/master/src/assets/img/portfolio/01-full.jpg
alt: Titanic logistic regression model

caption:
  title: Titanic Survival
  subtitle: Logistic Regression
  thumbnail: https://raw.githubusercontent.com/BlackrockDigital/startbootstrap-agency/master/src/assets/img/portfolio/01-thumbnail.jpg
---

This project demonstrates a complete logistic regression pipeline using the Titanic dataset. **Markdown** supported. The notebook walks through data preprocessing, model training, evaluation, and prediction on unseen test data.

{:.list-inline}

- Date: October 2025  
- Client: UTS Data Science Practice  
- Category: Classification

---

This project explores how to build a logistic regression model to predict survival on the Titanic. It includes:

- Data cleaning and imputation
- Avoiding data leakage
- Feature engineering
- Model training and evaluation
- Prediction on test data

---

- Missing `Age` values were filled using the average age grouped by `Sex` and `Pclass`.
- Missing `Fare` values in the test set were filled similarly.
- Categorical variables (`Sex`, `Embarked`) were converted to dummy variables using `pd.get_dummies`.

---

Using the `Survived` column to fill missing values like `Age` would cause **data leakage**, leading to overly optimistic model performance and poor generalization.

---

1. **Feature Selection**  
   Selected features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex_male`, `Embarked_Q`, `Embarked_S`

2. **Train-Test Split**  
   Used `train_test_split` to divide data into training and testing sets.

3. **Model Training**  
   Trained a `LogisticRegression` model with `max_iter=400`.

4. **Prediction & Evaluation**  
   - Accuracy: **~0.85**
   - Confusion Matrix:
     ```
     [[46  8]
      [ 5 31]]
     ```

5. **Feature Importance**  
   Visualized using a bar plot. Positive coefficients increase survival probability; negative ones decrease it.

---

- Preprocessed test data to match training format
- Predicted survival outcomes
- Added predictions back to the test DataFrame

---

Created a new feature `FamilySize = SibSp + Parch` and retrained the model using:

- `Pclass`, `Age`, `FamilySize`, `Fare`, `Sex_male`, `Embarked_Q`, `Embarked_S`

New model accuracy: **~0.73**

---

This project is a great example of applying logistic regression to a real-world dataset with practical preprocessing and evaluation steps. Ideal for beginners learning classification models in Python.
