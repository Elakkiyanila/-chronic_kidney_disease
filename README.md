# chronic_kidney_disease
# Introduction
This project aims to develop a machine learning model for predicting the likelihood of an individual having Chronic Kidney Disease (CKD) based on various health attributes and medical test results. Additionally, the project includes data preprocessing and exploratory data analysis. The dataset used for this project consists of features such as age, blood pressure, urine characteristics, blood glucose, and more.
# Data Preprocessing
The dataset is loaded and preprocessed before model training. Duplicate records are removed, missing values are imputed using the KNN algorithm, and categorical variables are encoded.
+ We need to import the necessary libraries/modules for the code to work.
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
```
# Logistic Regression
Logistic Regression is applied as one of the classification models. Here are the evaluation metrics for the Logistic Regression model:
```
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
logistic.score(X_train,y_train)
y_pred = logistic.predict(X_test)
# Print evaluation metrics for Logistic Regression
print("\nEvaluating Logistic Regression Model:")
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('F1-score:', metrics.f1_score(y_test, y_pred))
print('Classification Report:\n', metrics.classification_report(y_test, y_pred))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_pred))

# Plot confusion matrix for Logistic Regression
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
```
# Decision Tree
Decision Tree classifier is employed with its corresponding evaluation metrics and confusion matrix:
```
logistic = LogisticRegression()
logistic.fit(X_train,y_train)
logistic.score(X_train,y_train)
y_pred = logistic.predict(X_test)
# Print evaluation metrics for Logistic Regression
print("\nEvaluating Logistic Regression Model:")
print('Precision:', metrics.precision_score(y_test, y_pred))
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
print('Recall:', metrics.recall_score(y_test, y_pred))
print('F1-score:', metrics.f1_score(y_test, y_pred))
print('Classification Report:\n', metrics.classification_report(y_test, y_pred))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, y_pred))

# Plot confusion matrix for Logistic Regression
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True)
plt.title('Confusion Matrix - Logistic Regression')
plt.show()
```
# Random Forest
Random Forest classifier is utilized, along with its evaluation metrics and confusion matrix:
```
# Random Forest
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)
# Print evaluation metrics for Random Forest
print("\nEvaluating Random Forest Model:")
print('Precision:', metrics.precision_score(y_test, rf_predictions))
print('Accuracy:', metrics.accuracy_score(y_test, rf_predictions))
print('Recall:', metrics.recall_score(y_test, rf_predictions))
print('F1-score:', metrics.f1_score(y_test, rf_predictions))
print('Classification Report:\n', metrics.classification_report(y_test, rf_predictions))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, rf_predictions))
print('ROC AUC Score:', roc_auc_score(y_test, rf_predictions))
# Plot confusion matrix for Random Forest
sns.heatmap(metrics.confusion_matrix(y_test, rf_predictions), annot=True)
plt.title('Confusion Matrix - Random Forest')
plt.show()
```
# Gradient Boosting
Gradient Boosting classifier is implemented, and its evaluation metrics along with the confusion matrix are provided.
```
# Create a Gradient Boosting classifier
gradient_boosting = GradientBoostingClassifier(random_state=0)
# Train the classifier
gradient_boosting.fit(X_train, y_train)
# Make predictions on the test set
gb_predictions = gradient_boosting.predict(X_test)
# Print evaluation metrics for Gradient Boosting
print("\nEvaluating Gradient Boosting Model:")
print('Precision:', metrics.precision_score(y_test, gb_predictions))
print('Accuracy:', metrics.accuracy_score(y_test, gb_predictions))
print('Recall:', metrics.recall_score(y_test, gb_predictions))
print('F1-score:', metrics.f1_score(y_test, gb_predictions))
print('Classification Report:\n', metrics.classification_report(y_test, gb_predictions))
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test, gb_predictions))
print('ROC AUC Score:', roc_auc_score(y_test, gb_predictions))

# Plot confusion matrix for Gradient Boosting
sns.heatmap(metrics.confusion_matrix(y_test, gb_predictions), annot=True)
plt.title('Confusion Matrix - Gradient Boosting')
plt.show()
```
# Conclusion

The evaluation results for both the Gradient Boosting and Random Forest models demonstrate excellent performance in predicting Chronic Kidney Disease (CKD) likelihood:

- **Precision**, **Recall**, and **F1-score** are all consistently high, indicating that the models are effectively identifying both positive and negative cases of CKD with minimal misclassification.
- The **accuracy** of around 97.5% suggests that the models are making accurate predictions on the test dataset.
- The **confusion matrices** show minimal false positives and false negatives, further confirming the models' robustness in distinguishing between CKD and non-CKD cases.
- The **ROC AUC score** of approximately 0.975 indicates a high level of discrimination capability of the models.

In conclusion, both the Gradient Boosting and Random Forest models exhibit strong predictive performance for CKD likelihood based on the given dataset. These models could be valuable tools for healthcare professionals in identifying individuals at risk of CKD, enabling early intervention and treatment. However, further validation on diverse datasets and clinical trials may be necessary before deploying these models in real-world healthcare settings.

