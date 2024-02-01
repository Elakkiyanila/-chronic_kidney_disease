# chronic_kidney_disease
# Introduction
This project aims to develop a machine learning model for predicting the likelihood of an individual having Chronic Kidney Disease (CKD) based on various health attributes and medical test results. Additionally, the project includes data preprocessing and exploratory data analysis. The dataset used for this project consists of features such as age, blood pressure, urine characteristics, blood glucose, and more.
# Data Preprocessing
The dataset is loaded and preprocessed before model training. Duplicate records are removed, missing values are imputed using the KNN algorithm, and categorical variables are encoded.
+ We need to import the necessary libraries/modules for the code to work.
```
# Target Label Encoding
target_label_encoder = LabelEncoder()
y = target_label_encoder.fit_transform(y)

# Splitting into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

# Feature Scaling
standard_feature_scaler = StandardScaler()
X_train = pd.DataFrame(standard_feature_scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(standard_feature_scaler.transform(X_test), columns=X.columns)
```
# Logistic Regression
Logistic Regression is applied as one of the classification models. Here are the evaluation metrics for the Logistic Regression model:
