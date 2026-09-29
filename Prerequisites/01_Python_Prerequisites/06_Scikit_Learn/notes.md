# Python Prerequisites: 06. Scikit-Learn Architecture & The Estimator API

Scikit-Learn (`sklearn`) is the industry-standard Machine Learning library in Python. Its worldwide popularity is due to one primary design decision: **The Unified Estimator API**. 

Whether you are scaling features, encoding text, training a simple Linear Regression, or tuning a complex Random Forest, **every single tool in Scikit-Learn uses the exact same core methods**.

---

## 1. What is Scikit-Learn?

Scikit-Learn is an open-source Python library that provides efficient implementations of:
- **Supervised Learning Algorithms:** Linear models, Decision Trees, SVMs, Naive Bayes, Ensembles.
- **Unsupervised Learning Algorithms:** K-Means clustering, PCA dimensionality reduction.
- **Data Preprocessing & Cleaning:** Scalers, Imputers, Categorical Encoders.
- **Model Evaluation:** Train/test splits, Cross-Validation, ROC-AUC, Confusion Matrices.

---

## 2. The Estimator Hierarchy: Transformers vs. Predictors

Every tool in Scikit-Learn is called an **Estimator**. Estimators fall into two major categories:

```text
                               ESTIMATOR
                 (Any object that learns from data via .fit())
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
            TRANSFORMER                              PREDICTOR
     (Preprocesses & modifies data)             (Makes ML predictions)
     Examples:                                  Examples:
     - StandardScaler                           - LogisticRegression
     - OneHotEncoder                            - DecisionTreeClassifier
     - SimpleImputer                            - KNeighborsClassifier
                 │                                       │
                 ▼                                       ▼
     Methods:                                   Methods:
     - .fit(X)                                  - .fit(X, y)
     - .transform(X)                            - .predict(X)
     - .fit_transform(X)                        - .predict_proba(X)
```

---

## 3. The 4 Fundamental Methods Explained

### 1. `fit()` — "Study and Learn"
- Calculates and stores necessary statistics or parameters from data.
- **For Transformers:** Learns means ($\mu$), standard deviations ($\sigma$), medians, or categories.
- **For Predictors:** Learns mathematical weights ($w$) and biases ($b$).
- **Syntax:** `transformer.fit(X_train)` or `model.fit(X_train, y_train)`

### 2. `transform()` — "Apply the Learned Rules"
- Uses the statistics learned during `fit()` to convert raw data into clean numbers.
- **Syntax:** `X_test_scaled = transformer.transform(X_test)`
- *Note:* Predictor models do NOT have a `transform()` method.

### 3. `fit_transform()` — "Learn and Convert in One Step"
- A convenient shortcut that runs `fit()` followed immediately by `transform()`.
- Used **only on training data**: `X_train_scaled = transformer.fit_transform(X_train)`.

### 4. `predict()` — "Forecast Output for New Samples"
- Uses learned model weights to output target predictions ($y$) on new feature data ($X$).
- **Syntax:** `y_pred = model.predict(X_test)`.

---

## 4. Walking Through a Concrete End-to-End Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Split data FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. TRANSFORMER: Scale features
scaler = StandardScaler()
# Learn mean/std from train AND transform train
X_train_scaled = scaler.fit_transform(X_train)
# Use train mean/std to transform test (NEVER call fit on test!)
X_test_scaled = scaler.transform(X_test)

# 3. PREDICTOR: Train model
model = LogisticRegression()
# Learn weights from scaled train data
model.fit(X_train_scaled, y_train)

# 4. PREDICT on test data
y_pred = model.predict(X_test_scaled)
```

---

## 5. Composition Tools: ColumnTransformer and Pipeline

As your projects grow, Scikit-Learn provides two essential orchestrators:

- **`ColumnTransformer` (WHERE):** Applies different preprocessing steps to different columns in parallel (e.g., scaling numerical features while one-hot encoding categorical features).
- **`Pipeline` (WHEN / ORDER):** Chains preprocessing and the model into a single automated sequence:
  ```text
  Raw Data ──► ColumnTransformer ──► Model ──► Prediction
```

*(Note: For deep practical guides on these tools, see [05_Column_Transformer](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/05_Column_Transformer/) and [06_Machine_Learning_Pipeline](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/06_Machine_Learning_Pipeline/).)*
