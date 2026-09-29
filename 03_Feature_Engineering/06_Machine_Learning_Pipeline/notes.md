# Module 03: Feature Engineering — Machine Learning Pipeline (`sklearn.pipeline.Pipeline`)

In machine learning, training a model is never just about calling `model.fit()`. Before any raw data can be ingested by an algorithm, it must travel through a sequence of data cleaning, imputation, encoding, and scaling steps.

If you handle these steps manually as loose pieces of code, your project quickly becomes messy, fragile, and vulnerable to **data leakage**.

Scikit-Learn's **`Pipeline`** solves this by bundling all your preprocessing steps and your machine learning model into a **single, organized, and reusable workflow**.

---

## 1. What is a Machine Learning Pipeline?

### Understand It First: The Problem with Loose Code

Suppose you receive raw data from a customer database to build a classification model (e.g., predicting whether a customer will buy a product).

Before the algorithm can learn anything, the data must go through several operations:

```text
Raw Data
   ↓
1. Handle Missing Values (e.g., fill missing ages with median)
   ↓
2. Encode Categorical Data (e.g., convert "Male"/"Female" to 0/1)
   ↓
3. Scale Numerical Features (e.g., standardize Salary to mean 0, variance 1)
   ↓
4. Train Machine Learning Model (e.g., Logistic Regression)
   ↓
5. Make Predictions
```

Normally, a beginner might write separate code for every single operation:

```python
# The manual, fragmented way:
imputer.fit_transform(...)
encoder.fit_transform(...)
scaler.fit_transform(...)
model.fit(...)
```

### Why This Loose Approach Becomes a Headache:
1. **Too Many Moving Parts:** You have to keep track of 3 or 4 separate transformer objects in memory.
2. **Order Matters:** If you accidentally scale before imputing, or encode after splitting incorrectly, the code breaks or produces silent errors.
3. **Repeated Work on Test Data:** When test data arrives, you must remember to call `transform()` with every single individual object in the exact same order.
4. **Production Deployment Nightmare:** If you want to deploy your model to an API, you cannot just save the model. You also have to save the imputer, the encoder, and the scaler, and write code to manually route new incoming data through all three before predicting.
5. **Data Leakage Risk:** It is dangerously easy to accidentally fit a scaler or imputer on the full dataset, corrupting your evaluation.

### The Solution: Enter the Pipeline

Instead of keeping preprocessing and modeling as separate tasks, **a Pipeline connects them into a single, unified chain**.

> **Formal Definition:**  
> A **Machine Learning Pipeline** is a sequence of data transformers followed by an optional final estimator (model) connected together, where the output of each step automatically serves as the input to the next step.

---

## 2. A Simple Pipeline Example

Think of an assembly line in an automobile factory:

```text
Raw Metal Sheets ──► Stamping Press ──► Welding Robot ──► Paint Booth ──► Finished Car
```

No worker carries half-finished doors across the factory floor manually. The assembly line carries the parts forward automatically from station to station.

In Scikit-Learn:

```text
                        THE PIPELINE
 ┌────────────────────────────────────────────────────────┐
 │  Raw Data ──► Preprocessing ──► Scaler ──► ML Model    │ ──► Predictions
 └────────────────────────────────────────────────────────┘
```

### Comparing the Two Approaches:

| Aspect | Without Pipeline (Manual) | With Pipeline |
| :--- | :--- | :--- |
| **Object Management** | Manage `imputer`, `encoder`, `scaler`, and `model` separately | Manage only **one** object: `pipeline` |
| **Training** | Call `fit_transform()` 3 times, then `model.fit()` | Call **`pipeline.fit(X_train, y_train)`** once |
| **Testing** | Call `transform()` 3 times, then `model.predict()` | Call **`pipeline.predict(X_test)`** once |
| **Deployment** | Save and load 4 different files | Save and load **1 file** |

> **Important Note:** A Pipeline does not invent new preprocessing math. It simply organizes existing transformers and models into an automated, sequential workflow.

---

## 3. Why Do We Need Pipelines?

Pipelines provide four major practical advantages in everyday Machine Learning:

### 1. Simplicity & Readability
Instead of 20 lines of repetitive slicing, fitting, and array transformation, your entire workflow is declared in one clean, readable block:

```text
Raw Data ──► SimpleImputer ──► StandardScaler ──► LogisticRegression
```

One command trains the entire chain: `pipeline.fit(X_train, y_train)`.

---

### 2. Consistency Across Data Splits
Every machine learning model must evaluate on:
- Training data
- Test data
- Future real-world production data

```text
Training Data   ──► [ Same Pipeline ] ──► Model Trained
Test / New Data ──► [ Same Pipeline ] ──► Clean Predictions
```

With a Pipeline, you are guaranteed that test data and new incoming production records receive the **exact same transformations in the exact same order** without writing custom cleaning scripts.

---

### 3. Preventing Data Leakage
**What is Data Leakage?**  
Data leakage occurs when information from outside the training dataset (such as test or validation data) influences the model during training.

*Example:* If you calculate the mean of `Age` using the entire dataset before splitting into train and test, the test set's mean has "leaked" into your training preprocessing. Your model looks unrealistically accurate during testing, but fails in production.

**How Pipeline Helps:**  
When you wrap your preprocessing and model inside a Pipeline, Scikit-Learn ensures that preprocessing parameters (means, standard deviations, categories) are learned **strictly from the training portion**. 

This is especially critical during **Cross-Validation**, where the pipeline guarantees that preprocessing is re-fit inside each training fold independently.

---

### 4. Effortless Deployment
In production, your model will receive raw, messy JSON requests from users (e.g., `{"Age": 28, "City": "Delhi"}`).

- **Without a Pipeline:** Your production server must load `imputer.pkl`, `scaler.pkl`, `encoder.pkl`, and `model.pkl`, run each step manually, and hope column order wasn't mixed up.
- **With a Pipeline:** Your server loads `pipeline.pkl` and calls `pipeline.predict(raw_data)`. Preprocessing and prediction happen in one seamless motion.

---

## 4. Pipeline Structure

A standard Scikit-Learn Pipeline consists of a list of sequential steps:

```text
Input Features (X)
        │
        ▼
   [ Step 1: Transformer ]  (e.g., SimpleImputer)
        │ (imputed array)
        ▼
   [ Step 2: Transformer ]  (e.g., StandardScaler)
        │ (scaled array)
        ▼
   [ Step 3: Estimator / Model ]  (e.g., LogisticRegression)
        │
        ▼
   Final Output (Predictions / Probabilities)
```

### The Golden Rule of Pipeline Steps:
- **Intermediate Steps:** Must all be **Transformers** (objects that implement both `fit()` and `transform()`, such as `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, or `ColumnTransformer`).
- **Final Step:** Can be a **Transformer** OR an **Estimator / Model** (such as `LogisticRegression`, `RandomForestClassifier`, etc.).

The output of Step 1 automatically becomes the input to Step 2, and the output of Step 2 becomes the input to Step 3.

---

## 5. Basic Scikit-Learn Pipeline Syntax

To create a Pipeline, import `Pipeline` from `sklearn.pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define the pipeline steps as a list of (name, object) tuples
pipeline = Pipeline(
    steps=[("scaler", StandardScaler()), ("model", LogisticRegression())]
)
```

### Understanding the Structure:
Each step in the list is a tuple containing **two parts**:
```text
('name_of_step', transformer_or_model_object)
```

- `'scaler'`: A string label of your choice to name the step.
- `StandardScaler()`: The transformer object.
- `'model'`: The string label for the final estimator.
- `LogisticRegression()`: The machine learning algorithm.

### Training and Predicting:
```python
# 1. Train the entire pipeline (scales X_train, then fits the model)
pipeline.fit(X_train, y_train)

# 2. Predict on test data (scales X_test using TRAIN statistics, then predicts)
predictions = pipeline.predict(X_test)
```

---

## 6. Pipeline + ColumnTransformer: The Dream Team

In our previous module, we learned about **`ColumnTransformer`**. Beginners often ask:  
*"Should I use `Pipeline` or `ColumnTransformer`?"*

The answer is: **You use both together! They solve different problems.**

```text
ColumnTransformer  ──►  Answers: "WHERE?" (Which columns get which preprocessing?)
Pipeline           ──►  Answers: "WHEN?"  (What sequence does the data travel through?)
```

```text
                  PIPELINE (The Overall Sequence)
                               │
                               ▼
                   [ ColumnTransformer ] (Step 1: Preprocessing)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Numerical Columns               Categorical Columns
        ['Age', 'Salary']               ['City', 'Gender']
               │                               │
               ▼                               ▼
        StandardScaler                   OneHotEncoder
               │                               │
               └───────────────┬───────────────┘
                               ▼
                   [ Combined Transformed Data ]
                               │
                               ▼
                   [ Final ML Model ] (Step 2: Estimation)
                   (e.g., LogisticRegression)
```

`ColumnTransformer` handles the column-specific routing, and `Pipeline` chains that preprocessing step directly to the machine learning model.

---

## 7. Complete Pipeline + ColumnTransformer Example

Let's look at a concrete, end-to-end example with mixed features:

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sample Data
df = pd.DataFrame(
    {
        "Age": [25, 30, 45, 35],
        "Salary": [50000, 70000, 120000, 85000],
        "Gender": ["Male", "Female", "Female", "Male"],
        "City": ["Delhi", "Mumbai", "Delhi", "Bangalore"],
        "Purchased": [0, 1, 1, 0],  # Target
    }
)

X = df[["Age", "Salary", "Gender", "City"]]
y = df["Purchased"]

# Step 1: Build ColumnTransformer for column-specific operations
preprocessor = ColumnTransformer(
    transformers=[
        ("scale_num", StandardScaler(), ["Age", "Salary"]),
        ("ohe_cat", OneHotEncoder(sparse_output=False), ["Gender", "City"]),
    ],
    remainder="drop",
)

# Step 2: Build Pipeline chaining Preprocessor -> Model
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)

# Step 3: Train everything in ONE call!
full_pipeline.fit(X, y)
```

Look how clean this is. The user only interacts with `full_pipeline`.

---

## 8. Handling Missing Values in a Pipeline

What if our dataset has missing values in both numerical and categorical columns?

- `Age`: Numerical with missing values $\longrightarrow$ Needs **Median Imputation** then **Scaling**.
- `City`: Categorical with missing values $\longrightarrow$ Needs **Most Frequent Imputation** then **One-Hot Encoding**.

Notice that a single column group may need a **sub-pipeline** of multiple steps!

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sub-Pipeline 1: For Numerical Columns (Impute -> Scale)
numerical_sub_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Sub-Pipeline 2: For Categorical Columns (Impute -> One-Hot Encode)
categorical_sub_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ]
)

# Combine both sub-pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_sub_pipeline, ["Age", "Salary"]),
        ("cat", categorical_sub_pipeline, ["Gender", "City"]),
    ]
)

# Full End-to-End Pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression()),
    ]
)
```

---

## 9. Full Realistic Pipeline Architecture

Here is the complete blueprint of a professional machine learning workflow:

```text
                           Raw Input Dataset (X)
                                     │
                                     ▼
                     [ Full Production Pipeline ]
                                     │
                                     ▼
                        [ ColumnTransformer ]
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
  Numerical Features                                      Categorical Features
  ['Age', 'Salary']                                       ['Gender', 'City']
         │                                                       │
         ▼                                                       ▼
 [ SimpleImputer (Median) ]                             [ SimpleImputer (Mode) ]
         │                                                       │
         ▼                                                       ▼
 [ StandardScaler ]                                     [ OneHotEncoder ]
         │                                                       │
         └───────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                        Combined Clean Numerical Array
                                     │
                                     ▼
                        [ LogisticRegression Model ]
                                     │
                                     ▼
                           Final Prediction (0 / 1)
```

Every single branch, transformer, and model is synchronized under one single command.

---

## 10. What Happens During `pipeline.fit()`?

When you execute:
```python
pipeline.fit(X_train, y_train)
```

Scikit-Learn steps through the pipeline sequentially:

```text
Step 1: Preprocessor
  ├── Slices X_train
  ├── Calls .fit() to learn medians, means, stds, and categories
  └── Calls .transform() to transform X_train into numbers
              │
              ▼ (processed array)
Step 2: Model (Final Step)
  └── Calls .fit() on the processed array and y_train to learn model weights
```

### What Each Component Learns During `fit()`:
- **`SimpleImputer`**: Learns the median/mode of each column.
- **`StandardScaler`**: Learns the mean ($\mu$) and standard deviation ($\sigma$).
- **`OneHotEncoder`**: Learns the unique categorical labels.
- **`LogisticRegression`**: Learns the feature coefficients ($w$) and intercept ($b$).

All learned parameters are stored inside the pipeline's respective step objects.

---

## 11. What Happens During `pipeline.predict()`?

When new test data or production data arrives:
```python
predictions = pipeline.predict(X_test)
```

The flow is fundamentally different from training:

```text
Step 1: Preprocessor
  └── Calls ONLY .transform() using the ALREADY-LEARNED training statistics!
      (Never calls .fit()!)
              │
              ▼ (processed array)
Step 2: Model (Final Step)
  └── Calls .predict() using the learned model weights
              │
              ▼
         Final Output
```

> **Crucial Rule:**  
> During `predict()`, preprocessing steps **never re-fit**. They strictly apply the parameters learned during `fit()` on the training data.

---

## 12. `fit_transform()` vs. `transform()` in a Pipeline

| Stage | What Scikit-Learn Calls on Transformers | Why? |
| :--- | :--- | :--- |
| **`pipeline.fit(X_train, y_train)`** | `fit_transform()` | Learns statistics from training data AND passes transformed numbers to the next step. |
| **`pipeline.predict(X_test)`** | `transform()` only | Preserves training statistics; ensures test data is scaled/encoded identically without data leakage. |

This automated separation is why Pipelines protect you against data leakage by default.

---

## 13. Pipeline vs. Manual Preprocessing: Code Comparison

### The Manual Way (Painful & Fragile):
```python
# Training
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
scaler.fit(X_train_imp)
X_train_scaled = scaler.transform(X_train_imp)
model.fit(X_train_scaled, y_train)

# Testing (Did you remember every step? Did you accidentally fit on test?)
X_test_imp = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imp)
y_pred = model.predict(X_test_scaled)
```

### The Pipeline Way (Clean & Bulletproof):
```python
# Training
pipeline.fit(X_train, y_train)

# Testing
y_pred = pipeline.predict(X_test)
```

Everything happens in two lines. The chance of human error drops to near zero.

---

## 14. Inspecting Pipeline Steps with `named_steps`

A Pipeline is not a black box. You can easily inspect any individual step using **`pipeline.named_steps`**:

```python
# Access the model step
model = pipeline.named_steps["classifier"]
print("Model coefficients:", model.coef_)

# Access the preprocessor step
preprocessor = pipeline.named_steps["preprocessor"]
print(
    "Encoded feature names:",
    preprocessor.named_transformers_["ohe_cat"].categories_,
)
```

`named_steps` works like a Python dictionary, allowing you to access model parameters, transformer categories, and feature names anytime.

---

## 15. Inspecting Configuration with `get_params()`

To see all configurable parameters across the entire pipeline, call:

```python
pipeline.get_params()
```

Parameters are identified using a **double underscore (`__`)** notation:
- `preprocessor__scale_num__with_mean`
- `classifier__C`
- `classifier__penalty`

This allows you to verify default settings and prepare for hyperparameter tuning.

---

## 16. Pipeline + Cross-Validation (Eliminating CV Leakage)

Cross-validation splits your data into $K$ folds (e.g., 5 folds). In each round, 4 folds train the model and 1 fold evaluates it.

### The Fatal Flaw of Preprocessing Outside Cross-Validation:
If you scale your entire dataset *before* calling `cross_val_score`:
```text
Full Dataset ──► Scaler.fit() ──► 5-Fold Cross-Validation  [DATA LEAKAGE!]
```
The scaler calculated its mean using the entire dataset, meaning validation folds leaked into the training process!

### The Correct Way with Pipeline:
```python
from sklearn.model_selection import cross_val_score

# Pass the PIPELINE directly into cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-Validation Accuracy:", scores.mean())
```

```text
Fold 1: Fit Scaler on Folds 2,3,4,5 ──► Transform Fold 1 ──► Evaluate Fold 1
Fold 2: Fit Scaler on Folds 1,3,4,5 ──► Transform Fold 2 ──► Evaluate Fold 2
...
```

Inside each fold, the Pipeline fits preprocessing **only on the training fold**. Validation data remains 100% uncorrupted.

---

## 17. Pipeline + GridSearchCV (Hyperparameter Tuning)

When tuning hyperparameters with `GridSearchCV`, passing a Pipeline allows you to tune model parameters AND preprocessing parameters simultaneously!

### Syntax: Double Underscore (`<step_name>__<parameter>`)
```python
from sklearn.model_selection import GridSearchCV

# Define parameters to search
param_grid = {
    # Tune model regularization parameter 'C'
    'classifier__C': [0.1, 1.0, 10.0],
    # Tune imputer strategy
    'preprocessor__num__imputer__strategy': ['mean', 'median'],
}

grid = GridSearchCV(full_pipeline, param_grid, cv=3)
grid.fit(X_train, y_train)

print('Best parameters:', grid.best_params_)
```

The double underscore (`__`) navigates down the pipeline chain to tune any nested component safely without data leakage.

---

## 18. Saving and Deploying a Pipeline

When deploying an ML application, saving individual components as separate files leads to version mismatch bugs. With a Pipeline, you save the **entire workflow as one single file**.

```python
import pickle

# 1. Save the full pipeline
with open("loan_model_pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)

# 2. Load the pipeline in your production app
with open("loan_model_pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# 3. Predict directly on raw user data!
new_customer = pd.DataFrame(
    [{"Age": 29, "Salary": 62000, "Gender": "Female", "City": "Delhi"}]
)
prediction = loaded_pipeline.predict(new_customer)
```

> **Security Note:**  
> Python's `pickle` module can execute arbitrary code during deserialization. **Only unpickle files from trusted sources** (e.g. models you trained yourself). For production environments, modern libraries like `joblib` or ONNX are also widely used.

---

## 19. Pipeline vs. ColumnTransformer: The Definitive Comparison

It is essential never to confuse these two tools:

| Dimension | `ColumnTransformer` | `Pipeline` |
| :--- | :--- | :--- |
| **Core Question** | **WHERE?** *(Which columns get what?)* | **WHEN / ORDER?** *(What sequence of steps?)* |
| **Orientation** | **Vertical / Parallel** across columns | **Horizontal / Sequential** through time |
| **Typical Steps** | Group 1 $\to$ Scaler, Group 2 $\to$ Encoder | Imputer $\to$ Scaler $\to$ Classifier |
| **Final Step** | Always produces a transformed array | Can produce predictions (`model.predict()`) |
| **Analogy** | Traffic Controller (routes lanes) | Assembly Line (moves parts through stages) |

### The Memory Trick:
- **ColumnTransformer** = Selecting columns in parallel.
- **Pipeline** = Chaining operations in sequence.

---

## 20. How Pipeline and ColumnTransformer Work Together

```text
                             PIPELINE
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
      [ ColumnTransformer ]             [ Final Estimator ]
               │                        (e.g., LogisticRegression)
       ┌───────┴───────┐
       ▼               ▼
   Numerical      Categorical
     Lane            Lane
```

`ColumnTransformer` sits inside the `Pipeline` as the first major preprocessing station. Once `ColumnTransformer` finishes preparing the numbers, `Pipeline` passes them straight to the final algorithm.

---

## 21. The Complete End-to-End ML Workflow

Here is how all the pieces we have learned fit together in real-world data science:

```text
                          Raw Dataset
                               │
                               ▼
                   Train / Test Split (Holdout)
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
            X_train                         X_test
               │                               │
               ▼                               │
      [ Define Pipeline ]                      │
      ├── ColumnTransformer                    │
      │   ├── Imputers                         │
      │   ├── Scalers                          │
      │   └── Encoders                         │
      └── ML Classifier                        │
               │                               │
               ▼                               │
      pipeline.fit(X_train, y_train)           │
               │                               ▼
               └──────────────────────► pipeline.predict(X_test)
                                               │
                                               ▼
                                      Evaluate Metric (Accuracy)
```

---

## 22. Important Practical Rules to Follow

1. **Split First:** Always split your dataset into `X_train` and `X_test` before fitting any pipeline.
2. **Keep Preprocessing Inside:** Do not perform scaling or encoding outside the pipeline; let the pipeline handle it.
3. **Never Fit on Test:** Never call `fit()` or `fit_transform()` on `X_test`. Use `pipeline.predict(X_test)`.
4. **Use Sub-Pipelines for Multi-Step Columns:** If a column needs both imputation and scaling, build a small sub-pipeline for it inside `ColumnTransformer`.
5. **Model Goes Last:** The machine learning algorithm is always the final step in the pipeline.
6. **Use Pipelines with Cross-Validation:** Pass `pipeline` into `cross_val_score` or `GridSearchCV` to prevent leakage between folds.
7. **Verify Unpickling Sources:** Only load serialized `.pkl` files created by your own trusted workflows.

---

## 23. Common Mistakes to Avoid

### 1. Fitting Preprocessing on the Whole Dataset Before Splitting
- **The Mistake:** Running `scaler.fit_transform(df)` before calling `train_test_split()`.
- **Why It Fails:** Test statistics leak into the training process, producing overly optimistic, deceptive performance metrics.

### 2. Confusing Pipeline and ColumnTransformer
- **The Mistake:** Trying to use `Pipeline` to apply different encoders to different columns, or using `ColumnTransformer` to chain an imputer into a model.
- **The Fix:** Remember: `ColumnTransformer` splits columns horizontally; `Pipeline` chains steps sequentially.

### 3. Putting the Estimator in the Middle of a Pipeline
- **The Mistake:** Putting `LogisticRegression()` as Step 1 and `StandardScaler()` as Step 2.
- **Why It Fails:** Intermediate steps must be transformers that output data. A classifier outputs predictions, not transformed features. The model must always be the final step.

### 4. Forgetting `handle_unknown='ignore'` in Production Pipelines
- **The Mistake:** Leaving default `OneHotEncoder()` settings so the pipeline crashes when an unseen category appears in test data.
- **The Fix:** Use `OneHotEncoder(handle_unknown='ignore')` inside your pipeline for bulletproof deployment.

---

## 24. Final Mental Model

Whenever you design an ML system, remember this simple hierarchy:

```text
1. ColumnTransformer = WHERE
   Which columns need imputation, scaling, or encoding?

2. Pipeline = WHEN
   What is the step-by-step sequence from raw data to final prediction?

3. fit(X_train) = LEARN
   Every transformer learns parameters; the model learns weights.

4. predict(X_test) = APPLY
   Transformers transform without re-learning; the model predicts.
```

> **The Big Takeaway:**  
> A Machine Learning Pipeline is your project's automated assembly line. Raw data enters, passes through standardized cleaning stations, and emerges as clean predictions—reliably, consistently, and without data leakage.
