# Python Prerequisites: 04. Data Handling Concepts for Machine Learning

Before writing any machine learning algorithms, you must understand the terminology used to describe data, how tables are divided into inputs and outputs, and why data splitting is the foundation of model evaluation.

---

## 1. Deconstructing a Dataset: Rows, Columns, and Cells

A **dataset** is simply a structured collection of information, typically arranged as a two-dimensional grid of rows and columns.

```text
               Column 1      Column 2      Column 3       Column 4
               (Feature)     (Feature)     (Feature)      (Target)
                [ Age ]     [ Gender ]     [ Salary ]   [ Purchased ]
  Row 0  ──►      25           Male          50000            0
  Row 1  ──►      30          Female         70000            1
  Row 2  ──►      28           Male          45000            0
```

### Key Vocabulary:
- **Sample / Observation / Row / Instance:** A single data record (e.g. Row 0 represents one individual customer).
- **Feature / Attribute / Column / Dimension:** A measurable property or characteristic of the samples (e.g. `Age`, `Salary`).
- **Target / Label / Ground Truth:** The outcome we want the model to learn how to predict (e.g. `Purchased`).

---

## 2. Inputs ($X$) vs. Outputs ($y$)

In mathematics and machine learning:
$$y = f(X)$$

- **$X$ (Feature Matrix - Capitalized):**
  - Represents the **independent variables** (the inputs).
  - Capitalized ($X$) because it is a **2-dimensional matrix** with multiple rows and columns.
- **$y$ (Target Vector - Lowercase):**
  - Represents the **dependent variable** (the outcome we are trying to predict).
  - Lowercase ($y$) because it is a **1-dimensional vector** of single values.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "Age": [25, 30, 28],
        "Salary": [50000, 70000, 45000],
        "Purchased": [0, 1, 0],
    }
)

# Standard Scikit-Learn extraction:
X = df.drop(columns=["Purchased"])  # 2D DataFrame: Features
y = df["Purchased"]  # 1D Series: Target

print("Feature Matrix X shape:", X.shape)  # (3, 2)
print("Target Vector y shape: ", y.shape)  # (3,)
```

---

## 3. Data Types: Numerical vs. Categorical

Every column falls into one of two major categories:

```text
                           Data Types
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
        Numerical Data                  Categorical Data
       (Numbers / Math)                 (Labels / Groups)
         ┌─────┴─────┐                     ┌─────┴─────┐
         ▼           ▼                     ▼           ▼
    Continuous   Discrete               Nominal     Ordinal
   (Decimals,   (Whole counts,         (No order,  (Has hierarchy,
    Salary,      Num Children,          City,       Education,
    Temperature) Siblings)              Gender)     Rating)
```

- **Continuous:** Can take any decimal value within a range (e.g., Temperature = 98.6°F, Salary = \$72,450.50).
- **Discrete:** Can only take distinct integer counts (e.g., Number of cars owned = 2).
- **Nominal:** Labels without any ranking (e.g., `Delhi`, `Mumbai`, `Bangalore`).
- **Ordinal:** Labels with a clear hierarchy (e.g., `High School` < `Graduate` < `Postgraduate`).

---

## 4. The 3 Data Partitions: Train, Validation, and Test

Why can't we simply train our model on all available data?

Imagine a student studying for an exam:
- If a teacher gives the student 10 practice questions with answers, and then gives the **exact same 10 questions** on the final exam, the student can score 100% just by memorizing the answers.
- That student hasn't actually *learned* how to solve new problems.
- To test real understanding, the exam must contain **new, unseen questions**.

```text
                          Entire Dataset (100%)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          Training Set (~70 - 80%)            Test Set (~20 - 30%)
          Used by algorithm to learn          Locked away in a safe!
          weights and patterns.               Evaluates true performance
                                              on brand new, unseen data.
```

- **Training Set ($X_{train}, y_{train}$):** The material the algorithm studies during `model.fit()`.
- **Validation Set ($X_{val}, y_{val}$):** Used to tune hyperparameters and check for overfitting during experimentation.
- **Test Set ($X_{test}, y_{test}$):** Unseen holdout data used strictly for the final exam.

---

## 5. Splitting Data in Scikit-Learn: `train_test_split`

Scikit-Learn provides a dedicated function to perform this split randomly and fairly:

```python
from sklearn.model_selection import train_test_split

# Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:    ", len(X_test))
```
- **`test_size=0.20`**: Allocates 20% of data to the test set and 80% to the training set.
- **`random_state=42`**: Sets a random seed so the split is identical every time you run the script.

---

## 6. What is Data Leakage?

> **Data Leakage** occurs when information from the test dataset accidentally enters the training process before or during model fitting.

### A Common Everyday Example:
Suppose you calculate the average age of customers across the *entire* dataset and use that average to fill missing values.  
Because the test set was included in calculating the average, **information from the future test set leaked into training**.

### The Rule to Prevent Leakage:
1. Always split your dataset into `X_train` and `X_test` **first**.
2. Learn all statistics (means, standard deviations, categories, medians) **only from `X_train`**.
3. Use those learned statistics to transform `X_test`.
