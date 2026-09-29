# Module 03: Feature Engineering — ColumnTransformer (`sklearn.compose.ColumnTransformer`)

When working on real-world Machine Learning problems, datasets are rarely composed of just one type of data. A single table will usually contain numbers, categories with a hierarchy, categories with no order, and even missing values.

Because different columns represent fundamentally different kinds of information, **we cannot apply the same preprocessing steps to the whole dataset**.

Scikit-Learn's **`ColumnTransformer`** is the standard tool that solves this challenge cleanly and automatically.

---

## 1. What is ColumnTransformer?

### Understand It First: A Realistic Dataset

Imagine you are building a machine learning model to predict employee performance or loan eligibility. Your raw dataset looks like this:

| Age | Gender | City | Education | Salary |
| :---: | :---: | :---: | :---: | :---: |
| 25 | Male | Delhi | Graduate | 50000 |
| 30 | Female | Mumbai | Postgraduate | 70000 |
| NaN | Male | Delhi | High School | 45000 |
| 35 | Female | Bangalore | Graduate | 80000 |

Let us look closely at each column:

1. **`Age`** (Numerical):
   - Contains numbers, but has missing values (`NaN`).
   - Requires **missing value imputation** (e.g., replacing `NaN` with the median age).
2. **`Gender`** (Nominal Categorical):
   - Categorical with no order (`Male`, `Female`).
   - Requires **One-Hot Encoding**.
3. **`City`** (Nominal Categorical):
   - Categorical with no natural ranking (`Delhi`, `Mumbai`, `Bangalore`).
   - Requires **One-Hot Encoding**.
4. **`Education`** (Ordinal Categorical):
   - Categorical with a clear rank: `High School` < `Graduate` < `Postgraduate`.
   - Requires **Ordinal Encoding** to preserve this natural order.
5. **`Salary`** (Numerical):
   - Values are in the tens of thousands ($50,000, $70,000).
   - Requires **Standardization** (`StandardScaler`) so its large scale does not dominate other features.

### The Central Problem: The Manual Preprocessing Headache

Without a dedicated tool, how would you preprocess this dataset?
You would have to do everything manually:

1. Slice out `Age`, fill its missing values, and save the result as an array.
2. Slice out `Gender` and `City`, run `OneHotEncoder`, and save the result.
3. Slice out `Education`, run `OrdinalEncoder`, and save the result.
4. Slice out `Salary`, run `StandardScaler`, and save the result.
5. Use `np.hstack()` or `pd.concat()` to stitch all the separate arrays back into a single table.

### Why This Manual Approach Fails in Real ML Projects:
- **Lengthy & Messy:** You end up writing dozens of lines of repetitive slicing and stacking code.
- **Error-Prone:** It is very easy to mix up column orders, misplace columns, or accidentally overwrite variables.
- **Hard to Maintain:** If you add or drop a column later, you have to rewrite all the manual slicing and index tracking.
- **Inconsistent Across Train & Test:** You must remember to apply the exact same sequence of steps to test data, which often leads to bugs and data leakage.

### The Solution: Formal Definition

> **ColumnTransformer** is a Scikit-Learn utility that allows different preprocessing transformations to be applied to different groups of columns within the same dataset in a single, unified step.

---

## 2. Why Do We Need ColumnTransformer?

The core motivation behind ColumnTransformer can be summarized in one simple mental model:

```text
Different Columns in Your Dataset
               ↓
 Require Different Preprocessing Techniques
               ↓
 Handled Together by ColumnTransformer
```

Instead of manually slicing and gluing arrays, ColumnTransformer allows you to declare a **clean preprocessing plan** all in one place:

```text
Age          ───► SimpleImputer(strategy='median') ───┐
Salary       ───► StandardScaler()                 ───┤
Education    ───► OrdinalEncoder(categories=[...]) ───┼──► Unified Transformed Array
Gender, City ───► OneHotEncoder()                  ───┘
```

You define the rules once. Scikit-Learn handles the column selection, transformation, and recombination automatically.

---

## 3. Basic Syntax

To use ColumnTransformer, import it from `sklearn.compose`:

```python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ('name1', transformer1, columns1),
        ('name2', transformer2, columns2),
        ('name3', transformer3, columns3),
    ]
)
```

The `transformers` parameter takes a **list of tuples**.
Each tuple represents one transformation task and always contains exactly **three parts**:

```text
(name, transformer, columns)
```

For example:
```python
('onehot', OneHotEncoder(), ['City', 'Gender'])
```

This instruction tells Scikit-Learn:
> *"Apply `OneHotEncoder()` to the columns `City` and `Gender`, and label this step `'onehot'`."*

---

## 4. Understanding the Three Parts of a Transformer Tuple

Let's break down each element of `('onehot', OneHotEncoder(), ['City', 'Gender'])`:

### Part 1: Name (`'onehot'`)
- A simple string of your choice to name the transformation step.
- Acts as an identifier so you can easily reference or debug this specific step later.
- Example names: `'num_imputer'`, `'cat_ohe'`, `'education_ord'`.

### Part 2: Transformer (`OneHotEncoder()`)
- An instance of a Scikit-Learn transformer object that defines **what operation to perform**.
- Examples:
  - `SimpleImputer()` for filling missing values
  - `StandardScaler()` or `MinMaxScaler()` for numerical scaling
  - `OneHotEncoder()` for nominal categories
  - `OrdinalEncoder()` for ordered categories

### Part 3: Columns (`['City', 'Gender']`)
- A list of column names (or column indices) that specifies **where to apply the transformation**.
- Examples: `['Age', 'Salary']`, `['Education']`, or `[0, 1]`.

### Summary Table

| Part | Type | Purpose | Example |
| :--- | :--- | :--- | :--- |
| **1. Name** | `str` | Identifies this transformation step | `'encoder_nominal'` |
| **2. Transformer** | Scikit-Learn object | Specifies what operation to perform | `OneHotEncoder()` |
| **3. Columns** | `list` of strings/ints | Specifies which columns to transform | `['City', 'Gender']` |

---

## 5. A Simple Example: OneHot + Ordinal Encoding

Let's look at a concrete beginner example with a small table:

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Sample dataset
df = pd.DataFrame(
    {
        "Age": [25, 30, 28],
        "Gender": ["Male", "Female", "Male"],
        "Education": ["Graduate", "Postgraduate", "High School"],
        "City": ["Delhi", "Mumbai", "Delhi"],
    }
)
```

### The Plan:
1. `Education` has an order: `High School` < `Graduate` < `Postgraduate` $\longrightarrow$ Use **OrdinalEncoder**.
2. `Gender` and `City` have no order $\longrightarrow$ Use **OneHotEncoder**.

### The ColumnTransformer Definition:
```python
transformer = ColumnTransformer(
    transformers=[
        (
            "ordinal",
            OrdinalEncoder(
                categories=[["High School", "Graduate", "Postgraduate"]]
            ),
            ["Education"],
        ),
        ("onehot", OneHotEncoder(sparse_output=False), ["Gender", "City"]),
    ]
)
```

Before running any code, notice how readable this is:
- We can immediately see which transformer handles which columns.
- The order of categories for `Education` is explicitly stated.
- `sparse_output=False` is used so we can clearly view the numbers as a standard array.

---

## 6. What Happens When We Call `fit_transform()`?

When you call:
```python
X_transformed = transformer.fit_transform(df)
```

Under the hood, `ColumnTransformer` performs five distinct steps automatically:

```text
                     Original Dataset
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
    ['Education']                      ['Gender', 'City']
         │                                     │
         ▼                                     ▼
   OrdinalEncoder                        OneHotEncoder
 (learns categories:                  (learns unique labels:
  High School, Grad, PG)               Male/Female, Delhi/Mumbai)
         │                                     │
         ▼                                     ▼
 Transformed 1D Array                 Transformed Binary Array
   (e.g., [1, 2, 0])                    (e.g., [1, 0, 1, 0])
         │                                     │
         └──────────────────┬──────────────────┘
                            ▼
             Combined Final 2D Array (hstack)
```

1. **Splits:** Slices out the columns specified for each transformer.
2. **Fits:** Learns any required statistics or categories from each column group.
3. **Transforms:** Applies the transformation to produce processed numbers.
4. **Combines:** Stitches the resulting numerical blocks side-by-side into a single 2D NumPy array.

You never have to write `np.hstack()` or manage column slices yourself!

---

## 7. Understanding `remainder`

Look closely at our previous example with `df`:
- We transformed `Education` with `OrdinalEncoder`.
- We transformed `Gender` and `City` with `OneHotEncoder`.
- **Wait! What happened to `Age`?** We did not mention `Age` anywhere in `transformers`!

What does Scikit-Learn do with columns that were **not** assigned to any transformer?

This behavior is controlled by the **`remainder`** parameter:

```python
transformer = ColumnTransformer(
    transformers=[...],
    remainder='drop',  # or 'passthrough'
)
```

---

## 8. `remainder='drop'` (The Default Setting)

By default, `remainder='drop'`.

```python
transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=False), ["Gender", "City"])
    ],
    remainder="drop",  # Default behavior
)
```

### What Happens:
- **Specified columns:** `Gender` and `City` are encoded.
- **Unspecified columns:** `Age` and `Education` are **completely discarded (dropped)** from the output.

### Mental Model:
> **`remainder='drop'`** $\longrightarrow$ *"If I didn't explicitly give you a transformer, throw that column away."*

---

## 9. `remainder='passthrough'`

What if you want to keep `Age` exactly as it is without modifying it?
Set `remainder='passthrough'`:

```python
transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=False), ["Gender", "City"])
    ],
    remainder="passthrough",
)
```

### What Happens:
- **Specified columns:** `Gender` and `City` are encoded.
- **Unspecified columns:** `Age` and `Education` are **kept unchanged (passed through)** and appended to the final output table.

### Comparison Table

| Setting | What Happens to Unmentioned Columns | When to Use It |
| :--- | :--- | :--- |
| **`remainder='drop'`** *(default)* | They are **removed** from the final output. | When you only want to keep the specific features you transformed. |
| **`remainder='passthrough'`** | They are **kept as-is** without modification. | When some columns are already clean numbers that do not need scaling or encoding. |

---

## 10. Handling Multiple Preprocessing Techniques Together

Now let's see the true practical power of `ColumnTransformer` when managing multiple different transformations simultaneously:

### Scenario:
Suppose we have a dataset with 5 columns:
- `Age` and `Salary` $\longrightarrow$ Numerical features with different scales.
- `Education` $\longrightarrow$ Ordinal feature (`High School` < `Graduate` < `Postgraduate`).
- `City` and `Gender` $\longrightarrow$ Nominal features with no order.

### The Unified Recipe:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

transformer = ColumnTransformer(
    transformers=[
        # 1. Scale numerical columns
        ("scale_num", StandardScaler(), ["Age", "Salary"]),
        # 2. Ordinally encode education with explicit ranking
        (
            "ord_edu",
            OrdinalEncoder(
                categories=[["High School", "Graduate", "Postgraduate"]]
            ),
            ["Education"],
        ),
        # 3. One-hot encode nominal categories
        ("ohe_cat", OneHotEncoder(sparse_output=False), ["Gender", "City"]),
    ],
    remainder="drop",
)
```

### The Flow:
```text
Age, Salary       ──► StandardScaler() ──┐
Education         ──► OrdinalEncoder() ──┼──► Clean, Uniform Preprocessed Array
Gender, City      ──► OneHotEncoder()  ──┘
```

Notice how organized and readable this code is. Everything is declared in a single, transparent configuration block.

---

## 11. ColumnTransformer + Missing Values (`SimpleImputer`)

In real datasets, numerical columns often contain missing values (`NaN`).
We can include Scikit-Learn's **`SimpleImputer`** directly inside `ColumnTransformer`.

### Simple Example:
Suppose `Age` has missing values:

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame(
    {
        "Age": [25.0, 30.0, np.nan, 40.0],
        "Gender": ["Male", "Female", "Female", "Male"],
        "City": ["Delhi", "Mumbai", "Delhi", "Mumbai"],
    }
)
```

### The Solution:
```python
transformer = ColumnTransformer(
    transformers=[
        # Fill missing values in Age with the median
        ("impute_age", SimpleImputer(strategy="median"), ["Age"]),
        # One-hot encode categorical columns
        ("ohe_cat", OneHotEncoder(sparse_output=False), ["Gender", "City"]),
    ],
    remainder="passthrough",
)
```

### How It Works:
```text
Age (with NaN)  ──► SimpleImputer(median) ──► Clean Age without NaNs ──┐
                                                                       ├──► Final Output
Gender, City    ──► OneHotEncoder()       ──► Binary 0/1 Columns     ──┘
```

Missing values are safely filled and categorical strings are converted to numbers, all in one shot.

---

## 12. Why ColumnTransformer is Better Than Manual Preprocessing

| Feature | Manual Preprocessing (Pandas / NumPy) | With `ColumnTransformer` |
| :--- | :--- | :--- |
| **Code Length** | Dozens of slicing, encoding, and concatenating lines. | A single, clean definition block. |
| **Error Risk** | High risk of column misalignment or lost columns. | Low; Scikit-Learn tracks column mapping automatically. |
| **Train/Test Consistency** | Must manually repeat each slice and transform on test data. | Call `fit_transform(X_train)` then `transform(X_test)`. |
| **Data Leakage** | Easy to accidentally fit scalers on full data before splitting. | Natural separation between fit and transform stages. |
| **Pipeline Integration** | Cannot be plugged into Scikit-Learn `Pipeline`. | Plugs directly into `Pipeline` alongside any ML model. |

---

## 13. ColumnTransformer + Pipeline: The Production Standard

In professional machine learning workflows, preprocessing is never done in isolation. It is connected directly to a machine learning model using a **`Pipeline`**.

```text
Raw Data ──► ColumnTransformer (Preprocessing) ──► ML Model (e.g., LogisticRegression) ──► Predictions
```

### The Relationship:
- **`ColumnTransformer`** prepares the columns (imputes, scales, encodes).
- **`Pipeline`** chains `ColumnTransformer` together with the final algorithm.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Build the complete end-to-end pipeline
full_pipeline = Pipeline(
    steps=[
        ("preprocessor", transformer),  # Our ColumnTransformer from earlier
        ("classifier", LogisticRegression()),  # The ML model
    ]
)

# Train everything in one single line:
full_pipeline.fit(X_train, y_train)

# Predict on new unseen data:
# The pipeline automatically applies the exact same ColumnTransformer before predicting!
predictions = full_pipeline.predict(X_test)
```

### Why This Is Essential:
1. **Zero Preprocessing Leaks:** The test data is guaranteed to use the exact statistics and categories learned from the training set.
2. **One-Line Deployment:** When deploying to production, you don't write complex data cleaning functions; you just pass new raw user data to `full_pipeline.predict(new_data)`.

---

## 14. The Train / Test Workflow (Preventing Data Leakage)

Just like individual encoders and scalers, **you must never fit a `ColumnTransformer` on your test data**.

```text
         Training Data (X_train)                   Test Data (X_test)
                   │                                        │
           fit_transform()                                  │
                   │                                        │
       Learns medians, categories,                          │
       means, and standard deviations                       │
                   │                                   transform()
                   ▼                                        │
           Clean X_train_preprocessed                       ▼
                                                Clean X_test_preprocessed
                                           (Uses statistics learned from TRAIN)
```

### The Correct Code Pattern:
```python
# 1. Split FIRST
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. fit_transform on TRAIN only
X_train_processed = transformer.fit_transform(X_train)

# 3. transform on TEST only (NEVER call fit or fit_transform on test!)
X_test_processed = transformer.transform(X_test)
```

- **`fit_transform(X_train)`**: Learns the means, medians, and categories from `X_train`, and transforms `X_train`.
- **`transform(X_test)`**: Uses those **already-learned** values to transform `X_test`.

---

## 15. Complete Practical Walkthrough

Let us trace a full example from start to finish:

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# 1. Create realistic sample dataset
df = pd.DataFrame(
    {
        "Age": [25, 30, 28],
        "Gender": ["Male", "Female", "Male"],
        "City": ["Delhi", "Mumbai", "Delhi"],
        "Education": ["Graduate", "Postgraduate", "High School"],
        "Salary": [50000, 70000, 40000],
    }
)

# 2. Set up ColumnTransformer
transformer = ColumnTransformer(
    transformers=[
        ("scale_num", StandardScaler(), ["Age", "Salary"]),
        (
            "ord_edu",
            OrdinalEncoder(
                categories=[["High School", "Graduate", "Postgraduate"]]
            ),
            ["Education"],
        ),
        ("ohe_cat", OneHotEncoder(sparse_output=False), ["Gender", "City"]),
    ],
    remainder="drop",
)

# 3. Fit and transform
output_array = transformer.fit_transform(df)
```

### Displaying the Result:
Because we used `sparse_output=False` inside `OneHotEncoder`, the output is a standard 2D NumPy array that is easy to print and inspect:

```text
[[ -0.707,  -0.267,   1.0,    0.0,  1.0,   1.0,  0.0 ],
 [  1.414,   1.336,   2.0,    1.0,  0.0,   0.0,  1.0 ],
 [ -0.707,  -1.069,   0.0,    0.0,  1.0,   1.0,  0.0 ]]
```

---

## 16. Understanding the Transformed Output

Notice that our original table had **5 columns**, but the transformed output has **7 columns**!

### Why Did the Column Count Change?
1. **`['Age', 'Salary']`** (Scaled):
   - Scaled with `StandardScaler`.
   - 2 input columns $\longrightarrow$ **2 output columns** (values standardized with mean 0).
2. **`['Education']`** (Ordinal):
   - Encoded with `OrdinalEncoder`.
   - 1 input column $\longrightarrow$ **1 output column** (`High School=0`, `Graduate=1`, `Postgraduate=2`).
3. **`['Gender', 'City']`** (One-Hot Encoded):
   - `Gender` has 2 unique values (`Female`, `Male`) $\longrightarrow$ **2 columns**.
   - `City` has 2 unique values in our sample (`Delhi`, `Mumbai`) $\longrightarrow$ **2 columns**.
   - Total One-Hot columns = $2 + 2 =$ **4 columns**.

$$2 \text{ (scaled)} + 1 \text{ (ordinal)} + 4 \text{ (one-hot)} = \mathbf{7 \text{ total features}}$$

---

## 17. Column Order and `get_feature_names_out()`

A common beginner question is:  
*"How do I know which column in the final array corresponds to which original feature?"*

### The Column Ordering Rule:
`ColumnTransformer` concatenates columns **in the exact order that the transformers were listed in the `transformers` list**.

1. First, all columns from transformer 1 (`scale_num` $\longrightarrow$ `Age`, `Salary`).
2. Next, all columns from transformer 2 (`ord_edu` $\longrightarrow$ `Education`).
3. Next, all columns from transformer 3 (`ohe_cat` $\longrightarrow$ `Gender`, `City`).
4. Finally, any columns kept via `remainder='passthrough'`.

### Inspecting Names with `get_feature_names_out()`:
Modern Scikit-Learn makes inspecting column names simple:

```python
feature_names = transformer.get_feature_names_out()
print(feature_names)
```

Output:
```text
['scale_num__Age',
 'scale_num__Salary',
 'ord_edu__Education',
 'ohe_cat__Gender_Female',
 'ohe_cat__Gender_Male',
 'ohe_cat__City_Delhi',
 'ohe_cat__City_Mumbai']
```

Scikit-Learn prefixes each column with `<transformer_name>__<feature_name>`, making it crystal-clear where every number originated.

---

## 18. Important Parameters Summary

| Parameter | Type / Options | Meaning | Simple Example |
| :--- | :--- | :--- | :--- |
| **`transformers`** | `list` of tuples | The list of transformation tasks to execute. | `[('ohe', OneHotEncoder(), ['City'])]` |
| **`name`** | `str` | A descriptive label for that step. | `'num_scaler'` |
| **`transformer`** | Object / `'drop'` / `'passthrough'` | The preprocessing estimator to apply. | `StandardScaler()` |
| **`columns`** | `list` of strings or indices | Which columns to route to this transformer. | `['Age', 'Salary']` |
| **`remainder`** | `'drop'` *(default)* | Drops any column not mentioned in `transformers`. | `remainder='drop'` |
| **`remainder`** | `'passthrough'` | Keeps unmentioned columns untouched in the output. | `remainder='passthrough'` |

---

## 19. Common Mistakes to Avoid

### 1. Applying the Same Preprocessing to Every Column
- **The Mistake:** Trying to run `StandardScaler()` on the whole DataFrame without separating categorical strings first.
- **The Fix:** Group numerical columns together for scaling, and route string columns to encoders.

### 2. Forgetting to Specify Column Lists
- **The Mistake:** Writing `('ohe', OneHotEncoder(), 'Gender')` as a single string instead of a list `['Gender']`.
- **The Fix:** Always pass column names as a list: `['Gender']`.

### 3. Accidentally Losing Columns with `remainder='drop'`
- **The Mistake:** Forgetting that `remainder='drop'` is the default, and wondering why unscaled numerical columns vanished from the output.
- **The Fix:** Set `remainder='passthrough'` if you have columns you want to retain without changes.

### 4. Calling `fit_transform()` on the Test Data
- **The Mistake:** Writing `X_test_pre = transformer.fit_transform(X_test)`.
- **The Fix:** Always call **`fit_transform(X_train)`** and then **`transform(X_test)`**. Never fit on test data!

### 5. Using the Wrong Encoding for the Wrong Category Type
- **The Mistake:** Passing `City` to `OrdinalEncoder` (which invents a fake ranking like `Delhi < Mumbai`) or passing `Education` to `OneHotEncoder` (which discards the true hierarchy).
- **The Fix:** Nominal $\longrightarrow$ `OneHotEncoder`, Ordinal $\longrightarrow$ `OrdinalEncoder(categories=[...])`.

---

## 20. ColumnTransformer as a Traffic Controller

A wonderful way to remember ColumnTransformer is as an **intelligent traffic controller**:

```text
                           Raw Input Dataset
                                   │
                                   ▼
                      [ ColumnTransformer ]
                     (Traffic Controller Hub)
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   Numerical Lane            Ordinal Lane              Nominal Lane
   ['Age', 'Salary']        ['Education']            ['City', 'Gender']
         │                         │                         │
         ▼                         ▼                         ▼
  [ StandardScaler ]      [ OrdinalEncoder ]        [ OneHotEncoder ]
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
                       Unified Final Clean Table
```

### The Big Insight:
**ColumnTransformer does not invent any new preprocessing technique.**  
It does not perform scaling or encoding on its own. It is simply an organizer that routes each column to the right tool and puts all the finished pieces back together.

---

## 21. ColumnTransformer vs. Individual Transformers

To keep your mental concepts clear, compare their individual roles:

| Component | What It Does | Scope |
| :--- | :--- | :--- |
| **`StandardScaler`** | Centers numbers to mean 0, variance 1 | Works on numerical numbers |
| **`SimpleImputer`** | Replaces missing values (mean, median, most_frequent) | Works on missing cells |
| **`OneHotEncoder`** | Creates binary 0/1 columns for nominal categories | Works on unordered text categories |
| **`OrdinalEncoder`** | Assigns ordered numbers (0, 1, 2) based on rank | Works on ordered text categories |
| **`ColumnTransformer`** | **Orchestrates all of the above together** | Organizes the whole dataset |

---

## 22. Final Mental Model

Whenever you start preparing a dataset for Machine Learning, remember this simple flow:

```text
1. Look at your columns: Which are numerical? Which are nominal? Which are ordinal?
                   ↓
2. Choose the right tool for each group (Imputer, Scaler, OHE, OrdinalEncoder)
                   ↓
3. Wrap them together in a ColumnTransformer
                   ↓
4. fit_transform(X_train), then transform(X_test)
                   ↓
5. Pass directly into your ML Model!
```

> **The Golden Rule:**  
> *"ColumnTransformer does not perform a new type of encoding or scaling. Instead, it organizes multiple existing preprocessing techniques and applies each one to the correct columns automatically."*
