# Module 03: Feature Engineering — Handling Missing Categorical Data

In the previous module ([Handling Missing Numerical Data](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/11_Handling_Missing_Numerical_Data/)), we learned how to impute continuous numbers using the **Mean**, **Median**, **Arbitrary Constants**, and **End-of-Distribution** boundaries.

However, categorical variables (like `City`, `Gender`, or `Embarked`) contain text labels rather than continuous numbers. You cannot calculate the "average" of Delhi, Mumbai, and Gurgaon!

When dealing with missing categorical data (`NaN`), we use two primary univariate imputation strategies:
1. **Most Frequent (Mode) Imputation**
2. **Missing Category Imputation**

---

## 1. The Core Challenge of Missing Categorical Data

When a categorical feature contains missing values, algorithms like Logistic Regression, Random Forests, and KNN cannot process `NaN`. 

We have two fundamental choices:
- **Guess the most likely category** from existing labels $\implies$ **Most Frequent Imputation**.
- **Acknowledge the missingness** by creating an explicit new label $\implies$ **Missing Category Imputation**.

```text
Raw Categorical Observation: [ City: NaN ]
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
1. Most Frequent (Mode):               2. Missing Category:
   [ City: 'Delhi' ]                      [ City: 'Missing' ]
   (Assumes missing record was            (Preserves the fact that
    the most common city)                  the city was not recorded)
```

---

## 2. Most Frequent (Mode) Imputation

### The Simple Idea
Replace every missing value with the **Mode** — the category that appears most frequently in that feature.

### Step-by-Step Toy Example
Suppose we collect the city of residence for 7 customers:

| Customer ID | City |
| :---: | :---: |
| 1 | Delhi |
| 2 | Mumbai |
| 3 | Delhi |
| 4 | Gurgaon |
| 5 | **NaN** |
| 6 | Delhi |
| 7 | **NaN** |

#### Step 1: Count category frequencies
- `Delhi`: **3** times (Most Frequent $\implies$ **Mode**)
- `Mumbai`: 1 time
- `Gurgaon`: 1 time

#### Step 2: Replace all `NaN` values with the Mode (`'Delhi'`)

| Customer ID | City (After Mode Imputation) |
| :---: | :---: |
| 1 | Delhi |
| 2 | Mumbai |
| 3 | Delhi |
| 4 | Gurgaon |
| 5 | **Delhi** *(Imputed)* |
| 6 | Delhi |
| 7 | **Delhi** *(Imputed)* |

Both missing entries are now filled with `'Delhi'`. Zero missing values remain.

---

## 3. When Should You Use Most Frequent Imputation?

Most Frequent Imputation works well when:
1. **Missingness is small**: Typically **under 5%** of the rows in the column.
2. **Missingness is purely random (MCAR)**: The value is missing by chance, not because of a systematic pattern.
3. **The mode is dominant**: One category already represents the vast majority of the population (e.g., 85% of users are from one country).

---

## 4. The Critical Problem with Most Frequent Imputation: Mode Inflation

What happens when missingness is high?

Imagine a survey with 150 responses:
- `Delhi`: 90 people
- `Mumbai`: 5 people
- `Gurgaon`: 5 people
- **Missing (`NaN`)**: **50 people**

If you replace all 50 missing values with the mode (`Delhi`):
- `Delhi`: $90 + 50 =$ **140 people** (93.3% of the dataset!)
- `Mumbai`: 5 people (3.3%)
- `Gurgaon`: 5 people (3.3%)

```text
Original Balance:   Delhi (60%)  |  Mumbai (3.3%)  |  Gurgaon (3.3%)  |  Missing (33.3%)
After Mode Fill:    Delhi (93.3%) |  Mumbai (3.3%)  |  Gurgaon (3.3%)
```

### The Statistical Consequence:
- You artificially **oversaturate the majority category**.
- The model treats `Delhi` as almost guaranteed, ignoring the genuine diversity of the minority classes (`Mumbai` and `Gurgaon`).
- It hides the fact that **one-third of the customers chose not to disclose their city**!

> [!WARNING]
> **The Takeaway:**  
> The more missing values you have, the more you distort the original category proportions by forcing them into the mode.

---

## 5. Missing Category Imputation

### The Simple Idea
Instead of pretending that missing observations belong to an existing category, we create an **entirely new category called `"Missing"`** (or `"Unknown"` / `"None"`).

### Step-by-Step Example
Consider the same table:

| Customer ID | Raw City | City (After Missing Category Imputation) |
| :---: | :---: | :---: |
| 1 | Delhi | Delhi |
| 2 | Mumbai | Mumbai |
| 3 | **NaN** | **Missing** |
| 4 | Delhi | Delhi |
| 5 | **NaN** | **Missing** |

Now, instead of 2 unique cities, the feature has 3 distinct levels:
`['Delhi', 'Mumbai', 'Missing']`

---

## 6. Why is Missing Category Imputation So Powerful?

### 1. It Doesn't Fake Data
It makes zero assumptions about what the missing value "should" have been. It does not inflate the frequency of Delhi or Mumbai.

### 2. It Preserves Informative Missingness (MNAR Signal)
In many real-world problems, **the fact that a value is missing is itself a predictive feature**:
- In the Titanic dataset, over 77% of passengers had `Cabin = NaN`. Why? Because 3rd class passengers were never assigned private luxury cabins!
  - If we replaced missing cabins with the most frequent cabin, we would tell the algorithm that poor passengers had luxury rooms!
  - By replacing `NaN` with `'Missing'`, the model easily learns:
    $$\text{Cabin} = \text{'Missing'} \implies \text{3rd Class Passenger} \implies \text{Low Survival Odds}$$
- In medical records, a missing test result often means the doctor saw no symptoms requiring the test.
- In loan applications, omitting an employer name often indicates unemployment.

---

## 7. Comparison: Most Frequent vs. Missing Category

| Comparison Dimension | Most Frequent Imputation | Missing Category Imputation |
| :--- | :--- | :--- |
| **What replaces `NaN`?** | The most common category (e.g., `'Delhi'`) | A new category string (`'Missing'`) |
| **New categories created?** | No (uses existing vocabulary) | **Yes** (adds 1 new category level) |
| **Best when missingness is...** | **Low** ($< 5\%$) | **High** ($> 10\% - 20\%$) |
| **Preserves missing signal?** | ❌ No (blends into majority class) | ✅ **Yes** (isolates missing cohort) |
| **Risk of category distortion?** | **High** if missingness is moderate/high | **Zero** (original classes unchanged) |
| **Scikit-Learn Parameter** | `strategy="most_frequent"` | `strategy="constant", fill_value="Missing"` |

---

## 8. Scikit-Learn `SimpleImputer` Implementation

In production machine learning, we use Scikit-Learn's `SimpleImputer` to prevent data leakage.

### Strategy 1: Most Frequent
```python
from sklearn.impute import SimpleImputer

# Learns the mode on X_train and fills NaN
imputer_mode = SimpleImputer(strategy="most_frequent")

X_train_clean = imputer_mode.fit_transform(X_train)
X_test_clean  = imputer_mode.transform(X_test)
```

### Strategy 2: Missing Category
```python
# Fills all NaN with the explicit string 'Missing'
imputer_missing = SimpleImputer(strategy="constant", fill_value="Missing")

X_train_clean = imputer_missing.fit_transform(X_train)
X_test_clean  = imputer_missing.transform(X_test)
```

> [!IMPORTANT]
> **Strict Rule Against Data Leakage:**  
> When using `strategy="most_frequent"`, always call `fit(X_train)` strictly on the training data. If you calculate the mode across the full dataset, test set information leaks into your training pipeline!

---

## 9. Connecting Categorical & Numerical Imputation

You now have a complete, unified understanding of Univariate Imputation across all data types:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      UNIVARIATE IMPUTATION TOOLKIT                     │
└────────────────────────────────────────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   NUMERICAL FEATURES                    CATEGORICAL FEATURES
 (Age, Salary, Fare)                   (City, Gender, Cabin)
            │                                     │
   ┌────────┴────────┐                   ┌────────┴────────┐
   ▼                 ▼                   ▼                 ▼
Normal/Symmetric   Skewed/Outliers    Small Null % (<5%)   Large Null % (>10%)
   │                 │                   │                 │
   ▼                 ▼                   ▼                 ▼
  Mean             Median           Most Frequent       "Missing"
Imputation       Imputation          (Mode) Fill        Category
```

---

## 10. Common Mistakes to Avoid

1. **Blindly filling high-missingness columns with the Mode**:
   Replacing 40% missing values with the most common category completely destroys the original probability distribution of your categories.
2. **Treating missing values as accidental when they are informative**:
   Always ask: *"Why is this value missing?"* If missingness carries meaning (like missing Titanic Cabins), use `'Missing'`.
3. **Calculating the Mode before `train_test_split`**:
   Never compute `.mode()` on the entire dataset. Always split first, fit on `X_train`, and transform `X_test`.
4. **Forgetting to encode after imputation**:
   Imputing `'Missing'` leaves the column as a text string. You must still pass it into [`OneHotEncoder`](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/04_One_Hot_Encoding/) or [`OrdinalEncoder`](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/03_Encoding_Categorical_Data/) before feeding it to a model.

---

## 11. Final Mental Model

```text
                     Missing Categorical Value
                                 │
                                 ▼
                     How much data is missing?
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          Small (< 5%)                    Large (> 10%)
                 │                               │
                 ▼                               ▼
       Is missingness random?             Does missingness
                                           carry signal?
                 │                               │
         ┌───────┴───────┐                       ▼
         ▼               ▼             Create New Category:
     YES (MCAR)       NO (MNAR)            "Missing"
         │               │                       │
         ▼               ▼                       ▼
   Most Frequent      "Missing"         Preserves natural classes
    (Mode) Fill       Category           & informs the algorithm!
```
