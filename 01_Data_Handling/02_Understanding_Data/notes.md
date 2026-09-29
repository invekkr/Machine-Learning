# Module 01: Data Handling — Understanding Your Data

Before training a single machine learning model or writing complex transformations, you must understand the **structure, health, and statistical properties** of your data.

This process answers critical questions:
- How big is the dataset?
- What are the data types of each feature?
- Are there missing values or duplicate records?
- What are the ranges, spreads, and typical values?
- How do numerical features correlate with one another?

---

## 1. Structural Inspection

### 1.1 `df.shape`
* **What it is:** A tuple representing `(number of rows, number of columns)`.
* **Why it matters:** Gives you the scale of the problem immediately. For example, `(891, 12)` indicates 891 observations and 12 features.

```python
rows, cols = df.shape
print(f"Dataset has {rows} rows and {cols} columns.")
```

### 1.2 `df.columns`
* **What it is:** An index object containing all column names.
* **Why it matters:** Helps spot formatting bugs (such as unexpected leading/trailing whitespace like `" Age "`) and lets you iterate or filter column subsets.

### 1.3 `df.dtypes`
* **What it is:** The data type assigned to each column (`int64`, `float64`, `object`, `bool`, `datetime64`).
* **Why it matters:** Machine learning models require numerical input. If a numeric column is mistakenly parsed as `object` (string) due to dirty characters (e.g., `"$10.50"`), mathematical operations will fail.

### 1.4 `df.info()`
* **What it is:** A unified diagnostic summary that prints:
  - Total number of rows and column count
  - Column names and non-null counts
  - Data types (`dtypes`)
  - Approximate memory footprint
* **Why it matters:** The best single method to run immediately after loading a dataset to identify missing data and type mismatches.

---

## 2. Summary Statistics with `df.describe()`

`describe()` computes descriptive statistics summarizing the central tendency, dispersion, and shape of a dataset's distribution.

### Numerical Summary (Default)
By default, `df.describe()` operates on numerical columns and provides:
- **`count`**: Number of non-missing values
- **`mean`**: Arithmetic average
- **`std`**: Standard deviation (spread of data)
- **`min`**: Minimum value observed
- **`25%` (Q1)**: First quartile (25% of data is below this value)
- **`50%` (Median)**: Middle value (50th percentile)
- **`75%` (Q3)**: Third quartile (75% of data is below this value)
- **`max`**: Maximum value observed

```python
df.describe()
```

### Including Categorical Features (`include='all'`)
Categorical features cannot have a mean or standard deviation. Passing `include='all'` adds categorical diagnostics:
- **`unique`**: Number of distinct categories
- **`top`**: Most frequent category (mode)
- **`freq`**: Frequency of the most frequent category

```python
df.describe(include="all")
```

---

## 3. Missing Value Detection

Real-world datasets are plagued by missing entries (unrecorded observations, sensor failures, optional survey fields).

### 3.1 `df.isnull()` / `df.isna()`
Returns a boolean DataFrame of the same shape, where each cell is `True` if missing (`NaN`) and `False` otherwise.

### 3.2 `df.isnull().sum()`
Because Python treats `True` as `1` and `False` as `0`, summing boolean masks counts the exact number of missing entries per column:

```python
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_percent})
```

> **Titanic Insight:**
> - `Age` is missing ~19.9% of values (requires imputation).
> - `Cabin` is missing ~77.1% of values (too sparse; often dropped or converted into a binary "has cabin" flag).
> - `Embarked` is missing only 2 values (can be imputed with the mode).

---

## 4. Duplicate Record Detection

Duplicates skew model training, artificially inflating the importance of repeated observations and leading to data leakage if duplicates cross train/test splits.

### 4.1 `df.duplicated()`
Returns a boolean Series indicating whether each row is an identical duplicate of a prior row.

```python
# Check total duplicate rows across all columns
total_duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {total_duplicates}")
```

To inspect duplicated rows:
```python
# View the duplicated rows
df[df.duplicated(keep=False)]
```

---

## 5. Unique Values & Cardinality

Understanding the distinct values of a feature dictates how it should be encoded for machine learning.

### 5.1 `unique()`
Returns an array of all distinct values in a column.
```python
df["Embarked"].unique()
# Returns: array(['S', 'C', 'Q', nan], dtype=object)
```

### 5.2 `nunique()`
Returns the *count* of unique values.
```python
df.nunique()
```
- **Low cardinality** (e.g., `Sex` = 2, `Pclass` = 3): Ideal for One-Hot Encoding.
- **High cardinality** (e.g., `Ticket` = 681, `Name` = 891): Requires domain extraction or target encoding; direct one-hot encoding creates extreme dimensionality.

### 5.3 `value_counts()`
Counts the frequency of each distinct value in a categorical column.
```python
# Raw counts
df["Sex"].value_counts()

# Proportions / Percentages
df["Sex"].value_counts(normalize=True) * 100
```

---

## 6. Correlation Analysis

### 6.1 What is Correlation?
Correlation measures the strength and direction of a **linear relationship** between two numerical variables.

The most standard metric is the **Pearson Correlation Coefficient ($r$)**:
$$r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}$$

* **$r = +1.0$**: Perfect positive linear relationship (as $X$ increases, $Y$ increases proportionately).
* **$r = 0.0$**: No linear relationship.
* **$r = -1.0$**: Perfect negative linear relationship (as $X$ increases, $Y$ decreases proportionately).

### 6.2 `df.corr()` & The Correlation Matrix
```python
# Compute correlation matrix on numerical features only
corr_matrix = df.corr(numeric_only=True)
corr_matrix
```

### 6.3 Correlation Heatmap
A correlation matrix is much easier to interpret visually with a heatmap:
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Numerical Feature Correlation Matrix")
plt.show()
```

---

## 7. Crucial Concept: Correlation vs. Causation

One of the most dangerous traps in data science is assuming that because two variables are strongly correlated, one causes the other.

> **Golden Rule:** **Correlation does not imply causation.**

### Why Correlation Does NOT Equal Causation:

1. **Confounding Variables (Third-Variable Problem):**
   * *Example:* There is a strong positive correlation between ice cream sales and shark attacks.
   * *The Confounder:* **Hot summer weather**. When it is hot, more people eat ice cream AND more people swim in the ocean. Ice cream does not attract sharks.

2. **Directionality Problem:**
   * Even if a causal link exists, correlation does not tell you if $A \rightarrow B$ or $B \rightarrow A$.

3. **Spurious / Accidental Correlations:**
   * With enough variables, coincidences appear statistically significant purely by chance.

### In the Titanic Dataset:
* `Pclass` and `Fare` have a moderate negative correlation ($r \approx -0.55$). A lower class number (1st class) had much higher fares.
* `Fare` has a positive correlation with `Survived` ($r \approx +0.26$).
* Did paying more money *cause* a passenger's biological body to survive? No! Wealthier passengers were housed on upper decks near the lifeboats and were prioritized during evacuation ("women and children in 1st class first"). `Fare` is a proxy for socioeconomic privilege and cabin proximity, not a direct medical cause of survival.

---

## Summary Checklist
- [x] Use `shape`, `columns`, `dtypes`, and `info()` for structural overview.
- [x] Run `describe()` for numerical distributions and `describe(include='all')` for categorical modes.
- [x] Quantify missingness using `isnull().sum()` and duplicate rows with `duplicated().sum()`.
- [x] Check cardinality using `nunique()` and categorical balance using `value_counts(normalize=True)`.
- [x] Compute linear correlations with `corr(numeric_only=True)` and visualize via heatmaps.
- [x] Never claim causation without experimental control or verified domain mechanisms.
