# Discretization & Binarization

In previous modules, we explored how to convert **categorical data into numbers** so algorithms could process them (for example, converting `City` into numerical binary flags using One-Hot Encoding).

In this module, we explore the exact reverse journey: **converting continuous numerical numbers into discrete groups or categories**.

```text
Categorical Feature  ────────►  Numerical Representation  (One-Hot / Ordinal Encoding)
Continuous Numerical ────────►  Discrete Representation   (Discretization / Binning)
```

---

## 1. Big Picture

In machine learning, continuous numerical features (like Age, Salary, Temperature, or Exam Marks) can take infinitely many fine-grained values. However, treating every minor decimal variation as distinct is not always beneficial:

```text
Raw Continuous Values:     23.4,  24.1,  25.7,  26.2,  34.5,  52.1,  71.8
                                       │
                                       ▼
Discrete Intervals / Bins:   [20–30)        [20–30)   [30–40) [50–60) [70–80)
                               │              │          │       │       │
Categorical Meaning:         Young          Young      Adult   Senior  Elder
                               │              │          │       │       │
Numerical Encoding:            0              0          1       3       4
```

This transformation process is called **Discretization** (or **Binning** / **Bucketization**), and its special two-class counterpart is known as **Binarization**.

---

## 2. What is Discretization?

### The Core Idea:
What happens when we convert a continuous numerical feature into a finite number of discrete intervals?

Consider employee ages:
$$17, \ 22, \ 27, \ 34, \ 42, \ 51, \ 68$$

Instead of forcing a model to treat every number as an individual continuous point, we can group them into discrete cohorts:
- **0–18:** Young
- **19–30:** Adult
- **31–45:** Middle-aged
- **46+:** Senior

The original raw numbers are now represented using a small, finite set of discrete categories or bins.

> [!NOTE]
> **Formal Definition:**  
> **Discretization** (also known as **Binning** or **Bucketization**) is the process of converting a continuous numerical feature into a finite number of discrete intervals or bins.

### Key Terminology:
- **Continuous Numerical Variable:** A numerical feature that can take any value within a given range (e.g., $24.183\dots$ years old).
- **Discrete Intervals:** Defined numerical spans with lower and upper boundaries (e.g., $[20, 30)$).
- **Bins / Buckets:** The individual containers or categories that represent each interval.
- **Bucketization:** An industry synonym for binning often used in production ML systems.

---

## 3. Why Do We Use Binning?

Binning is used for five major reasons in practical machine learning:

### 1. Simplify Data
Instead of handling noisy decimal variations ($23.4, 24.1, 25.7$), values are summarized into a clean cohort ($20–30$). This eliminates minor measurement noise and reduces model complexity.

### 2. Handle Extreme Values / Reduce Sensitivity to Individual Values
In financial data, an extreme income of ₹10,00,000 can dominate distance calculations in KNN or pull regression lines off course. Grouping values into a boundary bin (`₹50k+`) bounds its influence.

> [!IMPORTANT]
> **CRITICAL TAKEAWAY:**  
> Binning does **NOT** magically remove an outlier from the dataset.  
> It simply changes how the value is represented. An extreme outlier and a high earner both receive the same boundary bin label, preventing numerical leverage from exploding.

### 3. Capture Non-Linear Relationships
Many features exhibit non-linear or piecewise relationships with target outcomes:
- **Credit Card Default by Income Bracket:**
  - ₹0 – ₹30,000: High risk
  - ₹30,000 – ₹70,000: Low risk
  - ₹70,000 – ₹1,50,000: Very low risk
  - ₹1,50,000+: Moderate risk (higher credit limits)
  
A simple linear model cannot fit a single straight line through this zig-zag pattern. By converting income into 4 discrete bins, the model can assign an independent weight to each range!

### 4. Make Results Easier to Interpret
Model outputs become much more explainable to non-technical stakeholders when formulated in terms of brackets (e.g., *"Customers in the 30–40 age bracket have a 25% churn rate"*).

### 5. Domain / Business Interpretation
Many industries already operate with statutory or institutional tiers (e.g., Income Tax Slabs: *0%, 5%, 20%, 30%*; Credit Ratings: *AAA, AA, A, BBB*).

---

## 4. Equal Width Binning

### Definition:
> **Equal Width Binning** divides the entire numerical range of a feature into intervals that all have the **exact same numerical width**.

### Formula:
$$\text{Bin Width} = \frac{\text{Maximum} - \text{Minimum}}{\text{Number of Bins}}$$

$$\text{Bin Width} = \frac{x_{\max} - x_{\min}}{k}$$

Where:
- $x_{\min}$: Minimum observed value in the feature.
- $x_{\max}$: Maximum observed value in the feature.
- $k$: Desired number of bins (`n_bins`).

### Range Calculation Example:
- Minimum = $0$
- Maximum = $100$
- Number of bins = $5$

$$\text{Bin Width} = \frac{100 - 0}{5} = 20$$

Resulting intervals:
$$[0, 20), \quad [20, 40), \quad [40, 60), \quad [60, 80), \quad [80, 100]$$

> [!WARNING]
> **Equal Width means:** Each bin covers the **same numerical span**.  
> It does **NOT** mean each bin contains the same number of observations!

### Manual Dataset Example & Boundary Handling:
Consider 9 ages:
$$10, \ 15, \ 20, \ 25, \ 30, \ 35, \ 40, \ 45, \ 50$$
We want **$k = 4$ bins**.

1. $\text{Min} = 10, \ \text{Max} = 50$
2. $\text{Range} = 50 - 10 = 40$
3. $\text{Bin Width} = 40 / 4 = 10$
4. **Boundary Definitions:** To avoid ambiguity, intervals are left-closed and right-open $[a, b)$, with the final bin closed on both ends $[a, b]$:
   - Bin 0: $[10, 20)$
   - Bin 1: $[20, 30)$
   - Bin 2: $[30, 40)$
   - Bin 3: $[40, 50]$

| Value | Belongs to Interval | Assigned Bin |
| :---: | :---: | :---: |
| **10** | $[10, 20)$ | **Bin 0** |
| **15** | $[10, 20)$ | **Bin 0** |
| **20** | $[20, 30)$ | **Bin 1** |
| **25** | $[20, 30)$ | **Bin 1** |
| **30** | $[30, 40)$ | **Bin 2** |
| **35** | $[30, 40)$ | **Bin 2** |
| **40** | $[40, 50]$ | **Bin 3** |
| **45** | $[40, 50]$ | **Bin 3** |
| **50** | $[40, 50]$ | **Bin 3** |

### Advantages:
- Extremely intuitive and easy for humans to calculate and understand.
- Preserves the original scale of the feature (every bucket represents the same step size).

### Limitations:
- Vulnerable to skewed distributions or outliers. If one value sits at $1000$ while everyone else is under $50$, Bin 0 gets $99\%$ of the data and the middle bins remain completely empty!

---

## 5. Equal Frequency Binning (Quantile Binning)

### Definition:
> **Equal Frequency Binning** (also called **Quantile Binning**) divides sorted data into intervals such that **each bin contains approximately the same number of observations**.

Instead of dividing the numerical range equally, we divide the data points according to their **statistical quantiles / percentiles**.

### Manual Calculation on a Sorted Dataset:
Consider 10 values:
$$[10, \ 12, \ 15, \ 18, \ 20, \ 25, \ 30, \ 40, \ 50, \ 60]$$

#### Case A: $k = 2$ Bins (Median Split)
We want $10 / 2 = 5$ observations per bin:
- Sort values: $10, 12, 15, 18, 20 \quad \vert \quad 25, 30, 40, 50, 60$
- Median split point = $22.5$
  - **Bin 0:** $[10, 22.5] \longrightarrow$ Contains **5 observations** ($10, 12, 15, 18, 20$)
  - **Bin 1:** $(22.5, 60] \longrightarrow$ Contains **5 observations** ($25, 30, 40, 50, 60$)

#### Case B: $k = 5$ Bins (Quintiles)
We want $10 / 5 = 2$ observations per bin:
- **Bin 0:** $[10, 13.5] \longrightarrow 2$ values ($10, 12$)
- **Bin 1:** $(13.5, 19.0] \longrightarrow 2$ values ($15, 18$)
- **Bin 2:** $(19.0, 27.5] \longrightarrow 2$ values ($20, 25$)
- **Bin 3:** $(27.5, 45.0] \longrightarrow 2$ values ($30, 40$)
- **Bin 4:** $(45.0, 60.0] \longrightarrow 2$ values ($50, 60$)

### Advantages:
- Handles skewed data gracefully.
- Guarantees that every bin has sufficient training observations (no empty bins).

### Limitations:
- Bin widths can vary drastically. Dense clusters get narrow bins, while sparse tails get massive bins.

---

## 6. Equal Width vs. Equal Frequency

| Concept | Equal Width Binning | Equal Frequency Binning |
| :--- | :--- | :--- |
| **What is equal?** | **Numerical width / range** of each bin | **Number of observations** in each bin |
| **Bin width** | Exactly identical across all bins | Varies widely across bins |
| **Number of observations** | Can differ significantly (empty bin risk) | Approximately equal ($N / k$) |
| **Sensitivity to skewed data** | Very high (outliers empty out middle bins) | Very low (adapts naturally to skewness) |
| **Typical use** | When the numerical interval has direct real-world meaning | When data is skewed or heavily clustered |
| **Main advantage** | Simple to interpret; preserves feature geometry | Balanced data distribution across all buckets |
| **Main limitation** | Can produce severely unbalanced observation counts | Unequal interval spans make intervals harder to compare |

> [!TIP]
> **The Core Memory Rule:**  
> - **Equal Width** $\longrightarrow$ Same **numerical range**  
> - **Equal Frequency** $\longrightarrow$ Similar **number of data points**

---

## 7. K-Means Binning

### Definition:
> **K-Means Binning** applies 1-dimensional K-Means clustering to discover natural groupings in the data, using the resulting cluster boundaries as discrete bins.

### Step-by-Step Mechanism:
1. **Choose number of bins $K$:** (e.g., $K = 3$).
2. **Apply 1D K-Means:** The algorithm iteratively locates $K$ cluster centroids that minimize the squared distance to their assigned points.
3. **Assign points to clusters:** Each data point is assigned to its nearest centroid.
4. **Establish bin boundaries:** The boundary edges are placed at the **midpoints between adjacent cluster centroids** (plus the minimum and maximum of the dataset).
5. **Interpret clusters as bins:** Points assigned to Cluster 0 become Bin 0, Cluster 1 become Bin 1, etc.

```text
Data:         [10, 12, 14, 15]        [50, 52, 55, 57]        [100, 105, 110, 115]
                     │                       │                         │
Centroids:         12.75                   53.5                      107.5
                     │                       │                         │
Boundaries: 10 ────────────► Midpoint = 33.1 ──────────► Midpoint = 80.5 ────────► 115
                     │                       │                         │
Bins:             [ Bin 0 ]               [ Bin 1 ]                 [ Bin 2 ]
```

### When Is It Useful?
When numerical data **naturally clumps into distinct clusters** separated by large empty gaps (e.g., Customer Spending: *Budget, Mid-Market, Enterprise*).

### Why It Differs from Equal Width and Equal Frequency:
- Equal Width divides the *range* blindly.
- Equal Frequency divides the *ranks* blindly.
- **K-Means Binning discovers the boundaries from the natural geometry of the data itself.**

---

## 8. Custom / Domain-Based Binning

Not all boundaries should be found statistically. Often, the most meaningful boundaries come directly from **human domain expertise, business rules, or legal statutes**.

### Example: Real-World Age Groups
```text
0 – 17     ──► Minor (Dependent)
18 – 30    ──► Young Adult
31 – 50    ──► Adult
51+        ──► Senior
```

### Why Domain Bins Make Sense:
- **Banking / Credit:** Age 18 is legally required to sign a credit contract.
- **Healthcare:** Risk categories (e.g., Blood Pressure: Normal $<120$, Elevated $120–129$, Stage 1 Hypertension $130–139$).
- **Tax Policy:** Tax brackets defined by revenue thresholds.

### Trade-Offs:
- **Advantage:** Maximum human interpretability and direct business alignment.
- **Limitation:** Subjective; boundaries depend on human judgment rather than empirical patterns.

---

## 9. Scikit-Learn `KBinsDiscretizer`

Scikit-Learn provides `KBinsDiscretizer` in `sklearn.preprocessing`:

```python
from sklearn.preprocessing import KBinsDiscretizer

kbd = KBinsDiscretizer(
    n_bins=5,
    strategy='quantile',
    encode='ordinal'
)
```

### Parameter Breakdown:

#### 1. `n_bins` (int or array-like, default=5)
The number of discrete bins to create. If set to $5$, the feature is divided into 5 intervals ($0, 1, 2, 3, 4$).

#### 2. `strategy` (`'uniform'`, `'quantile'`, `'kmeans'`, default=`'quantile'`)
Controls how the bin boundaries are calculated:
- **`'uniform'`:** Equal Width Binning.
- **`'quantile'`:** Equal Frequency Binning.
- **`'kmeans'`:** 1D K-Means Cluster Midpoints.

#### 3. `encode` (`'ordinal'`, `'onehot'`, `'onehot-dense'`, default=`'onehot'`)
Controls how the discrete bins are numerically output:
- **`'ordinal'`:** Returns a single column containing integer indices ($0, 1, 2, \dots, k-1$).
- **`'onehot'`:** Returns a sparse binary matrix with $k$ columns.
- **`'onehot-dense'`:** Returns a dense 2D NumPy array with $k$ columns.

---

## 10. `pandas.cut` and `pandas.qcut`

In exploratory data analysis, Pandas provides two handy functions:

### 1. `pd.cut()`: For Equal-Width or Custom Bins
```python
import pandas as pd

# Custom domain intervals
ages = pd.Series([15, 22, 35, 58, 72])
custom_bins = [0, 18, 30, 50, 100]
labels = ['Young', 'Adult', 'Middle-aged', 'Senior']

binned_age = pd.cut(ages, bins=custom_bins, labels=labels, right=False)
```

### 2. `pd.qcut()`: For Equal-Frequency (Quantile) Bins
```python
# 4 equal-frequency quartiles
quartiles = pd.qcut(ages, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### Pandas vs. Scikit-Learn `KBinsDiscretizer`:
| Feature | Pandas (`pd.cut`, `pd.qcut`) | Scikit-Learn (`KBinsDiscretizer`) |
| :--- | :--- | :--- |
| **Primary Purpose** | Fast data exploration & analysis | Production machine learning pipelines |
| **Output Type** | Categorical Series or String labels | Numerical indices (`ordinal`) or binary matrices (`onehot`) |
| **Train/Test Handling** | Manual; prone to data leakage | Automatic `fit()` on train, `transform()` on test |
| **Pipeline Integration** | Requires custom wrappers | Natively supported in `ColumnTransformer` & `Pipeline` |

---

## 11. Information Loss from Binning

Binning permanently discards numerical detail.

Consider four customers:
$$29, \ 30, \ 31, \ 32$$

In continuous numbers:
$$29 \ne 32$$
The model knows Customer 4 is 3 years older than Customer 1.

Now suppose we apply binning with interval `[20, 40)`:
$$29 \longrightarrow \text{Bin 1}, \quad 30 \longrightarrow \text{Bin 1}, \quad 31 \longrightarrow \text{Bin 1}, \quad 32 \longrightarrow \text{Bin 1}$$

After binning:
$$\text{Customer 1} = \text{Customer 4} = \text{Bin 1}$$
The 3-year age gap is permanently erased. The model now treats both customers as completely identical.

### The Fundamental Trade-Off:
$$\text{More Simplicity} \quad \longleftrightarrow \quad \text{Less Numerical Precision}$$

- **Too few bins ($k = 2$):** Massive information loss. Important within-group differences are destroyed.
- **Too many bins ($k = 50$):** Minimal simplification. Increases overfitting without providing meaningful cohort summaries.

---

## 12. Binning Does Not Remove Outliers

A common beginner misconception is that binning removes outliers.

Consider this data:
$$10, \ 20, \ 30, \ 40, \ 1000$$

If we apply 4 bins where the top bin is `[40, max]`:
- $10 \to \text{Bin 0}$
- $20 \to \text{Bin 1}$
- $30 \to \text{Bin 2}$
- $40 \to \text{Bin 3}$
- $1000 \to \text{Bin 3}$

Notice:
- $1000$ was **not** deleted from the dataset.
- Its representation changed from $1000$ to **`Bin 3`**.
- Its extreme numerical leverage is capped, but the row is still fully present.

$$\mathbf{Outlier \ Removal \ne Binning}$$

---

## 13. Binning vs. Mathematical Transformation

Do not confuse Binning with Mathematical Transformations (studied in Module 03, Topic 07):

```text
Raw Value: x = 100
             │
             ├──► Mathematical Transformation (Log):  x' = ln(100) ≈ 4.605   (Still a continuous number!)
             │
             └──► Discretization (Binning):           x' = Bin 3              (A discrete group / category!)
```

| Dimension | Mathematical Transformation (Log, Sqrt) | Discretization / Binning |
| :--- | :--- | :--- |
| **Output Type** | Continuous numerical float ($\ln(x), \sqrt{x}$) | Discrete integer category or binary flag |
| **Information Preserved** | Order and fine-grained differences preserved | Within-bin differences permanently discarded |
| **Core Mechanism** | Smooth mathematical function $f(x)$ | Partitioning into discrete intervals $[a, b)$ |
| **Primary Goal** | Reduce skewness, linearize relationships | Group data, simplify, create piecewise bins |

---

## 14. Binning vs. Binarization

| Feature | Discretization (Binning) | Binarization |
| :--- | :--- | :--- |
| **Number of Groups** | Multiple groups ($k \ge 3$) | Exactly **two** groups ($0$ and $1$) |
| **Output** | Ordinal ($0, 1, 2, \dots$) or One-Hot vectors | Binary flag ($0$ or $1$) |
| **Main Purpose** | Group continuous values into cohort intervals | Answer a single threshold-based YES/NO question |
| **Example** | Age $\longrightarrow$ Young, Adult, Senior | Marks $\ge 40 \longrightarrow$ Pass ($1$), Fail ($0$) |
| **Typical Tool** | `KBinsDiscretizer`, `pd.cut` | `Binarizer` |

---

## 15. Binarization

### Formal Idea:
> **Binarization** is the process of converting numerical values into two groups, usually represented as **$0$ and $1$**, based on a specified threshold.

### Real-World Examples:
- **Exam Marks:** $< 40 \longrightarrow 0$ (Fail), $\ge 40 \longrightarrow 1$ (Pass).
- **Adult Status:** $< 18 \longrightarrow 0$ (Minor), $\ge 18 \longrightarrow 1$ (Adult).
- **Customer Activity:** $\text{Purchases} == 0 \longrightarrow 0$ (Inactive), $\text{Purchases} > 0 \longrightarrow 1$ (Active).

---

## 16. Scikit-Learn `Binarizer`

Scikit-Learn provides `Binarizer` in `sklearn.preprocessing`:

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=40.0)
```

### Critical Implementation Rule on Threshold Inequality:
Scikit-Learn's `Binarizer` implements this exact mathematical condition:
$$\text{Output} = \begin{cases} 1 & \text{if } x > \text{threshold} \\ 0 & \text{if } x \le \text{threshold} \end{cases}$$

> [!IMPORTANT]
> **BE PRECISE ABOUT THE INEQUALITY:**  
> Notice that values **equal to the threshold are mapped to 0**!  
> If you set `threshold=40.0`, a score of `40.0` becomes `0` (because $40.0$ is not strictly greater than $40.0$).  
> If you want a score of $40$ to pass as $1$ ($\text{Marks} \ge 40$), set the threshold to **`39.5`** or **`39.0`**!

---

## 17. Train / Test Data & Data Leakage

When preprocessing continuous features with `KBinsDiscretizer`, you must follow the strict **Split-First Protocol**:

```text
WRONG (DATA LEAKAGE):
Fit KBinsDiscretizer on FULL dataset ──► Split Train / Test
(Bin edges calculated from test values leak into training!)

CORRECT:
1. Split data into Train and Test sets FIRST
2. Fit transformer on Train set ONLY:   kbd.fit(X_train)
3. Transform Train set:                X_train_binned = kbd.transform(X_train)
4. Transform Test set with TRAIN edges: X_test_binned = kbd.transform(X_test)
```

> [!NOTE]
> If you fit separate bin boundaries on the test set, `Bin 1` would represent different numerical ranges in train and test, corrupting model predictions!

---

## 18. Pipeline Integration

Connecting `KBinsDiscretizer` directly to an estimator inside a Scikit-Learn `Pipeline` guarantees that the split-first rule is enforced automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('discretizer', KBinsDiscretizer(n_bins=5, strategy='quantile', encode='ordinal')),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Fit on training data ONLY
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)
```

Inside a `ColumnTransformer`, you can bin selected numerical columns while scaling others and encoding categoricals:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('age_bins', KBinsDiscretizer(n_bins=5, strategy='quantile', encode='onehot-dense'), ['Age']),
        ('fare_scale', StandardScaler(), ['Fare']),
        ('sex_ohe', OneHotEncoder(drop='first', sparse_output=False), ['Sex'])
    ],
    remainder='drop'
)
```

---

## 19. When Should We Use Binning?

1. **Interpretability is Important:** Business stakeholders require policy-aligned categories (credit tiers, customer cohorts).
2. **Domain Categories Matter:** Legal or regulatory boundaries dictate decisions (e.g., Age 18, 65).
3. **Non-Linear Piecewise Effects:** The relationship between feature and target jumps abruptly across ranges.
4. **Extreme Tail Values:** Outliers distort linear models or distance metrics.
5. **Noisy Continuous Measurements:** Minor sensor measurement jitter should be smoothed out.

---

## 20. When Not to Use Binning?

1. **Exact Numerical Information Matters:** When small differences carry critical predictive power (e.g., financial pricing, medical dosage).
2. **Using Tree-Based Models:** Decision Trees, Random Forests, and XGBoost **already find optimal threshold splits naturally**. Binning features beforehand often reduces their splitting precision.
3. **Arbitrary Boundaries:** Placing cuts without domain justification splits identical observations across boundaries ($17.9$ vs $18.0$).
4. **When Validation Accuracy Drops:** If cross-validation shows that continuous scaling outperforms binning, discard the bins.

---

## 21. Common Mistakes to Avoid

1. **Thinking Equal Width Means Equal Observations:** Equal width divides the range, not the data count.
2. **Thinking Equal Frequency Means Equal Spans:** Equal frequency produces narrow bins in dense areas and huge bins in sparse tails.
3. **Believing Binning Removes Outliers:** Outliers are capped into boundary bins, not deleted.
4. **Using Too Few Bins ($k=2$):** Causes severe information loss.
5. **Using Too Many Bins ($k=50$):** Increases overfitting without meaningful simplification.
6. **Forgetting Information Loss:** Assuming binning preserves full numerical precision.
7. **Fitting Preprocessing Separately on Test Data:** Causing data leakage.
8. **Confusing Binning with Mathematical Transformations:** Log changes continuous shape; binning creates discrete groups.
9. **Confusing Binning with Binarization:** Binarization produces exactly 2 classes ($0/1$).
10. **Using Bins Without Validating Model Benefit:** Applying binning blindly without checking holdout metrics.
11. **Not Checking `Binarizer` Threshold Behavior:** Overlooking that $x = \text{threshold} \implies 0$ (strict greater-than inequality).
12. **Treating Automatic Binning as Superior to Domain Binning:** Overlooking that legal and business rules often outperform statistical cuts.

---

## 22. Quick Reference Table

| Technique | Input | Output | Main Idea | Python / sklearn Tool |
| :--- | :--- | :--- | :--- | :--- |
| **Equal Width** | Continuous numerical | Discrete bins | Bins have the exact same numerical range | `KBinsDiscretizer(strategy='uniform')` |
| **Equal Frequency** | Continuous numerical | Discrete bins | Bins contain equal observation counts | `KBinsDiscretizer(strategy='quantile')` |
| **K-Means Binning** | Continuous numerical | Discrete bins | 1D clusters determine bin boundaries | `KBinsDiscretizer(strategy='kmeans')` |
| **Custom Binning** | Continuous numerical | Discrete bins | Human-defined business / legal boundaries | `pd.cut(bins=[...])` |
| **KBinsDiscretizer** | 2D numerical array | Binned matrix | Production Scikit-Learn transformer | `sklearn.preprocessing.KBinsDiscretizer` |
| **Binarization** | Continuous numerical | Binary 0 / 1 | Converts numbers to 2 classes via threshold | `Binarizer(threshold=...)` |
| **Binarizer** | 2D numerical array | Binary 0 / 1 | Production Scikit-Learn binarizer ($x > t \implies 1$) | `sklearn.preprocessing.Binarizer` |

---

## 23. Final Mental Model

```text
                           Numerical Feature
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        Keep Numerical       Mathematical        Discretization
            Values          Transformation       (Binning)
        (Scale / Clean)     (log, sqrt, 1/x)          │
                                                      ├── Equal Width
                                                      ├── Equal Frequency
                                                      ├── K-Means
                                                      └── Custom / Domain
                                                              │
                                                              ▼
                                                   Multiple Discrete Groups


                           Numerical Feature
                                   │
                                   ▼
                              Binarization
                                   │
                                   ▼
                           Two Groups: 0 / 1
```

---

## Where This Fits in the ML Workflow

```text
Raw Data
   │
   ▼
Data Cleaning (Missing values, duplicates)
   │
   ▼
Exploratory Data Analysis (EDA: Univariate, Bivariate, Multivariate)
   │
   ▼
Feature Engineering
   │
   ├── Feature Scaling (StandardScaler, MinMaxScaler)      [Topic 01 & 02]
   ├── Categorical Encoding (OneHot, Ordinal, Label)       [Topic 03 & 04]
   ├── Mathematical Transformations (Log, Sqrt)           [Topic 07]
   │
   └── Numerical Discretization & Binarization            [THIS TOPIC 08]
         │
         ├── ColumnTransformer (Route columns)            [Topic 05]
         └── Pipeline (Automate workflow)                 [Topic 06]
   │
   ▼
Model Training & Holdout Cross-Validation
   │
   ▼
Model Evaluation & Metric Comparison
```
