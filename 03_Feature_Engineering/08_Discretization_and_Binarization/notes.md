# Module 03: Feature Engineering — Encoding Numerical Features (Discretization & Binarization)

In previous modules, we explored how to convert **categorical data into numbers** so algorithms could process them (for example, converting `City` into numerical binary flags using One-Hot Encoding).

In this module, we explore the exact reverse journey: **converting continuous numerical numbers into discrete groups or categories**.

```text
Previous Journey:  Categorical Feature ──────► Numerical Representation  (One-Hot / Ordinal Encoding)
Current Journey:   Continuous Numerical ──────► Discrete Representation   (Discretization / Binning)
```

For instance, consider a continuous feature like `Age`:
$$18, \ 21, \ 25, \ 32, \ 41, \ 56, \ 72$$

Instead of forcing a model to treat every individual integer or decimal as a distinct point along a continuous line, we can segment the values into meaningful groups:
- **18–25:** Young
- **26–40:** Adult
- **41–60:** Middle-Aged
- **61+:** Senior

This transformation is known as **Discretization** (or **Binning**), and its special two-class counterpart is known as **Binarization**.

---

## 1. Numerical → Discrete Representation

In raw data, continuous features often have thousands of unique values with tiny decimal variations:
```text
Customer A: 23.4 years old
Customer B: 24.1 years old
Customer C: 25.7 years old
Customer D: 26.2 years old
```

While computers can calculate with these numbers, treating them as continuous numbers isn't always optimal for every problem. Grouping them into a single interval like `[20, 30)` allows a machine learning model to treat all young adults in their twenties as belonging to a common behavioral cohort.

```text
Continuous Input:     23.4,  24.1,  25.7,  26.2,  34.5,  52.1,  71.8
                                 │
                                 ▼
Discrete Bins:         [20–30)        [20–30)        [30–40) [50–60) [70–80)
                         │              │               │       │       │
Categorical Label:     Young          Young           Adult   Senior  Elder
                         │              │               │       │       │
Ordinal / One-Hot:       0              0               1       3       4
```

---

## 2. Discretization / Binning

> [!NOTE]
> **Formal Definition:**  
> **Discretization** (also called **Binning**) is the process of converting a continuous numerical feature into a finite number of discrete intervals or bins.

Each interval is defined by an upper and lower boundary:
$$\text{Bin } k = [b_k, \ b_{k+1})$$

Any data point $x$ falling within that boundary range is assigned the identifier of that bin.

---

## 3. Why Do We Use Binning?

There are four primary reasons data scientists use binning in machine learning workflows:

### 1. Simplifying Data
Instead of dozens of noisy decimal variations ($23.4, 24.1, 25.7, 26.2$), values are summarized into simple, robust buckets ($20–30$). This reduces noise and minor measurement errors.

### 2. Handling Extreme Values (Outliers)
Extreme values that stretch out into distant tails can be grouped into boundary buckets (such as `Income > ₹10,00,000`).

> [!IMPORTANT]
> **CRITICAL TAKEAWAY:**  
> Binning does **NOT** delete or remove an outlier from the dataset.  
> It simply changes how the value is represented. A billionaire and a multi-millionaire both fall into the `₹10L+` bin, preventing extreme numerical magnitudes from distorting distance or linear calculations.

### 3. Capturing Non-Linear Relationships
Many real-world features have non-linear or piecewise relationships with the target variable.
For example, credit card default risk or insurance claims by salary tier:
- **₹0 – ₹30,000 (Low):** High default risk
- **₹30,000 – ₹70,000 (Medium):** Low default risk
- **₹70,000 – ₹1,50,000 (High):** Very low default risk
- **₹1,50,000+ (Very High):** Moderate risk (often higher leverage)

A simple linear regression model cannot draw a single straight line through this zig-zag pattern. But if salary is converted into 4 distinct one-hot encoded bins, the model assigns a separate, independent weight to each salary bracket!

### 4. Domain Interpretation
Business stakeholders and human operators often think in terms of policy categories rather than continuous formulas (e.g., Credit Scores: *Poor, Fair, Good, Excellent*; Tax Brackets: *10%, 20%, 30%*).

---

## 4. Equal Width Binning

### Understand It First:
Imagine you have a 100-centimeter ruler and you want to cut it into 5 equal pieces. Each piece will be exactly 20 centimeters long.

In **Equal Width Binning**, we divide the entire range from the minimum value to the maximum value into intervals that all have the **exact same numerical width**.

### Formula:
$$\text{Range} = x_{\max} - x_{\min}$$

$$\text{Bin Width} = w = \frac{x_{\max} - x_{\min}}{k}$$

Where:
- $x_{\min}$: The minimum observed value of the feature.
- $x_{\max}$: The maximum observed value of the feature.
- $k$: The desired number of bins (`n_bins`).
- $w$: The width of each individual bin.

### Bin Boundaries:
The boundaries are placed at:
$$b_0 = x_{\min}, \quad b_1 = x_{\min} + w, \quad b_2 = x_{\min} + 2w, \quad \dots, \quad b_k = x_{\max}$$

### Small Manual Example:
Suppose we have an `Age` feature for 9 people:
```text
10,  15,  20,  25,  30,  35,  40,  45,  50
```
We want to create **$k = 4$ equal-width bins**.

1. **Find Min and Max:** $x_{\min} = 10, \quad x_{\max} = 50$
2. **Calculate Range:** $\text{Range} = 50 - 10 = 40$
3. **Calculate Bin Width:** $w = \frac{40}{4} = 10$
4. **Determine Boundaries:**
   - Bin 0: $[10, 20)$
   - Bin 1: $[20, 30)$
   - Bin 2: $[30, 40)$
   - Bin 3: $[40, 50]$

5. **Assign Each Observation:**

| Observation Value | Bin Interval | Assigned Bin Index |
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
- Very simple to calculate and easy for humans to interpret.
- Preserves the intuitive scale of the feature (each bucket covers the exact same numerical span).

### Limitations:
- Highly vulnerable to skewed data or outliers. If a single outlier sits at $1000$ while everyone else is between $10$ and $50$, almost all observations will end up crammed into Bin 0, leaving the other bins empty!

> [!WARNING]
> **Equal Width means:** Each bin covers the **same numerical span**.  
> It does **NOT** mean each bin contains the same number of observations!

---

## 5. Equal Frequency (Quantile) Binning

### Understand It First:
Suppose you have 100 students taking an exam and you want to award grades across 5 bins. Instead of caring about the test score numbers, you place the bottom 20 students in Grade F, the next 20 students in Grade D, the next 20 in Grade C, and so on.

In **Equal Frequency Binning** (also called **Quantile Binning**), we divide sorted data so that **each bin contains approximately the same number of observations**.

### Proper Definition:
> **Equal Frequency Binning** divides sorted data into intervals such that each bin contains an approximately equal count of data points (samples).

The boundaries are determined by the statistical **quantiles / percentiles** of the feature:
- 2 bins $\longrightarrow$ Split at the 50th percentile (Median).
- 4 bins $\longrightarrow$ Split at the 25th, 50th, and 75th percentiles (Quartiles).
- 10 bins $\longrightarrow$ Split at every 10th percentile (Deciles).

### Small Manual Example on Skewed Data:
Consider a heavily right-skewed dataset with 10 values:
```text
1,  2,  2,  3,  3,  4,  5,  6,  50,  100
```
Notice the massive jump at the end ($50$ and $100$).

Suppose we want **$k = 2$ bins**:

#### If we used Equal Width:
- $x_{\min} = 1, \ x_{\max} = 100 \implies \text{Width} = \frac{100 - 1}{2} = 49.5$
- Bin 0: $[1, 50.5) \longrightarrow$ Contains **9 values** ($1, 2, 2, 3, 3, 4, 5, 6, 50$)!
- Bin 1: $[50.5, 100] \longrightarrow$ Contains **only 1 value** ($100$)!
- **Result:** Severely unbalanced groups!

#### Using Equal Frequency (Quantile):
We want $10 / 2 = 5$ observations per bin:
- Sort values: $1, 2, 2, 3, 3 \quad \vert \quad 4, 5, 6, 50, 100$
- Split at the median:
  - **Bin 0:** $[1, 3.5] \longrightarrow$ Contains **5 values** ($1, 2, 2, 3, 3$)
  - **Bin 1:** $(3.5, 100] \longrightarrow$ Contains **5 values** ($4, 5, 6, 50, 100$)
- **Result:** Perfectly balanced observation counts!

### Advantages:
- Handles skewed distributions gracefully.
- Guarantees that every bin has sufficient data to train model parameters without empty buckets.

### Limitations:
- Bin widths can be wildly unequal. In the example above, Bin 0 has a width of only $2.5$, while Bin 1 has a width of $96.5$!

---

## 6. Equal Width vs. Equal Frequency

| Dimension | Equal Width Binning (`uniform`) | Equal Frequency Binning (`quantile`) |
| :--- | :--- | :--- |
| **Core Concept** | Each bin has the **same numerical width** | Each bin has **similar observation counts** |
| **Boundary Rule** | Calculated from range: $\frac{x_{\max} - x_{\min}}{k}$ | Calculated from percentiles / quantiles |
| **Observation Counts** | Can vary dramatically across bins | Approximately equal across all bins |
| **Interval Widths** | Exactly equal ($w_0 = w_1 = \dots = w_k$) | Widely unequal depending on data density |
| **Best Used When** | The underlying numerical scale has direct meaning | The data is highly skewed or has dense clusters |
| **Risk** | Outliers cause empty or overcrowded bins | Bins with identical values or tied ranks |

> [!TIP]
> **Memory Trick:**  
> - **Width** $\longrightarrow$ Same **Range size**  
> - **Frequency** $\longrightarrow$ Same **Number of data points**

---

## 7. K-Means Binning

### Understand It First:
Sometimes data naturally clusters into distinct groups with wide gaps between them.

For example, consider daily website transactions (in ₹):
```text
Cluster 1 (Micro-transactions):    ₹10,  ₹12,  ₹14,  ₹15
Cluster 2 (Standard retail):        ₹50,  ₹52,  ₹55,  ₹57
Cluster 3 (Bulk enterprise):       ₹100, ₹105, ₹110, ₹115
```

Neither equal width nor equal frequency accounts for the fact that these numbers naturally form three separate clumps separated by empty deserts.

**K-Means Binning** runs a 1D K-Means clustering algorithm directly on the numerical feature:
1. It identifies $k$ cluster centers (centroids) in 1-dimensional space.
2. The bin boundaries are placed at the **midpoints between adjacent cluster centroids**.
3. All points closest to centroid 1 fall into Bin 0, points closest to centroid 2 fall into Bin 1, etc.

```text
Data:     [10, 12, 14, 15]          [50, 52, 55, 57]          [100, 105, 110, 115]
                 │                         │                           │
Centroids:      12.75                    53.5                        107.5
                 │                         │                           │
Boundaries: 10 ───────► Midpoint = 33.1 ────────► Midpoint = 80.5 ──────────► 115
                 │                         │                           │
Bins:         [ Bin 0 ]                 [ Bin 1 ]                   [ Bin 2 ]
```

### Proper Definition:
> **K-Means Binning** applies 1-dimensional K-Means clustering to group numerical values into natural clusters, which are then used as discrete bin intervals.

### When Is It Useful?
- When the feature distribution is multimodal (contains multiple peaks separated by low-density valleys).
- When you want data-driven boundaries that adapt to the natural geometry of the numbers.

### Limitations:
- Computationally more expensive than uniform or quantile binning because it iteratively trains an optimization algorithm.
- If data is uniform or strictly linear, K-Means binning adds unnecessary complexity.

---

## 8. Custom / Domain-Based Binning

Not all bin boundaries should be discovered statistically by an algorithm. Often, the most powerful and meaningful boundaries come directly from **human domain knowledge, legal statutes, or business rules**.

### Example: Age Groups in Healthcare & Insurance
```text
Age < 18    ──► Minor / Dependent
18 – 24     ──► Young Adult / College
25 – 49     ──► Working Adult
50 – 64     ──► Pre-Retirement
65+         ──► Senior / Medicare Eligible
```

These boundaries are not chosen by equal width or quantiles—they reflect real-world legal ages of majority, insurance eligibility brackets, and medical risk milestones.

### Proper Definition:
> **Custom / Domain-Based Binning** uses predefined, human-specified boundaries based on expert domain knowledge, business rules, or statutory criteria.

### Python Implementation with `pd.cut()`:
```python
import pandas as pd

ages = pd.Series([12, 19, 28, 54, 71])
bins = [0, 18, 25, 50, 65, 120]
labels = ['Minor', 'Young Adult', 'Adult', 'Pre-Senior', 'Senior']

age_categories = pd.cut(ages, bins=bins, labels=labels, right=False)
```

---

## 9. Scikit-Learn `KBinsDiscretizer`

Scikit-Learn provides a dedicated, production-grade transformer: **`KBinsDiscretizer`**.

```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(
    n_bins=5,
    strategy='quantile',
    encode='ordinal'
)
```

### Parameter Breakdown:

#### 1. `n_bins` (int or array-like, default=5)
Specifies the number of bins to create.
- If an integer (e.g. `n_bins=5`), all selected features are split into 5 bins.
- Produces bin indices: $0, 1, 2, 3, 4$.

#### 2. `strategy` (`'uniform'`, `'quantile'`, `'kmeans'`, default=`'quantile'`)
Controls how the bin boundaries are calculated:
- **`'uniform'`:** Equal-width binning.
- **`'quantile'`:** Equal-frequency binning.
- **`'kmeans'`:** 1D K-Means cluster-based binning.

#### 3. `encode` (`'onehot'`, `'onehot-dense'`, `'ordinal'`, default=`'onehot'`)
Controls how the resulting discrete bins are numerically output:
- **`'ordinal'`:** Returns a single column containing integer bin indices ($0, 1, 2, \dots, k-1$).
- **`'onehot'`:** Returns a sparse binary matrix with $k$ columns.
- **`'onehot-dense'`:** Returns a dense NumPy array with $k$ columns.

> [!NOTE]
> **Mental Model:**  
> - **`strategy`** decides WHERE the boundaries are placed.  
> - **`encode`** decides HOW the resulting bins are represented.

---

## 10. Binarization

### Understand It First:
What if you don't need multiple groups, but simply a **binary YES or NO** answer?

For example:
- Is this passenger an adult? (`Age >= 18` $\longrightarrow$ `1`, `Age < 18` $\longrightarrow$ `0`)
- Did this customer make a purchase? (`Amount > 0` $\longrightarrow$ `1`, `Amount == 0` $\longrightarrow$ `0`)
- Is this a large family? (`FamilyMembers >= 3` $\longrightarrow$ `1`, `< 3` $\longrightarrow$ `0`)

This is called **Binarization**.

### Proper Definition:
> **Binarization** is the process of converting numerical values into binary values ($0$ and $1$) based on a specified threshold.

### Scikit-Learn `Binarizer`:
```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=18.0)
```

### Critical Rule on `Binarizer` Threshold Inequality:
Scikit-Learn's `Binarizer` implements this exact mathematical condition:
$$\text{Output} = \begin{cases} 1 & \text{if } x > \text{threshold} \\ 0 & \text{if } x \le \text{threshold} \end{cases}$$

> [!IMPORTANT]
> **BE PRECISE ABOUT THE INEQUALITY:**  
> Notice that values **equal to the threshold are mapped to 0**!  
> If you set `threshold=18.0`, an age of `18.0` becomes `0` (because $18.0$ is not strictly greater than $18.0$).  
> If you want an integer age of $18$ to be classified as an adult ($1$), set the threshold to **`17.5`** or **`17.0`**!

---

## 11. Binning vs. Binarization

| Dimension | Binning (Discretization) | Binarization |
| :--- | :--- | :--- |
| **Number of Groups** | Multiple groups ($k \ge 3$) | Exactly **two** groups ($0$ and $1$) |
| **Representation** | Ordinal ($0, 1, 2, \dots$) or One-Hot vectors | Binary flag ($0$ or $1$) |
| **Decision Rule** | Multiple interval boundaries ($b_0, b_1, \dots, b_k$) | Single threshold boundary |
| **Typical Tool** | `KBinsDiscretizer` or `pd.cut` | `Binarizer` |
| **Example** | Age $\longrightarrow$ Young, Adult, Senior | Age $\longrightarrow$ Minor ($0$) vs Adult ($1$) |

> **Memory Trick:**  
> - **Binning** $\longrightarrow$ **Many buckets**  
> - **Binarization** $\longrightarrow$ **Two buckets**

---

## 12. Information Loss from Binning

While binning provides simplification, it comes with an unavoidable cost: **Information Loss**.

Consider two customers:
- Customer 1: Age = $29$
- Customer 2: Age = $30$

In raw continuous data:
$$29 \ne 30$$
The algorithm knows Customer 2 is older than Customer 1.

Now suppose we apply binning with interval `[20, 40)`:
- Customer 1 (29) $\longrightarrow$ **Bin 1**
- Customer 2 (30) $\longrightarrow$ **Bin 1**

After binning:
$$\text{Customer 1} = \text{Customer 2} = \text{Bin 1}$$
The exact difference of 1 year has been erased. The model now treats both customers as completely identical in terms of age.

> [!WARNING]
> **The Binning Trade-Off:**  
> Binning reduces variance and simplifies non-linear boundaries, but it permanently discards fine-grained numerical details. If exact numerical distances matter for your prediction task, binning can hurt model accuracy!

---

## 13. Binning and Outliers

Consider a company salary dataset:
```text
₹20k,  ₹25k,  ₹30k,  ₹40k,  ₹50k,  ₹10,00,000
```
The ₹10,00,000 executive salary is an extreme outlier that will heavily distort linear regression slope lines and Euclidean distance metrics in KNN.

If we apply a 4-bin strategy where the highest bin is `[₹50k, max]`:
- ₹20k $\longrightarrow$ Bin 0
- ₹25k $\longrightarrow$ Bin 0
- ₹30k $\longrightarrow$ Bin 1
- ₹40k $\longrightarrow$ Bin 2
- ₹50k $\longrightarrow$ Bin 3
- ₹10,00,000 $\longrightarrow$ **Bin 3**

Notice what happened:
- The extreme value ₹10,00,000 received the exact same label as ₹50,000 (**Bin 3**).
- Its extreme magnitude no longer pulls the average or distance calculations off a cliff.

> **Remember:** Binning does not remove outliers; it **caps their representation** into a bounded category.

---

## 14. Binning vs. Mathematical Transformation

Do not confuse Binning with Mathematical Transformations (studied in topic 07):

```text
Raw Value: x = 100
             │
             ├──► Mathematical Transformation (Log):  x' = ln(100) ≈ 4.605   (Still a continuous number!)
             │
             └──► Discretization (Binning):           x' = Bin 3              (A discrete group / category!)
```

| Dimension | Mathematical Transformation | Discretization (Binning) |
| :--- | :--- | :--- |
| **Output Type** | Continuous numerical float ($\ln(x), \sqrt{x}$) | Discrete integer category or binary flag |
| **Information Preserved** | Order and fine-grained differences preserved | Fine-grained within-bin differences discarded |
| **Core Mechanism** | Smooth mathematical function $f(x)$ | Partitioning into discrete intervals $[a, b)$ |
| **Primary Goal** | Reduce skewness, linearize relationships | Group data, simplify, create piecewise bins |

---

## 15. Train / Test Workflow & Preventing Data Leakage

When using `KBinsDiscretizer` in a real machine learning project, you must follow the strict **Split-First Protocol**:

```text
WRONG (DATA LEAKAGE):
Fit KBinsDiscretizer on FULL dataset ──► Split Train / Test
(Bin edges calculated from test values leak into training!)

CORRECT:
1. Split data into Train and Test sets
2. Fit KBinsDiscretizer on Train set ONLY:  kbd.fit(X_train)
3. Transform Train set:                     X_train_binned = kbd.transform(X_train)
4. Transform Test set with TRAIN edges:    X_test_binned = kbd.transform(X_test)
```

> [!IMPORTANT]
> If you fit separate bin boundaries on the test set, the same bin index (e.g. `Bin 1`) would represent different numerical ranges in train and test, destroying model predictions!

---

## 16. Pipeline Integration

Connecting `KBinsDiscretizer` directly to an estimator inside a Scikit-Learn `Pipeline` guarantees that the split-first rule is enforced automatically:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('discretizer', KBinsDiscretizer(n_bins=5, strategy='quantile', encode='ordinal')),
    ('classifier', LogisticRegression(random_state=42))
])

# Fit on training data ONLY
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)
```

---

## 17. ColumnTransformer Integration

In real-world datasets with mixed column types, you rarely bin all columns. You use **`ColumnTransformer`** to bin selected numerical columns while scaling others and encoding categoricals:

```text
                           Raw Input Data
                                 │
                                 ▼
                       [ ColumnTransformer ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
      ['Age']                 ['Fare']                ['Sex']
         │                       │                       │
         ▼                       ▼                       ▼
  KBinsDiscretizer         StandardScaler          OneHotEncoder
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                     Clean Feature Matrix
```

### Python Implementation:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OneHotEncoder

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

## 18. Model Performance Experiment (Continuous Baseline vs. Binned Feature)

In Section 20 of the accompanying notebook ([`discretization_binarization.ipynb`](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/08_Discretization_and_Binarization/discretization_binarization.ipynb)), we evaluated whether binning passenger `Age` improves, matches, or degrades Logistic Regression prediction on the Titanic holdout test set (25% split):

| Model Workflow | Test Accuracy | Test F1 Score | Practical Takeaway |
| :--- | :---: | :---: | :--- |
| **Continuous Baseline (`StandardScaler`)** | **78.77%** | **72.86%** | Continuous age has a subtle linear gradient preserved by scaling |
| **Binned Age (`KBinsDiscretizer`, $k=5$)** | **77.09%** | **71.33%** | Discards within-bin differences, resulting in a ~1.7% drop |

### Key Practical Observation:
This real experiment proves that **binning is NOT an automatic magic bullet**. Because continuous age had a gentle linear relationship with survival, partitioning ages into 5 broad buckets caused **information loss** (e.g. an 18-year-old and a 25-year-old were treated as identical), which slightly reduced accuracy.  
**Always benchmark your binned model against a continuous baseline using holdout cross-validation!**

---

## 19. When to Use & When NOT to Use Binning

### When to Use:
1. When non-linear, threshold-based relationships exist (e.g., credit tiers, tax brackets).
2. When extreme tail values distort linear or distance-based models.
3. When business stakeholders require interpretable, policy-based categories.
4. When continuous sensor measurements have minor noise that shouldn't affect downstream decisions.

### When NOT to Use:
1. When exact numerical values carry critical precision (e.g., financial pricing, medical dosage).
2. When using tree-based models (Decision Trees, Random Forests, XGBoost) because trees **already perform their own optimal threshold splits** on raw continuous features!
3. When arbitrary bin boundaries split highly related points into different buckets (e.g., $17.9$ vs $18.0$).
4. When cross-validation shows that binning reduces test set accuracy or $R^2$.

---

## 20. Common Mistakes to Avoid

1. **Confusing Equal Width with Equal Frequency:** Thinking equal width means equal number of samples per bin.
2. **Assuming Equal Frequency Means Equal Spans:** Overlooking that dense areas produce narrow bins and sparse tails produce huge bins.
3. **Using K-Means Binning Blindly:** Applying K-Means when the data has no natural clusters.
4. **Creating Arbitrary Custom Bins:** Inventing boundary numbers without domain justification.
5. **Using Too Many Bins:** Setting `n_bins=50` on a small dataset, creating sparse bins and overfitting.
6. **Ignoring Information Loss:** Forgetting that all points inside a bin lose their relative differences.
7. **Believing Binning Deletes Outliers:** Failing to realize that outliers are simply grouped into the outermost bin.
8. **Confusing Binning with Mathematical Transformations:** Forgetting that log changes continuous form, while binning creates discrete groups.
9. **Confusing Binning with Binarization:** Forgetting that binarization creates only 2 buckets ($0/1$).
10. **Fitting Bin Boundaries on Test Data:** Causing data leakage instead of using the training boundaries.
11. **Assuming Binning Always Improves Models:** Not evaluating the impact against a raw scaled baseline.
12. **Misinterpreting `Binarizer(threshold=t)`:** Forgetting that $x = t \implies 0$ (strict greater-than inequality).
13. **Confusing `strategy` with `encode`:** `strategy` places edges; `encode` formats outputs.
14. **Binning Features for Tree Models:** Spending time binning features before training Random Forest or XGBoost, which already find optimal split thresholds naturally.

---

## 21. Quick Reference Table

| Technique | Core Idea | scikit-learn Class | Key Parameter |
| :--- | :--- | :--- | :--- |
| **Equal Width** | Bins have the exact same numerical width | `KBinsDiscretizer` | `strategy='uniform'` |
| **Equal Frequency** | Bins have approximately equal data counts | `KBinsDiscretizer` | `strategy='quantile'` |
| **K-Means Binning** | 1D clusters become discrete bins | `KBinsDiscretizer` | `strategy='kmeans'` |
| **Custom Binning** | Human/business defined boundaries | `pd.cut()` | `bins=[b0, b1, ...]` |
| **Binarization** | Splits values into 2 classes ($0$ and $1$) | `Binarizer` | `threshold=val` ($x > val \implies 1$) |

### `KBinsDiscretizer` Strategies:
- **`uniform`:** Equal Width
- **`quantile`:** Equal Frequency
- **`kmeans`:** 1D K-Means Cluster Centroids

### `KBinsDiscretizer` Encodings:
- **`ordinal`:** Integer indices ($0, 1, 2, \dots$)
- **`onehot`:** Sparse binary matrix
- **`onehot-dense`:** Dense 2D NumPy array

---

## 22. Final Mental Model

```text
                           Numerical Feature
                                   │
                                   ▼
                        Need Discrete Groups?
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
                YES                                  NO
                 │                                   │
                 ▼                                   ▼
        Need only YES / NO?                  Keep Continuous
                 │                           (Scale or Transform)
        ┌────────┴────────┐
        ▼                 ▼
       YES                NO
        │                 │
        ▼                 ▼
   Binarization        Binning (KBinsDiscretizer)
   (Binarizer)            │
   Threshold ──► 0 / 1    ├───────────────┬───────────────┐
                          ▼               ▼               ▼
                       Uniform        Quantile         K-Means
                      (Eq. Width)   (Eq. Frequency)  (Clustered)
```

> **Core Summary:**  
> - **Discretization (Binning)** converts continuous numbers into **multiple discrete bins**.  
> - **Binarization** converts continuous numbers into **two binary groups ($0$ and $1$)** using a threshold.
