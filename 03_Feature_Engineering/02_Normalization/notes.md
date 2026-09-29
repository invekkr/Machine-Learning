# Module 03: Feature Engineering — Normalization (`MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`)

Feature scaling is not a one-size-fits-all problem. While **Standardization** shifts data to have a mean of 0 and a standard deviation of 1, **Normalization** refers to a family of scaling techniques that adjust the bounds, centers, or spreads of numerical features using ranges, absolute maximums, or quartiles.

---

## 1. What is Normalization?

### Understand It First: The Core Problem

Imagine you are looking at customer profiles:
- **Age:** ranges from `20` to `60` years.
- **Salary:** ranges from `\$30,000` to `\$120,000` per year.
- **Credit Score:** ranges from `300` to `850`.

Each feature operates on an entirely different scale. 
An algorithm computing distances or gradient steps will be heavily dominated by Salary (in the tens of thousands) while completely ignoring Age (in the double digits).

Normalization transforms these varied numbers onto a **common, comparable scale** (often between $0$ and $1$, or $-1$ and $+1$) while strictly preserving the relative ranking and proportions between data points.

### Proper Definition

> **Normalization** is a feature-scaling technique used to transform numerical features into a standardized bounded or relative scale while preserving the relative mathematical relationships among observations.

*Note:* In machine learning, "Normalization" is an umbrella term that encompasses several specific techniques, primarily **Min-Max Scaling**, **Mean Normalization**, **Max Absolute Scaling**, and **Robust Scaling**.

---

## 2. Min-Max Scaling

Min-Max Scaling is the most widely used normalization method.

### Understand It First
The logic is straightforward:
- Take the smallest number in the column and make it **0**.
- Take the largest number in the column and make it **1**.
- Map every other number linearly to a decimal **between 0 and 1**.

### Small Numerical Example
Suppose we have five values:
$$10,\; 20,\; 30,\; 40,\; 50$$

- **Minimum ($x_{min}$):** $10$
- **Maximum ($x_{max}$):** $50$
- **Range ($x_{max} - x_{min}$):** $50 - 10 = 40$

Let's manually scale each number:
- For $10$: $\frac{10 - 10}{40} = \frac{0}{40} = \mathbf{0.00}$
- For $20$: $\frac{20 - 10}{40} = \frac{10}{40} = \mathbf{0.25}$
- For $30$: $\frac{30 - 10}{40} = \frac{20}{40} = \mathbf{0.50}$ (halfway between min and max)
- For $40$: $\frac{40 - 10}{40} = \frac{30}{40} = \mathbf{0.75}$
- For $50$: $\frac{50 - 10}{40} = \frac{40}{40} = \mathbf{1.00}$

### The Formula
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

| Component | Meaning |
| :---: | :--- |
| **$x$** | The original raw feature value. |
| **$x_{min}$** | The minimum value observed in that feature across the dataset. |
| **$x_{max}$** | The maximum value observed in that feature across the dataset. |
| **$x_{max} - x_{min}$** | The total feature range (span). |
| **$x_{scaled}$** | The normalized value, strictly bounded in $[0, 1]$. |

### Why Use Min-Max Scaling?
1. **Guaranteed Bounded Range $[0, 1]$:** Essential for algorithms that require inputs within a fixed range (e.g., neural networks with Sigmoid activations, image pixel arrays where $0 \dots 255 \rightarrow 0 \dots 1$).
2. **Preserves Exact Zeroes:** If the minimum value is $0$ (e.g. counts), 0 stays 0.
3. **No Assumptions About Normality:** Works well even when the data does not follow a bell curve.

### The Major Limitation: Extreme Sensitivity to Outliers
What happens if we add an outlier of $1,000$ to our sample?
$$10,\; 20,\; 30,\; 40,\; 1000$$

- $x_{min} = 10$, $x_{max} = 1000$. Range = $990$.
- For $20$: $\frac{20 - 10}{990} = \frac{10}{990} \approx \mathbf{0.010}$
- For $30$: $\frac{30 - 10}{990} = \frac{20}{990} \approx \mathbf{0.020}$
- For $40$: $\frac{40 - 10}{990} = \frac{30}{990} \approx \mathbf{0.030}$
- For $1000$: $\frac{1000 - 10}{990} = \frac{990}{990} = \mathbf{1.000}$

> **The Outlier Trap:**
> Because $1,000$ stretched the denominator ($x_{max} - x_{min}$) to $990$, **all normal observations are squashed into a tiny strip between 0.01 and 0.03**. The model loses the ability to distinguish between $20$ and $40$!

---

## 3. Mean Normalization

### Understand It First
Min-Max scaling centers the data at $0.5$ (if symmetric). What if we want the data **centered at 0**, but scaled using the **range** rather than standard deviation?
That is **Mean Normalization**.

### Small Numerical Example
Using the same values:
$$10,\; 20,\; 30,\; 40,\; 50$$

- **Mean ($\mu$):** $\frac{10 + 20 + 30 + 40 + 50}{5} = 30$
- **Minimum ($x_{min}$):** $10$
- **Maximum ($x_{max}$):** $50$
- **Range:** $50 - 10 = 40$

Let's calculate $x'$ for each number:
- For $10$: $\frac{10 - 30}{40} = \frac{-20}{40} = \mathbf{-0.50}$
- For $30$: $\frac{30 - 30}{40} = \frac{0}{40} = \mathbf{0.00}$ (mean becomes 0)
- For $40$: $\frac{40 - 30}{40} = \frac{10}{40} = \mathbf{+0.25}$
- For $50$: $\frac{50 - 30}{40} = \frac{+20}{40} = \mathbf{+0.50}$

The resulting values are centered at $0$ and lie strictly within $[-0.5, +0.5]$ (or $[-1, +1]$ depending on skew).

### The Formula
$$x' = \frac{x - \mu}{x_{max} - x_{min}}$$

### Crucial Distinction: Mean Normalization vs. Standardization
Do not confuse these two transformations:

```text
Mean Normalization:   x' = (x - μ) / (x_max - x_min)   <-- Divides by the RANGE
Standardization:      z  = (x - μ) / σ                 <-- Divides by the STANDARD DEVIATION
```

*Note:* Scikit-Learn does not have a dedicated `MeanNormalizer` class because practitioners either use `StandardScaler` (for zero-mean variance-scaling) or `MinMaxScaler` (for range-bounding). However, it remains an important theoretical benchmark.

---

## 4. Max Absolute Scaling (`MaxAbsScaler`)

### Understand It First
Instead of finding both minimum and maximum, what if we simply divide every number by the **largest absolute magnitude** observed in that feature?

### Small Numerical Example
Suppose a feature contains both positive and negative values:
$$-100,\; -50,\; 0,\; 50,\; 100$$

- **Maximum absolute value ($\max(|x|)$):** $100$

Let's divide each value by $100$:
- $-100 \rightarrow \frac{-100}{100} = \mathbf{-1.0}$
- $-50 \rightarrow \frac{-50}{100} = \mathbf{-0.5}$
- $0 \rightarrow \frac{0}{100} = \mathbf{0.0}$
- $50 \rightarrow \frac{50}{100} = \mathbf{+0.5}$
- $100 \rightarrow \frac{100}{100} = \mathbf{+1.0}$

The output is strictly bounded within $[-1.0, +1.0]$.

### The Formula
$$x' = \frac{x}{\max(|x|)}$$

---

## 5. The Critical Superpower of MaxAbsScaler: Sparse Data

What is **Sparse Data**?
A dataset is called *sparse* when the vast majority of its cells are zeros.
Examples include:
- **Text features:** Bag-of-Words and TF-IDF matrices (thousands of vocabulary columns, where any single document only uses a few dozen words).
- **One-Hot Encoded features:** High-cardinality categorical variables with dozens of columns filled with 0s and a single 1.
- **Recommendation systems:** User-item interaction matrices where most users haven't rated most items.

### Why Standard Scalers Fail on Sparse Data:
- `StandardScaler` (with mean centering) subtracts $\mu$ from every cell: $0 - \mu = -\mu$.
- `MinMaxScaler` subtracts $x_{min}$ from every cell: $0 - x_{min} = -x_{min}$.
- If your sparse matrix contains 10 million zeros, subtracting a number turns all 10 million zeros into non-zero decimals. This **destroys sparsity**, converting a compact memory structure into a massive dense matrix that crashes your computer's RAM!

### Why MaxAbsScaler Is Perfect for Sparse Data:
- `MaxAbsScaler` only divides by a positive constant:
$$\frac{0}{\max(|x|)} = 0$$
- **Zeros remain exactly zero!**
- It rescales all non-zero entries into $[-1, 1]$ without shifting the origin, preserving the sparse matrix format and keeping memory footprint tiny.

---

## 6. Robust Scaling (`RobustScaler`)

### Understand It First: Why Outliers Break Other Scalers
Consider two summary statistics for central tendency:
- **Mean:** Extremely vulnerable to extreme values.
- **Median:** Highly resistant (robust) to extreme values.

Consider two summary statistics for spread:
- **Range / Standard Deviation:** Easily inflated by a single massive number.
- **Interquartile Range (IQR):** Measures only the middle 50% of the data, completely ignoring the outer tails!

### What is the Interquartile Range (IQR)?
1. **$Q_1$ (25th percentile):** 25% of data lies below this value.
2. **Median / $Q_2$ (50th percentile):** The exact middle point.
3. **$Q_3$ (75th percentile):** 75% of data lies below this value.
4. **$IQR$:** The spread of the central 50%:
$$IQR = Q_3 - Q_1$$

### Small Numerical Example
Suppose we have a dataset with an extreme outlier:
$$10,\; 12,\; 13,\; 14,\; 15,\; 1000$$

- **Median:** $\frac{13 + 14}{2} = 13.5$
- **$Q_1$:** $12.0$
- **$Q_3$:** $15.0$
- **$IQR$:** $15.0 - 12.0 = 3.0$

Notice how the extreme value ($1,000$) did not inflate the Median ($13.5$) or the IQR ($3.0$)!

Now apply the **Robust Scaling Formula**:
$$x' = \frac{x - \text{Median}}{IQR}$$

- For $10$: $\frac{10 - 13.5}{3.0} = \frac{-3.5}{3.0} \approx \mathbf{-1.17}$
- For $13$: $\frac{13 - 13.5}{3.0} = \frac{-0.5}{3.0} \approx \mathbf{-0.17}$
- For $14$: $\frac{14 - 13.5}{3.0} = \frac{+0.5}{3.0} \approx \mathbf{+0.17}$
- For $15$: $\frac{15 - 13.5}{3.0} = \frac{+1.5}{3.0} = \mathbf{+0.50}$
- For $1000$: $\frac{1000 - 13.5}{3.0} = \frac{986.5}{3.0} \approx \mathbf{+328.8}$

### The Takeaway:
The normal points ($10$ through $15$) are spread nicely across a clean range from $-1.17$ to $+0.50$.
Unlike Min-Max scaling, the normal points were **NOT squashed into an invisible micro-band**! The outlier gets a large value, but the rest of the dataset retains its meaningful internal variance.

---

## 7. Comprehensive Comparison: All Scalers at a Glance

| Scaler | Formula | Center | Scale Factor | Output Range | Outlier Sensitivity | Best Used When |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Min-Max** (`MinMaxScaler`) | $\frac{x - x_{min}}{x_{max} - x_{min}}$ | Minimum becomes 0 | Range ($x_{max} - x_{min}$) | Strictly $[0, 1]$ | **Extremely High** | Bounded output needed (images, neural nets, algorithms assuming $[0, 1]$). |
| **Mean Normalization** | $\frac{x - \mu}{x_{max} - x_{min}}$ | Mean becomes 0 | Range ($x_{max} - x_{min}$) | Roughly $[-0.5, +0.5]$ | **High** | Zero-centered bounded data needed. |
| **Max Absolute** (`MaxAbsScaler`) | $\frac{x}{\max(\|x\|)}$ | Not centered (0 stays 0) | $\max(\|x\|)$ | Strictly $[-1, 1]$ | **High** | **Sparse data** (TF-IDF, Bag-of-Words, one-hot vectors) to preserve zero structure. |
| **Robust Scaler** (`RobustScaler`) | $\frac{x - \text{Median}}{IQR}$ | Median becomes 0 | $IQR = Q_3 - Q_1$ | Unbounded | **Very Low (Robust)** | Data contains **significant outliers** that cannot be removed. |
| **Standardization** (`StandardScaler`) | $\frac{x - \mu}{\sigma}$ | Mean becomes 0 | Std Dev ($\sigma$) | Unbounded (mean 0, std 1) | **Moderate** | General-purpose default for distance/gradient models (KNN, SVM, Logistic Reg). |

---

## 8. Normalization vs. Standardization

| Dimension | Normalization (Min-Max) | Standardization (Z-Score) |
| :--- | :--- | :--- |
| **Bounding** | Strictly bounded in $[0, 1]$ (or $[-1, 1]$) | Unbounded (typically $[-3, +3]$, but can exceed) |
| **Center** | Minimum maps to 0 | Mean maps to 0 |
| **Scale** | Range ($x_{max} - x_{min}$) | Standard Deviation ($\sigma$) |
| **Outliers** | Compresses normal data into a tiny band | Outliers remain out in tails, but don't crush the standard deviation as severely as a range |
| **Primary Algorithms** | Neural Networks, Image Processing, K-Means (when positive bounds needed) | Logistic Regression, SVM, Linear Regression, PCA, Neural Networks |

---

## 9. When to Use Which Scaler (Practical Rulebook)

1. **Default starting point:** If in doubt and data has no severe outliers, **Standardization (`StandardScaler`)** is the standard workhorse for linear models and distance algorithms.
2. **If features have strict physical bounds or you are feeding neural networks / image models:** Use **`MinMaxScaler`**.
3. **If your dataset is sparse (mostly zeros, e.g. text TF-IDF or wide dummy matrices):** Use **`MaxAbsScaler`** to avoid breaking memory limits.
4. **If your dataset contains noticeable outliers that you cannot drop:** Use **`RobustScaler`**.
5. **If using tree-based algorithms (Random Forest, XGBoost, LightGBM):** **Do not scale at all!** Trees split on individual feature ranks, so scaling has zero effect on model performance.

---

## 10. Train/Test Data Leakage Prevention

Always remember the cardinal rule of ML preprocessing:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Split raw data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Fit ONLY on X_train, then transform X_train
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 3. Transform X_test using the parameters learned from X_train
X_test_scaled = scaler.transform(X_test)
```

**Never call `fit()` or `fit_transform()` on `X_test`.**
If you do, the scaler uses the minimum and maximum of the test set, leaking knowledge about the test distribution into the evaluation pipeline.

---

## Summary Checklist
- [x] Min-Max scaling maps minimum to 0 and maximum to 1 via $\frac{x - x_{min}}{x_{max} - x_{min}}$.
- [x] Outliers severely squash normal data in Min-Max scaling because the range expands.
- [x] Mean Normalization centers data at 0 and scales by range: $\frac{x - \mu}{x_{max} - x_{min}}$.
- [x] Max Absolute Scaling divides by $\max(|x|)$, producing $[-1, 1]$ and preserving sparse zeros.
- [x] Robust Scaling uses Median and IQR ($\frac{x - \text{Median}}{IQR}$) to resist extreme outliers.
- [x] Always split before scaling, fitting scalers exclusively on `X_train`.
