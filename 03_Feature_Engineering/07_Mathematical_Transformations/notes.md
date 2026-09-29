# Module 03: Feature Engineering — Mathematical Transformations

When working with numerical features in real-world machine learning, the raw numbers are often distributed in ways that make it difficult for algorithms to learn clean patterns. 

For instance, financial data (like income, house prices, or transaction amounts) typically contains a huge cluster of everyday values and a few massive outliers that stretch out across a long tail.

In this module, we study **Mathematical Transformations**—functions like $\log(x)$, $\sqrt{x}$, $1/x$, and $x^2$—that reshape numerical distributions, stabilize variance, and help machine learning models discover stronger mathematical relationships.

---

## 1. What are Mathematical Transformations?

### Understand It First: The Problem of Skewed Data

Consider a dataset containing the annual salaries of employees at a tech company:

```text
Salaries (in thousands):
20,  25,  28,  30,  35,  40,  45,  50,  250,  800
```

Notice what is happening:
- Eight out of ten employees earn between \$20,000 and \$50,000.
- Two executives earn \$250,000 and \$800,000.

If you plot this data on a chart:
```text
  Number of People
       ▲
       │  ████
       │  ██████
       │  ████████
       │  ██████████                                        █         █
       └───────────────────────────────────────────────────────────────────► Salary
         $20k - $50k (Huge cluster!)                       $250k     $800k
                                                      (Long tail to the right!)
```

Most data points are crowded into a small space on the left, while the graph stretches far out to the right. This is called a **Right-Skewed Distribution**.

### The Solution: Applying a Mathematical Function
Instead of feeding raw, stretched numbers into our model, we can pass each number through a mathematical function $f(x)$:

$$x \longrightarrow f(x)$$

Common transformations include:
- **Logarithmic:** $x \longrightarrow \log(x)$ or $\log(1 + x)$
- **Square Root:** $x \longrightarrow \sqrt{x}$
- **Reciprocal:** $x \longrightarrow \frac{1}{x}$
- **Square:** $x \longrightarrow x^2$

### What Transformations Aim to Do:
1. **Compress extreme values** and reduce skewness.
2. **Stabilize variance** so errors don't explode for large numbers.
3. **Linearize relationships** so simple linear models can capture them easily.

> **CRITICAL RULE:**  
> A mathematical transformation does **NOT** automatically mean making the data normally distributed.  
> Do not teach or assume: *"Transformations make data normal."*  
> Instead, transformations change the mathematical representation and reduce skewness in ways that can make learning easier for specific models.

---

## 2. Understanding Skewness from Scratch

### What is a Distribution?
A **distribution** simply shows all the possible values of a feature and how frequently those values appear.

### Symmetric vs. Skewed Distributions

```text
1. Symmetric (Bell Curve / Normal)       2. Right-Skewed (Positive Skew)        3. Left-Skewed (Negative Skew)
               ▲                                        ▲                                      ▲
             /   \                                    /   \                                  /   \
            /     \                                  /     \                                /     \
          /         \                              /         \____                        /         \
       ──┴───────────┴──                        ──┴───────────────┴────                ────┴───────────────┴──
       Mean ≈ Median ≈ Mode                     Mean > Median                          Mean < Median
       Balanced on both sides                   Long tail on the RIGHT                 Long tail on the LEFT
```

### 1. Right-Skewed (Positive Skewness):
- **Visual:** The peak is on the left, and the **tail stretches far to the right**.
- **Math:** A few huge numbers pull the average up: $\text{Mean} > \text{Median}$.
- **Real-World Examples:**
  - Individual income and wealth
  - House prices
  - Healthcare costs per patient
  - City population sizes
  - Website page views per article

### 2. Left-Skewed (Negative Skewness):
- **Visual:** The peak is on the right, and the **tail stretches far to the left**.
- **Math:** A few tiny numbers pull the average down: $\text{Mean} < \text{Median}$.
- **Real-World Examples:**
  - Human lifespan / age of death (most people live past 60; few die in early childhood)
  - Scores on an easy high-school exam (most students score 80–95%; a few fail with 10–20%)
  - Retirement ages

> **Formal Definition:**  
> **Skewness** is a statistical measure of the asymmetry of a feature's probability distribution around its mean. A symmetric distribution has a skewness close to $0$.

---

## 3. Why Apply Mathematical Transformations?

Not every machine learning problem requires transformations. However, they are valuable in specific scenarios:

### 1. Non-Linear Relationships
Suppose an employee's salary grows exponentially with years of experience:
$$\text{Salary} \approx e^{\text{Experience}}$$
If you fit a standard Linear Regression model directly, a straight line will fail to capture the exponential bend.  
However, if you transform salary using a logarithm:
$$\log(\text{Salary}) \approx \text{Experience}$$
The relationship suddenly becomes a **clean straight line** that Linear Regression can fit with high precision!

### 2. Changing Variance (Heteroscedasticity)
In financial data, high earners often exhibit far greater variance in spending than low earners. Logarithmic transformations stabilize this spread, making prediction errors more consistent across all income brackets.

---

## 4. Log Transformation ($\log(x)$)

Log transformation is by far the most famous and widely used mathematical transformation in data science.

### Intuition: Compressing the Extremes
Remember the definition of a logarithm:
$$\log_{10}(10) = 1, \quad \log_{10}(100) = 2, \quad \log_{10}(1,000) = 3$$

Notice how the numbers on the left grow by factors of ten ($10 \longrightarrow 100 \longrightarrow 1,000$), but the log values grow by single steps ($1 \longrightarrow 2 \longrightarrow 3$)!

> **The Log Superpower:**  
> A logarithm compresses huge numbers dramatically while barely touching smaller numbers. It pulls the distant right tail inward toward the center.

### Formula:
$$x' = \ln(x) \quad \text{or} \quad x' = \log_{10}(x)$$

- $x$: Original positive raw feature value.
- $x'$: Transformed value.
- $\ln$: Natural logarithm (base $e \approx 2.718$).

### Tiny Manual Example:
Values: $[10, 100, 1000]$
- $x = 10 \implies \log_{10}(10) = 1.0$
- $x = 100 \implies \log_{10}(100) = 2.0$
- $x = 1000 \implies \log_{10}(1000) = 3.0$

The original ratio of maximum to minimum was $1,000 / 10 = \mathbf{100}$.  
After transformation, the ratio is $3.0 / 1.0 = \mathbf{3.0}$. The extreme difference has been safely compressed.

### The Major Limitation of Plain $\log(x)$:
1. $\log(0)$ is **mathematically undefined** ($-\infty$).
2. $\log(\text{negative number})$ is **impossible** for real numbers.
3. Therefore, standard $\log(x)$ **can only be applied to strictly positive numbers ($x > 0$)**.

---

## 5. The Solution for Zeroes: $\log_{1p}(x)$

What if your dataset contains zero values (e.g., number of purchases, salary bonuses, or count features where $0$ is common)?

If you run `np.log(0)`, Python outputs `-inf` and crashes your model.

### The $\log_{1p}$ Formula:
$$\log_{1p}(x) = \log(1 + x)$$

### Why Adding 1 Fixes Everything:
When $x = 0$:
$$\log_{1p}(0) = \log(1 + 0) = \log(1) = \mathbf{0.0}$$

- A count of $0$ stays cleanly at $0.0$!
- Positive numbers are smoothly compressed just like standard logarithms.

```python
import numpy as np

# Plain log crashes or produces -inf on zero:
# np.log(0) -> -inf

# log1p safely handles zero:
print(np.log1p(0))  # 0.0
print(np.log1p(9))  # ln(10) ≈ 2.302
```

---

## 6. Square Root Transformation ($\sqrt{x}$)

### Intuition:
A square root also compresses large numbers, but it is **much milder (less aggressive) than a logarithm**.

### Formula:
$$x' = \sqrt{x}$$

### Tiny Manual Example:
Values: $[4, 9, 16, 100]$
- $\sqrt{4} = 2$
- $\sqrt{9} = 3$
- $\sqrt{16} = 4$
- $\sqrt{100} = 10$

### Square Root vs. Logarithm:
- **Zero-Friendly:** Unlike plain $\log(x)$, $\sqrt{0} = 0$.
- **Mild Compression:** Useful for moderately right-skewed count data (e.g., number of website clicks, household size) where a full log transformation might compress the values too aggressively.

---

## 7. Reciprocal Transformation ($\frac{1}{x}$)

### Intuition:
A reciprocal inverts every number by dividing 1 by that number.

### Formula:
$$x' = \frac{1}{x}$$

### Tiny Manual Example:
Values: $[2, 4, 10]$
- $x = 2 \implies \frac{1}{2} = \mathbf{0.50}$
- $x = 4 \implies \frac{1}{4} = \mathbf{0.25}$
- $x = 10 \implies \frac{1}{10} = \mathbf{0.10}$

### Two Critical Properties of Reciprocal:
1. **Reverses Ordering for Positive Numbers:**
   - In original data: $2 < 4 < 10$.
   - After reciprocal: $0.50 > 0.25 > 0.10$. Small numbers become large; large numbers become tiny decimals!
2. **Zero is Undefined:**
   - $\frac{1}{0}$ is mathematically undefined (division by zero). It cannot be used if zeroes exist.

---

## 8. Square Transformation ($x^2$)

### Intuition:
The square transformation multiplies a number by itself. It does the opposite of a square root: **it expands large values much faster than small values**.

### Formula:
$$x' = x^2$$

### Tiny Manual Example:
Values: $[2, 3, 4, 5]$
- $2^2 = 4$
- $3^2 = 9$
- $4^2 = 16$
- $5^2 = 25$

### When Might It Help?
Square transformations can **sometimes help with certain left-skewed distributions** where we want to stretch out the dense cluster of high values.

> **CRITICAL PHRASING:**  
> Do **NOT** say: *"Square transformation fixes left skewness."*  
> Say: *"It can sometimes help reshape certain left-skewed distributions."*

---

## 9. Comprehensive Comparison Table

| Transformation | Formula | Effect on Data | Best Used For | Important Limitation |
| :--- | :---: | :--- | :--- | :--- |
| **Log ($\ln$)** | $\ln(x)$ | Strong compression of high values | Strongly right-skewed data | Strictly requires $x > 0$ |
| **Log1p** | $\ln(1 + x)$ | Strong compression of high values | Right-skewed data containing $0$ | Requires $x \ge 0$ |
| **Square Root** | $\sqrt{x}$ | Moderate compression of high values | Moderately right-skewed counts | Requires $x \ge 0$ |
| **Reciprocal** | $\frac{1}{x}$ | Extreme inversion (large $\to$ tiny) | Severe right-skewed rates | $x = 0$ is undefined; reverses order |
| **Square** | $x^2$ | Stretches high values apart | Certain left-skewed data | Amplifies extreme high outliers |

---

## 10. How to Detect and Quantify Skewness

Before applying any transformation, you must visually inspect and mathematically measure the feature:

### 1. The Skewness Coefficient
Using `scipy.stats.skew` or `df['col'].skew()`:

| Skewness Value | Interpretation | Action |
| :--- | :--- | :--- |
| **$-0.5$ to $+0.5$** | **Fairly Symmetric** | Usually no transformation needed |
| **$+0.5$ to $+1.0$** | **Moderately Right-Skewed** | Consider Square Root or Log1p |
| **$> +1.0$** | **Highly Right-Skewed** | Strong candidate for Log / Log1p |
| **$< -0.5$** | **Left-Skewed** | Consider Square or custom reflection |

### 2. Visual Tools:
- **Histogram & KDE:** Shows the shape of the frequency distribution.
- **Box Plot:** Shows how far the median is shifted toward one edge, and displays outlier points beyond the whiskers.

---

## 11. Quantile-Quantile (QQ) Plots

A **QQ Plot (Quantile-Quantile Plot)** is the standard diagnostic graph used in data science to check how closely an empirical feature matches a theoretical normal distribution.

### What is a Quantile?
A quantile is simply a division point. For example:
- The median is the 50% quantile (half the data is below it).
- The 25th percentile ($Q_1$) is the 0.25 quantile.

### How a QQ Plot Works:
1. It sorts your actual data values into quantiles (Y-axis).
2. It calculates where those quantiles *would* sit if the data came from a perfect theoretical normal distribution (X-axis).
3. It plots the points against a red $45^\circ$ reference line:

```text
Normal Distribution (Good Fit!)           Right-Skewed Distribution (Poor Fit!)
         Actual Quantiles                          Actual Quantiles
             ▲                                         ▲
             │       /                                 │          /
             │     /•                                  │       •/  ◄── Upward curve!
             │   /•                                    │     •/
             │ /•                                      │  ••/
             └──────────► Theoretical                  └──────────► Theoretical
     Points hug the straight line!             Points curve strongly off the line!
```

- **If points hug the straight diagonal line:** The feature behaves closely to a normal distribution.
- **If points bend into a curve away from the line:** The feature diverges from normality.

```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate a QQ plot
stats.probplot(data, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()
```

---

## 12. `FunctionTransformer` in Scikit-Learn

In Pandas, you can manually write:
```python
df["Income"] = np.log1p(df["Income"])
```

However, manual assignments break in production pipelines because they cannot be saved with models or integrated into cross-validation.

Scikit-Learn solves this with **`FunctionTransformer`**:

> **`FunctionTransformer`** wraps any standard Python or NumPy mathematical function (like `np.log1p` or `np.sqrt`) into a formal Scikit-Learn Transformer with `fit()` and `transform()` methods.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Create the transformer object
log_transformer = FunctionTransformer(np.log1p)

# Transform data
X_transformed = log_transformer.fit_transform(X)
```

---

## 13. Integrating with `ColumnTransformer` and `Pipeline`

In real workflows, you only want to transform the skewed column (e.g. `Income`), while leaving clean columns (e.g. `Age`) untouched and encoding categorical columns (e.g. `City`):

```text
                         Raw Input Data
                               │
                               ▼
                     [ ColumnTransformer ]
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
     ['Income']             ['Age']               ['City']
         │                     │                     │
         ▼                     ▼                     ▼
FunctionTransformer(np.log1p) Passthrough      OneHotEncoder
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
               Ready for Machine Learning Model!
```

### Python Code:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
import numpy as np

preprocessor = ColumnTransformer(
    transformers=[
        (
            'log_income',
            FunctionTransformer(np.log1p, feature_names_out='one-to-one'),
            ['Income'],
        ),  # Mathematical Transformation
        ('ohe_city', OneHotEncoder(), ['City']),  # Categorical Encoding
    ],
    remainder='passthrough',  # Keeps 'Age' untouched
)
```

> **Tip (`feature_names_out='one-to-one'`):**  
> Supplying `feature_names_out='one-to-one'` instructs Scikit-Learn that the transformer preserves input column names, allowing `preprocessor.get_feature_names_out()` to work seamlessly without raising an `AttributeError`.

---

## 14. Train / Test Workflow & Data Leakage

For fixed mathematical functions like `np.log1p` or `np.sqrt`:
- The mathematical rule $x \to \ln(1+x)$ is fixed and does **not** learn parameters from data.
- However, keeping it inside your `Pipeline` is still essential to guarantee that test data and live production inputs are transformed identically without manual wrangling.

---

## 15. Do All Machine Learning Models Need Transformations?

### Tree-Based Models: **NO**
Decision Trees, Random Forests, XGBoost, and LightGBM make decisions using threshold splits:
$$\text{Is Income} \le 50,000?$$

Because logarithms and square roots are **strictly monotonic** (they never change the relative order of numbers: if $A < B$, then $\log(A) < \log(B)$), tree splits are completely unaffected.  
**Tree models generally do not require transformations solely to make features normal.**

### Linear & Distance-Based Models: **YES, OFTEN**
- **Linear Regression & Logistic Regression:** Benefit when transformations make curved relationships linear.
- **K-Means & KNN:** Benefit when transformations prevent extreme tail values from distorting distance calculations.

> **IMPORTANT CORRECTION:**  
> Do **NOT** teach: *"Linear Regression requires every feature to be normally distributed."*  
> Linear Regression does not require inputs to be normal. It assumes that the **relationship is linear** and that the **residuals (prediction errors) are normally distributed**. Reshaping skewed features often helps satisfy these linear assumptions.

---

## 16. Feature Scaling vs. Mathematical Transformation

Do not confuse Feature Scaling with Mathematical Transformations:

| Dimension | Feature Scaling (StandardScaler, MinMaxScaler) | Mathematical Transformation (Log, Sqrt) |
| :--- | :--- | :--- |
| **Primary Goal** | Changes the **scale / range** of features | Changes the **shape / distribution** of features |
| **Effect on Skewness** | **Zero effect.** Skewness is unchanged! | **Reduces or alters skewness.** |
| **Effect on Shape** | Centers at 0 or bounds between 0 and 1 | Compresses or stretches the tail |
| **Formulas** | $z = \frac{x - \mu}{\sigma}$ | $x' = \ln(1 + x)$ |

---

## 17. Transformation Does NOT Automatically Improve Accuracy

Never assume that applying a log transformation will guarantee higher accuracy.

Suppose you run a real test:
- Model without transformation: $\text{MAE} = \$12,400$
- Model with log transformation: $\text{MAE} = \$13,800$

If error increases, the transformation did not benefit that specific dataset and model combination!  
**Always evaluate model metrics with cross-validation before keeping a transformation.**

---

## 18. Quick Reference Table

| Concept | Formula | Main Purpose | Important Limitation |
| :--- | :---: | :--- | :--- |
| **Log** | $\ln(x)$ | Compresses extreme right-skewed values | Strictly requires $x > 0$ |
| **Log1p** | $\ln(1 + x)$ | Compresses right-skewed values with zeroes | Requires $x \ge 0$ |
| **Square Root** | $\sqrt{x}$ | Mild compression for moderate skewness | Requires $x \ge 0$ |
| **Reciprocal** | $\frac{1}{x}$ | Inverts values; extreme compression | $x = 0$ is undefined; reverses order |
| **Square** | $x^2$ | Stretches high values apart | Amplifies large outliers |
| **QQ Plot** | Sample vs Normal Quantiles | Visual check for distribution normality | Diagnostic only; does not predict accuracy |
| **FunctionTransformer** | `FunctionTransformer(func)` | Wraps math functions into Scikit-Learn | Must ensure function handles all input ranges |

---

## 19. Final Mental Model

```text
                          Numerical Feature
                                  │
                       Check Distribution & Skew
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
         Skewness Between                  Skewness > +1.0
          -0.5 and +0.5                    (Right-Skewed)
                 │                                 │
                 ▼                                 ▼
        Keep Raw Feature                  Contains Zeroes?
        (Standardize/Scale)               ┌───────┴───────┐
                                          ▼               ▼
                                         YES              NO
                                          │               │
                                          ▼               ▼
                                     Use log1p(x)      Use log(x)
                                          │               │
                                          └───────┬───────┘
                                                  ▼
                                       Check New Distribution
                                                  ▼
                                       Verify with Cross-Validation!
```
