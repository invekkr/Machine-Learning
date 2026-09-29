# Module 03: Feature Engineering — Mathematical Transformations

When working with numerical features in real-world machine learning, the raw numbers are often distributed in ways that make it difficult for algorithms to learn clean patterns. 

For instance, financial data (like income, house prices, or transaction amounts) typically contains a massive cluster of everyday values and a few extreme values that stretch out across a long tail.

In this module, we study **Mathematical Transformations**—functions like $\log(x)$, $\sqrt{x}$, $1/x$, and $x^2$—that reshape numerical distributions, stabilize variance, and help machine learning models discover stronger mathematical relationships.

---

## 1. What are Mathematical Transformations?

### Understand It First: The Problem of Skewed Data

Consider a dataset containing the annual salaries of employees at a tech company:

```text
Salaries (in thousands):
20k,  25k,  30k,  35k,  40k,  50k,  2L,  5L
```

Notice what is happening:
- Most employees earn between ₹20,000 and ₹50,000.
- A few executives earn ₹2,00,000 (2L) and ₹5,00,000 (5L).

If you plot this data on a chart:
```text
  Number of People
       ▲
       │  ████
       │  ██████
       │  ████████
       │  ██████████                                        █         █
       └───────────────────────────────────────────────────────────────────► Salary
         ₹20k - ₹50k (Huge cluster!)                       ₹2L       ₹5L
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
1. **Reduce skewness:** Pull distant values inward so the data is less lopsided.
2. **Stabilize variance:** Prevent prediction errors from exploding on large numbers.
3. **Make relationships easier to learn:** Turn curved, exponential relationships into straight lines.
4. **Make data more suitable for model assumptions:** Help satisfy linear model requirements (such as normal residuals).

> [!IMPORTANT]
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
- **Visual:** The peak is on the left, and the **long tail extends towards the right**.
- **Intuition:** Most values are relatively small, and a few values are extremely large.
- **Math:** A few huge numbers pull the average up: $\text{Mean} > \text{Median}$.
- **Real-World Examples:**
  - **Income / Salary:** Most people earn average wages; a few billionaires stretch the right tail.
  - **House Prices:** Most homes sell at standard prices; a few luxury mansions create a long right tail.
  - **City Population:** Millions of small towns; a few mega-cities (Tokyo, Mumbai, New York).
  - **Transaction Amounts:** Most retail purchases are small; rare commercial orders are huge.

### 2. Left-Skewed (Negative Skewness):
- **Visual:** The peak is on the right, and the **long tail extends towards the left**.
- **Intuition:** Most values are relatively large, and a few values are very small.
- **Math:** A few tiny numbers pull the average down: $\text{Mean} < \text{Median}$.
- **Real-World Examples:**
  - **Human Lifespan / Age of Death:** Most people live past 65; relatively few die in childhood.
  - **Scores on an Easy Exam:** Most students score 80–95%; a few who missed the exam score 10–20%.
  - **Retirement Age:** Most retire around 60–65; a few retire very early at 30.

> [!NOTE]
> **Formal Definition:**  
> **Skewness** describes the asymmetry of a probability distribution around its central value (mean). A perfectly symmetric distribution has a skewness of $0$.

---

## 3. Why Apply Mathematical Transformations?

Not every machine learning model needs transformations. However, they are valuable in specific situations:

### 1. Highly Skewed Features
When a feature has a skewness greater than $+1.0$, algorithms that calculate distances or slopes can be completely dominated by the handful of massive numbers in the tail.

### 2. Changing Variance (Heteroscedasticity)
In financial data, high earners often exhibit far greater variance in spending than low earners. Logarithmic transformations stabilize this spread, making prediction errors more uniform across all ranges.

### 3. Non-Linear Relationships
Suppose an employee's salary grows exponentially with years of experience:
$$\text{Salary} \approx e^{\text{Experience}}$$
If you fit a standard Linear Regression model directly, a straight line will fail to capture the exponential curve.  
However, if you transform salary using a logarithm:
$$\ln(\text{Salary}) \approx \text{Experience}$$
The relationship suddenly becomes a **clean straight line** that Linear Regression can fit with high accuracy!

### 4. Satisfying Model Assumptions
Linear models perform best when the relationship between features and target is linear and the residuals (errors) are normally distributed. Reshaping skewed features often helps satisfy these conditions.

> [!IMPORTANT]
> The decision to apply a transformation must **always be evaluated based on the actual dataset and model performance**, not applied as a blind ritual.

---

## 4. Log Transformation ($\log(x)$)

### Understand It First: Compressing Extremes
Consider these numbers:
```text
10,  20,  30,  40,  50,  100,  500,  1000
```
Notice how $10$ to $50$ are clustered tightly together (differences of $10$), while $100$ to $1,000$ are separated by massive gulfs ($400$ and $500$).

When we take the logarithm, look at what happens:
$$\log_{10}(10) = 1, \quad \log_{10}(100) = 2, \quad \log_{10}(1,000) = 3$$

While the original numbers grew by factors of ten ($10 \to 100 \to 1,000$), their logarithms grew by simple single steps ($1 \to 2 \to 3$)!

> **The Log Principle:**  
> Logarithm compresses large values much more strongly than smaller values, pulling the distant right tail inward toward the center.

### Formal Definition:
A logarithmic transformation replaces each value $x$ with its logarithm:
$$x' = \log(x)$$

### Formula & Symbol Breakdown:
$$x' = \ln(x) \quad \text{or} \quad x' = \log_{10}(x)$$

- $x$: Original raw positive feature value ($x > 0$).
- $x'$: Transformed value.
- $\ln$: Natural logarithm (base $e \approx 2.71828$).
- $\log_{10}$: Common logarithm (base $10$).
- In data science, `np.log()` computes the natural logarithm ($\ln$). Different log bases exist ($\log_2, \log_{10}, \ln$), but they all compress large values similarly because they differ only by a constant multiplier ($\log_b(x) = \frac{\ln(x)}{\ln(b)}$).

### Small Numerical Example & Step-by-Step Calculation:
Let raw feature values be: $[10, 100, 1000]$
- $x = 10 \implies \log_{10}(10) = 1.0$
- $x = 100 \implies \log_{10}(100) = 2.0$
- $x = 1000 \implies \log_{10}(1000) = 3.0$

The original ratio of maximum to minimum was $\frac{1000}{10} = \mathbf{100}$.  
After log transformation, the ratio is $\frac{3.0}{1.0} = \mathbf{3.0}$. The extreme spread has been safely compressed.

### Python Implementation:
```python
import numpy as np

# Sample skewed values
x = np.array([10.0, 20.0, 50.0, 100.0, 500.0, 1000.0])

# Natural log transformation
x_log = np.log(x)

print("Original:", x)
print("Transformed:", np.round(x_log, 3))
```

### Output:
```text
Original: [  10.   20.   50.  100.  500. 1000.]
Transformed: [2.303 2.996 3.912 4.605 6.215 6.908]
```

### Interpretation:
Notice that while $1000$ was **100 times larger** than $10$, in log space $6.908$ is only **3 times larger** than $2.303$. The extreme right tail has been smoothly pulled closer to the rest of the data.

---

## 5. Why Log Transformation Helps

Let's look at the growth pattern:
```text
Original:     10  ──────►  100  ──────►  1000   (Multiplies by 10 each step!)
Log10:         1  ──────►    2  ──────►     3   (Increases by only 1 each step!)
```

Original values grow very quickly, while the logarithm grows much more slowly.

Therefore:
- **Large values** $\longrightarrow$ Strongly compressed.
- **Extreme differences** $\longrightarrow$ Become much smaller.
- **Right skewness** $\longrightarrow$ Significantly reduced.

> [!WARNING]
> **IMPORTANT:**  
> Do **NOT** claim that log transformation always makes data normal.  
> It compresses positive right-hand skewness, but depending on the underlying distribution, the result may still be bimodal, uniform, or slightly skewed. Always verify with visualization and statistical tests.

---

## 6. $\log(x)$ vs. $\log_{1p}(x)$

### The Problem with Plain $\log(x)$:
1. $\log(0)$ is **mathematically undefined** ($-\infty$).
2. $\log(\text{negative number})$ is **impossible** for real numbers.
3. Therefore, standard $\log(x)$ **strictly requires $x > 0$**.

If your dataset contains zero values (e.g., number of purchases = $0$, bonus = ₹$0$, website visits = $0$), running `np.log(0)` returns `-inf` and crashes machine learning algorithms!

### The Solution: $\log_{1p}(x)$
$$\log_{1p}(x) = \log(1 + x)$$

### Why Adding 1 Fixes Everything:
When $x = 0$:
$$\log_{1p}(0) = \log(1 + 0) = \log(1) = \mathbf{0.0}$$

- A count of $0$ stays cleanly at $0.0$!
- For positive numbers, $\log(1 + x) \approx \log(x)$ for large $x$, preserving the exact same compressive superpower.

### Small Numerical Example:
Values: $[0, 1, 10, 100]$
- $x = 0 \implies \ln(1 + 0) = \ln(1) = \mathbf{0.000}$
- $x = 1 \implies \ln(1 + 1) = \ln(2) \approx \mathbf{0.693}$
- $x = 10 \implies \ln(1 + 10) = \ln(11) \approx \mathbf{2.398}$
- $x = 100 \implies \ln(1 + 100) = \ln(101) \approx \mathbf{4.615}$

### Python Implementation:
```python
import numpy as np

x_with_zero = np.array([0.0, 1.0, 10.0, 100.0])

# np.log(x_with_zero) -> RuntimeWarning: divide by zero encountered in log [-inf, 0, 2.30, 4.60]
x_log1p = np.log1p(x_with_zero)

print("Original:", x_with_zero)
print("np.log1p:", np.round(x_log1p, 3))
```

### When to Use Each:
- Use **`np.log(x)`** when you know all values are strictly positive ($x > 0$).
- Use **`np.log1p(x)`** whenever the feature can contain zero values ($x \ge 0$).

---

## 7. Reciprocal Transformation ($1/x$)

### Intuition:
A reciprocal inverts every number by dividing 1 by that number:
$$x \longrightarrow \frac{1}{x}$$

### Formula:
$$x' = \frac{1}{x}$$

- $x$: Original feature value ($x \ne 0$).
- $x'$: Transformed reciprocal value.

### Small Numerical Example & Step-by-Step Calculation:
Values: $[2, 4, 10]$
- $x = 2 \implies \frac{1}{2} = \mathbf{0.50}$
- $x = 4 \implies \frac{1}{4} = \mathbf{0.25}$
- $x = 10 \implies \frac{1}{10} = \mathbf{0.10}$

Large values become small fractions!

### Two Critical Properties of Reciprocal:
1. **Reverses Ordering for Positive Numbers:**
   - In original data: $2 < 4 < 10$.
   - After reciprocal: $0.50 > 0.25 > 0.10$.
   - The smallest number ($2$) becomes the largest value ($0.50$), and the largest number ($10$) becomes the smallest ($0.10$).
2. **Zero is Undefined:**
   - $\frac{1}{0}$ is undefined (division by zero). Zero cannot be processed directly with a reciprocal transformation.

### Python Implementation:
```python
import numpy as np

x = np.array([2.0, 4.0, 10.0, 50.0])
x_reciprocal = 1.0 / x

print("Original:", x)
print("Reciprocal:", np.round(x_reciprocal, 3))
```

### When to Use:
Reciprocal is an extremely strong transformation. It is commonly used for rate features (e.g., converting time per task $\to$ tasks completed per hour).

---

## 8. Square Transformation ($x^2$)

### Intuition:
The square transformation multiplies each number by itself. It does the opposite of a square root: **larger values grow much faster than smaller values**.

### Formula:
$$x' = x^2$$

- $x$: Original feature value.
- $x'$: Squared value.

### Small Numerical Example & Step-by-Step Calculation:
Values: $[2, 3, 4, 5]$
- $x = 2 \implies 2^2 = \mathbf{4}$ (difference of $5$)
- $x = 3 \implies 3^2 = \mathbf{9}$ (difference of $7$)
- $x = 4 \implies 4^2 = \mathbf{16}$ (difference of $9$)
- $x = 5 \implies 5^2 = \mathbf{25}$

Notice that the gaps between consecutive numbers widen: $5 \to 7 \to 9$.

### When Might It Help?
Square transformations can **sometimes help with certain left-skewed distributions** where we want to stretch out the dense cluster of high values on the right and pull them away from each other.

> [!WARNING]
> **CRITICAL PHRASING:**  
> Do **NOT** say: *"Square transformation fixes left skewness."*  
> Say: *"It can sometimes help reshape certain left-skewed distributions."*

### Python Implementation:
```python
import numpy as np

x = np.array([2.0, 3.0, 4.0, 5.0])
x_square = np.square(x)  # or x ** 2

print("Original:", x)
print("Squared:", x_square)
```

---

## 9. Square Root Transformation ($\sqrt{x}$)

### Intuition:
A square root compresses large values, but it is **much milder (less aggressive) than a logarithm**.

### Formula:
$$x' = \sqrt{x}$$

- $x$: Original feature value ($x \ge 0$).
- $x'$: Transformed square-root value.

### Small Numerical Example & Step-by-Step Calculation:
Values: $[4, 9, 16, 100]$
- $x = 4 \implies \sqrt{4} = \mathbf{2}$
- $x = 9 \implies \sqrt{9} = \mathbf{3}$
- $x = 16 \implies \sqrt{16} = \mathbf{4}$
- $x = 100 \implies \sqrt{100} = \mathbf{10}$

### Square Root vs. Logarithm Comparison:
- **Zero-Friendly:** Unlike plain $\log(x)$, $\sqrt{0} = 0$. It handles zero without needing any shift.
- **Milder Compression:** While $\ln(10,000) \approx 9.2$, $\sqrt{10,000} = 100$. Square root compresses values, but does not squash them as aggressively as log.
- **Ideal For:** Moderately right-skewed count features (e.g., household size, number of items in a shopping cart).

### Python Implementation:
```python
import numpy as np

x = np.array([0.0, 4.0, 9.0, 16.0, 100.0])
x_sqrt = np.sqrt(x)

print("Original:", x)
print("Square Root:", np.round(x_sqrt, 3))
```

---

## 10. Compare Transformations

| Transformation | Formula | General Effect | Common Use | Important Limitation |
| :--- | :---: | :--- | :--- | :--- |
| **Log** | $\ln(x)$ | Strong compression of large values | Strongly right-skewed features | Strictly requires $x > 0$ |
| **Log1p** | $\ln(1 + x)$ | Strong compression; keeps $0 \to 0$ | Right-skewed data with zeroes | Requires $x \ge 0$ |
| **Square Root** | $\sqrt{x}$ | Mild compression of large values | Moderately right-skewed counts | Requires $x \ge 0$ |
| **Reciprocal** | $\frac{1}{x}$ | Extreme inversion (large $\to$ tiny) | Severe right-skewed rates | $x = 0$ is undefined; reverses order |
| **Square** | $x^2$ | Stretches high values apart | Certain left-skewed features | Amplifies large outliers |

### Basic Memory Rule:
- **Right-Skewed:** Log / Log1p / Square Root are common candidates.
- **Left-Skewed:** Square may sometimes help.

> [!NOTE]
> **This is NOT an absolute rule.** Always inspect the transformed distribution visually and validate the impact on your model with cross-validation.

---

## 11. How to Detect Skewness

Before applying any transformation, inspect and quantify the feature using three primary approaches:

### A. Histogram
A histogram groups continuous numbers into bins and displays their counts:
- **Symmetric:** Balanced bell shape centered in the middle.
- **Right-Skewed:** Tall bar clusters on the left, long dragging tail to the right.
- **Left-Skewed:** Tall bar clusters on the right, long dragging tail to the left.

### B. KDE / Density Plot
A **Kernel Density Estimation (KDE)** plot smooths the histogram into a continuous curve, making it easy to identify peaks (modes) and tail asymmetry.

### C. Box Plot
A box plot displays the 5-number summary (Min, $Q_1$, Median, $Q_3$, Max):
- In a **symmetric** distribution, the median line sits in the exact middle of the box, and both whiskers are equal length.
- In a **right-skewed** distribution, the median line is pushed toward the bottom/left of the box, the top/right whisker is much longer, and outlier dots appear far to the right.

---

## 12. Probability Density Function (PDF)

### Explain It from a Beginner Perspective:
When we measure a continuous variable (like height, temperature, or salary), the probability of finding a person who earns *exactly* ₹43,219.827364... is practically zero because there are infinite possible decimal values.

Instead of asking *"What is the probability of this exact number?"*, we ask:
> *"How densely packed are values around this region?"*

This is what a **Probability Density Function (PDF)** describes:
- **Formal Intuition:** A PDF, written as $f(x)$, represents the **relative likelihood / density of values** across the continuous spectrum.
- **Area Under the Curve:** The total area under the entire PDF curve always equals **$1.0$ (or $100\%$ probability)**:
  $$\int_{-\infty}^{\infty} f(x) \, dx = 1.0$$
- **Finding Probability:** The probability that a value falls between range $a$ and $b$ is the area under the curve between $a$ and $b$:
  $$P(a \le X \le b) = \int_{a}^{b} f(x) \, dx$$

### How Density Curves Reveal Skewness:
- **Symmetric (Normal PDF):** Bell curve centered around mean $\mu$. Area is split evenly ($50\%$ left, $50\%$ right).
- **Right-Skewed PDF:** Density peak is concentrated at low values; the right curve stretches out in a long shallow tail.
- **Left-Skewed PDF:** Density peak is concentrated at high values; the left curve stretches out in a long shallow tail.

> [!NOTE]
> **Practical ML Purpose:** In everyday feature engineering, we don't calculate calculus integrals by hand. We plot the estimated density curve (via Seaborn's `sns.kdeplot()`) to visually diagnose whether a feature's distribution is lopsided before deciding on a transformation.

---

## 13. Quantile-Quantile (QQ) Plots

A **Quantile-Quantile (QQ) Plot** is the standard diagnostic visualization used in machine learning to test whether a feature follows a normal distribution.

### What is a Quantile?
A quantile is simply a division mark that splits sorted data into equal fractions:
- The **0.50 quantile (50th percentile)** is the **Median** (half the data is below it).
- The **0.25 quantile (25th percentile)** is $Q_1$ (one-quarter of the data is below it).

### How a QQ Plot Works:
1. It sorts your actual sample values into empirical quantiles ($Y$-axis).
2. It calculates where those exact quantiles *should* fall if the data came from a theoretical normal distribution ($X$-axis).
3. It plots the pairs against a red $45^\circ$ reference line:

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

- **If points hug the straight red line:** The feature matches the normal distribution closely.
- **If points curve upward away from the line:** The feature has a heavy right tail (right-skewed).
- **If points curve downward away from the line:** The feature has a heavy left tail (left-skewed).

### Python Implementation:
```python
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate a QQ plot
stats.probplot(df['Income'], dist="norm", plot=plt)
plt.title("QQ Plot: Income vs. Theoretical Normal")
plt.show()
```

> [!IMPORTANT]
> A QQ plot provides visual diagnostic evidence about distribution shape. It does **not** alone determine whether an ML model will perform better.

---

## 14. Practical Experiment

In our accompanying notebook ([`mathematical_transformations.ipynb`](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/07_Mathematical_Transformations/mathematical_transformations.ipynb)), we generated a realistic synthetic dataset ($N = 200$) with a strongly right-skewed `Income` feature:

1. **Original Skewness:**
   - Raw `Income` skewness = **$+1.377$** (strongly right-skewed, $\text{Mean} = \$43,365 > \text{Median} = \$35,465$).
2. **Log Transformation:**
   - Applied `np.log1p(df['Income'])`.
   - Transformed skewness = **$+0.285$** (dropped from $+1.377$ to $+0.285$, well within the $[-0.5, +0.5]$ symmetric range!).
3. **QQ Plot Verification:**
   - **Before:** Points curved strongly upward away from the red diagonal line on the right.
   - **After:** Points hugged the straight diagonal reference line closely across all quantiles.

---

## 15. Scikit-Learn `FunctionTransformer`

### The Problem with Manual Assignments:
In raw exploratory scripts, data scientists often write:
```python
df['Income'] = np.log1p(df['Income'])
```
While simple, this manual approach causes severe problems in production:
- It cannot be bundled into an exported pipeline object (`model.pkl`).
- It cannot be validated inside Scikit-Learn's `cross_val_score()`.
- It risks train/test leakage if manual code is executed out of order.

### The Solution: `FunctionTransformer`
> **`FunctionTransformer`** wraps any standard Python or NumPy mathematical function into a formal Scikit-Learn Transformer with `fit()` and `transform()` methods.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Create the transformer object
log_transformer = FunctionTransformer(np.log1p, feature_names_out='one-to-one')

# Fit and transform
income_transformed = log_transformer.fit_transform(df[['Income']])
```

> **Why `feature_names_out='one-to-one'`?**  
> Passing `feature_names_out='one-to-one'` informs Scikit-Learn that the transformer preserves column names 1-to-1, allowing downstream methods like `ColumnTransformer.get_feature_names_out()` to function without errors.

---

## 16. `FunctionTransformer` + `ColumnTransformer`

Real-world datasets contain a mix of column types:
- `Income`: Skewed numerical feature $\longrightarrow$ Apply `FunctionTransformer(np.log1p)`
- `Age`: Normal numerical feature $\longrightarrow$ Apply `StandardScaler()`
- `City`: Categorical feature $\longrightarrow$ Apply `OneHotEncoder()`

We use **`ColumnTransformer`** as the traffic controller:

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
FunctionTransformer(np.log1p) StandardScaler   OneHotEncoder
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
               Ready for Machine Learning Model!
```

### Python Implementation:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
import numpy as np

preprocessor = ColumnTransformer(
    transformers=[
        ('log_income', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), ['Income']),
        ('scale_age', StandardScaler(), ['Age']),
        ('ohe_city', OneHotEncoder(sparse_output=False), ['City'])
    ],
    remainder='drop'
)
```

---

## 17. Transformations + `Pipeline`

Now we connect the preprocessor directly to an estimator inside a single, unified `Pipeline`:

```text
Raw Data
   │
   ▼
[ Train / Test Split ]
   │
   ▼
[ Pipeline ]
   │
   ├── Step 1: ColumnTransformer (Math Transform + Scaler + OHE)
   │
   └── Step 2: Estimator (LinearRegression)
   │
   ▼
Predictions
```

### Python Implementation:
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit on training data only
pipeline.fit(X_train, y_train)

# Predict on test data
y_pred = pipeline.predict(X_test)
```

---

## 18. Train / Test Data Leakage Prevention

### The Golden Rule of Preprocessing:
Always split your data **FIRST** before applying any preprocessing or model fitting.

```text
WRONG:
Fit Preprocessing on Full Data ──► Split Train/Test  (DATA LEAKAGE!)

CORRECT:
Split Train/Test ──► Fit on Train Data Only ──► Transform Test Data
```

### Fixed Math Functions vs. Learned Transformers:
- **Learned Transformers (`StandardScaler`, `MinMaxScaler`):** Learn parameters ($\mu, \sigma, \min, \max$) from the data. If fitted on test data, information leaks.
- **Fixed Math Functions (`np.log1p`, `np.sqrt`):** Apply a fixed mathematical rule ($x \to \ln(1+x)$) with no learned parameters.
- **Why Pipeline is Still Essential:** Even though $\log(1+x)$ has no learned parameters, encapsulating it inside a `Pipeline` guarantees that raw test data and live production inputs are transformed identically without manual wrangling or accidental omission.

---

## 19. Do All ML Models Need Mathematical Transformations?

### Tree-Based Models: **NO**
Decision Trees, Random Forests, XGBoost, and LightGBM make decisions using threshold splits:
$$\text{Is Income} \le ₹50,000?$$

Because logarithms and square roots are **strictly monotonic** (they never change the relative order of numbers: if $A < B$, then $\log(A) < \log(B)$), the tree's split point simply shifts to $\log(50,000)$. The partition of data points remains 100% identical!

> [!NOTE]
> **Tree-based models generally do not require transformations solely to make features normally distributed.**

---

## 20. Which Models May Benefit More?

Mathematical transformations are most beneficial for:
1. **Linear Regression & Logistic Regression:** Benefit when transformations linearize curved relationships or stabilize error variance.
2. **Distance-Based Algorithms (KNN, K-Means):** Benefit when transformations pull extreme tail outliers inward so they don't dominate distance calculations.
3. **Neural Networks:** Benefit from bounded, well-scaled inputs for smoother gradient descent.

> [!IMPORTANT]
> **CRITICAL STATISTICAL CORRECTION:**  
> Do **NOT** teach: *"Linear Regression requires every feature to be normally distributed."*  
> Linear Regression does not require inputs to be normal. It assumes:
> 1. The **relationship between features and target is linear**.
> 2. The **residuals (prediction errors) are normally distributed** with constant variance (homoscedasticity).  
> Transforming non-linear or skewed features often helps satisfy these linear and residual assumptions.

---

## 21. Feature Scaling vs. Mathematical Transformation

Do not confuse Feature Scaling with Mathematical Transformations:

| Dimension | Feature Scaling (StandardScaler, MinMaxScaler) | Mathematical Transformation (Log, Sqrt, Reciprocal) |
| :--- | :--- | :--- |
| **Primary Goal** | Changes the **scale / range / magnitude** of features | Changes the **shape / distribution / mathematical form** |
| **Effect on Skewness** | **Zero effect.** Skewness is completely unchanged! | **Reduces or alters skewness.** |
| **Effect on Shape** | Centers at 0 or bounds between $[0, 1]$; shape stays identical | Compresses or stretches the tail; shape is reshaped |
| **Formulas** | $z = \frac{x - \mu}{\sigma}, \quad x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $x' = \ln(1 + x), \quad x' = \sqrt{x}, \quad x' = \frac{1}{x}$ |
| **Order Preservation** | Strictly preserves relative distances and ratios | Non-linear: changes relative distances between points |

> **Mental Model:**  
> - Scaling changes the ruler (meters to feet).  
> - Transformation changes the geometry (curved to straight).

---

## 22. How to Choose a Transformation

Follow this practical 9-step decision workflow:

```text
                  1. Understand the Feature
                             │
                             ▼
                  2. Check Distribution (Histogram/KDE)
                             │
                             ▼
                  3. Identify Skewness (Calculate skew())
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Skewness in [-0.5, +0.5]           Skewness > +1.0 (Right-Skewed)
            │                                 │
            ▼                                 ▼
   Keep Raw Feature                  4. Choose Possible Transformation
   (Standardize/Scale)               (Contains 0? -> log1p; No 0? -> log; Mild? -> sqrt)
                                              │
                                              ▼
                                     5. Apply Transformation
                                              │
                                              ▼
                                     6. Check Distribution Again (KDE / QQ Plot)
                                              │
                                              ▼
                                     7. Train Model on Training Data
                                              │
                                              ▼
                                     8. Evaluate with Cross-Validation
                                              │
                                              ▼
                                     9. Compare Performance vs. Baseline!
```

---

## 23. Transformation Does NOT Automatically Improve Accuracy

Never assume that applying a transformation will automatically make your model perform better.

### A Conceptual Example:
- Model **without** transformation $\longrightarrow$ Accuracy = **90%** (or $\text{MAE} = \$1,200$)
- Model **with** log transformation $\longrightarrow$ Accuracy = **88%** (or $\text{MAE} = \$1,400$)

In this scenario, the transformation distorted a relationship that was already linear or suitable for the algorithm, hurting performance!

> **Main Principle:**  
> **Normal distribution $\ne$ automatically better model.**  
> Use transformations when they make sense for the feature and model assumptions, and **always validate** whether they actually improve your evaluation metrics on holdout test data.

---

## 24. Model Comparison Experiment (Real Results)

In Section 17 of our notebook ([`mathematical_transformations.ipynb`](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/07_Mathematical_Transformations/mathematical_transformations.ipynb)), we conducted a rigorous head-to-head experiment comparing a Baseline Linear Regression model (StandardScaler only) against a Transformed Pipeline (`FunctionTransformer(np.log1p)`) on the exact same 25% holdout test split:

| Model Workflow | Test MAE ($) | Test $R^2$ Score | Observation |
| :--- | :---: | :---: | :--- |
| **Without Transformation (Baseline)** | **$1,036.87** | **0.5402** | Struggles with compressed high-income range |
| **With Log Transformation (`np.log1p`)** | **$883.27** | **0.6275** | **~$153 lower MAE** and **+0.087 jump in $R^2$** |

Because customer spending grew logarithmically with income in our real-world scenario, log-transforming `Income` straightened the relationship, allowing Linear Regression to fit the data significantly more accurately!

---

## 25. Final Quick Reference & Mental Model

### Quick Reference Table:

| Concept | Formula | Main Purpose | Important Limitation |
| :--- | :---: | :--- | :--- |
| **Log** | $\ln(x)$ | Compresses extreme right-skewed values | Strictly requires $x > 0$ |
| **Log1p** | $\ln(1 + x)$ | Compresses right-skewed values with zeroes | Requires $x \ge 0$ |
| **Square Root** | $\sqrt{x}$ | Mild compression for moderate skewness | Requires $x \ge 0$ |
| **Reciprocal** | $\frac{1}{x}$ | Extreme inversion (large $\to$ tiny) | $x = 0$ is undefined; reverses order |
| **Square** | $x^2$ | Stretches high values apart | Amplifies large outliers |
| **PDF** | $f(x), \ \int f(x)dx=1$ | Describes relative density across continuous range | Theoretical; estimated via KDE |
| **QQ Plot** | Sample vs. Normal Quantiles | Visual check for distribution normality | Diagnostic only; does not predict accuracy |
| **FunctionTransformer** | `FunctionTransformer(func)` | Wraps math functions into Scikit-Learn | Function must handle full input domain |

### Final Mental Model Flowchart:

```text
                           FEATURE
                              │
                              ▼
                      Check Distribution
                              │
                              ▼
                        Is it skewed?
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
              NO                            YES
               │                             │
               ▼                             ▼
       Keep Original                Consider Transformation
       (Scale / Standardize)                 │
                                             ▼
                                Choose Suitable Transformation
                                             │
                                             ▼
                                         Transform
                                             │
                                             ▼
                                     Check Distribution
                                             │
                                             ▼
                                        Train Model
                                             │
                                             ▼
                                     Cross-Validation
                                             │
                                             ▼
                                   Check Model Performance
```

---

## 26. Important Points to Preserve

1. **Mathematical transformation** changes the mathematical form and distribution shape of a feature.
2. **Transformation does not automatically make data normally distributed.** It reduces skewness and stabilizes variance.
3. **Right-skewed data** commonly motivates trying log, log1p, or square root transformations.
4. **`log1p(x)`** ($\ln(1+x)$) is essential when zero values are present because $\log(0)$ is undefined.
5. **Reciprocal** ($1/x$) inverts values, reverses relative order for positive numbers, and cannot handle zero.
6. **Square transformation** ($x^2$) increases the relative influence of larger values and can sometimes help reshape certain left-skewed distributions.
7. **Square-root transformation** ($\sqrt{x}$) is generally milder than log transformation and safely handles zero.
8. **QQ plots** compare sample quantiles against theoretical normal quantiles; points on the diagonal indicate a close match to normality.
9. **`FunctionTransformer`** allows NumPy mathematical functions to be integrated seamlessly into Scikit-Learn workflows.
10. **`FunctionTransformer` + `ColumnTransformer`** allows applying transformations only to selected columns while scaling or encoding others.
11. **`Pipeline`** connects data preprocessing and modeling into an automated, leakage-proof workflow.
12. **Tree-based models** (Decision Trees, Random Forests, XGBoost) generally do not need transformations solely for normality because they make decisions using monotonic threshold splits.
13. **Linear Regression** does not require every input feature to be normally distributed; it assumes linear relationships and normal residuals.
14. **Feature Scaling vs. Mathematical Transformation:** Scaling changes range without changing distribution shape or skewness; transformation reshapes the distribution and alters skewness.
15. **Always validate transformations** using actual cross-validation model performance.
16. **Never assume** a transformation automatically improves accuracy.
17. **Always split train and test data FIRST** before fitting any preprocessing to prevent data leakage.
