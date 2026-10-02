# Module 03: Feature Engineering — Handling Missing Numerical Data (Univariate Imputation)

In the previous topic ([Complete Case Analysis](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/10_Complete_Case_Analysis/)), we explored what happens when we delete rows containing missing values. While deleting rows is simple, it often discards valuable training observations and cannot handle missing values during live production predictions.

In this module, we study **Univariate Imputation** — the practice of estimating and filling missing numerical values using statistical summaries from the **same column**, allowing us to retain 100% of our training samples.

```text
Row Deletion (CCA):     [ Age: NaN, Salary: 50,000 ] ──► DISCARD ENTIRE ROW
Imputation:             [ Age: NaN, Salary: 50,000 ] ──► [ Age: 28.0, Salary: 50,000 ] (RETAINED)
```

---

## 1. What is Imputation?

### The Core Idea

**Imputation** means replacing missing (`NaN`) values with an estimated, meaningful replacement value instead of dropping the row.

### Why "Univariate"?
- **Univariate Imputation**: We calculate the replacement value using data strictly from the **single feature itself** (e.g., using only the existing values of `Age` to fill missing `Age`).
- **Multivariate Imputation**: We predict the missing value using relationships across **multiple other features** (e.g., using KNN or linear regression across `Salary`, `Pclass`, and `Fare` to predict `Age`). Multivariate methods will be covered in later modules.

---

## 2. Mean and Median Imputation

The two most common univariate imputation methods are replacing missing values with either the **Mean** (average) or the **Median** (middle value) of the available data.

### 1. Mean Imputation
The arithmetic average of all observed non-null values:

$$\text{Mean} (\mu) = \frac{\sum_{i=1}^{n} x_i}{n}$$

#### Simple Example:
Suppose we have 5 observations for `Age`:
$$[20, \ 25, \ 30, \ 35, \ \text{NaN}]$$

1. Sum of observed values: $20 + 25 + 30 + 35 = 110$
2. Count of observed values: $4$
3. Mean: $\frac{110}{4} = 27.5$
4. Result after imputation: $[20, \ 25, \ 30, \ 35, \ \mathbf{27.5}]$

---

### 2. Median Imputation
The middle value when all observed non-null numbers are sorted in ascending order:

$$\text{Median} = \begin{cases} x_{\frac{n+1}{2}} & \text{if } n \text{ is odd} \\ \frac{x_{\frac{n}{2}} + x_{\frac{n}{2} + 1}}{2} & \text{if } n \text{ is even} \end{cases}$$

#### Simple Example:
Sorted observed values: $20, \ 25, \ 30, \ 35$ ($n=4$, even count).

$$\text{Median} = \frac{25 + 30}{2} = 27.5$$

Result after imputation: $[20, \ 25, \ 30, \ 35, \ \mathbf{27.5}]$

---

## 3. Mean vs. Median: Which Should You Use?

While the mean and median were identical in the symmetrical example above, real-world data is rarely symmetrical.

| Feature Property | Mean Imputation | Median Imputation |
| :--- | :--- | :--- |
| **Outlier Sensitivity** | **Highly Sensitive**: A single huge value pulls the mean upward. | **Robust**: Extreme outliers do not change the middle rank. |
| **Best Distribution** | **Normal / Symmetric** distributions (bell curve). | **Skewed** distributions (salary, house prices, web traffic). |
| **Statistical Meaning** | Center of gravity (mathematical balance point). | 50th percentile (half values below, half above). |

### Real-World Intuition Example:
Imagine 5 employees report their monthly salary:
$$[₹30,000, \ ₹35,000, \ ₹40,000, \ ₹45,000, \ ₹10,00,000 \ (\text{CEO})]$$

- **Mean Salary**: $\frac{30 + 35 + 40 + 45 + 1000}{5} = ₹2,30,000$ (A misleading estimate for regular staff!)
- **Median Salary**: $₹40,000$ (Accurately reflects typical employee compensation!)

> [!TIP]
> **The Golden Rule for Continuous Data:**  
> If the feature is roughly bell-shaped (Normal), use **Mean**.  
> If the feature has outliers or is skewed (Left or Right), use **Median**.

---

## 4. Side Effects & Problems with Mean/Median Imputation

Imputation is practical, but filling missing values with an identical constant creates three measurable side effects:

### Problem 1: Reduced Variance (Artificially Squeezed Data)
Variance measures how spread out numbers are around the center:

$$\text{Var}(X) = \frac{\sum (x_i - \mu)^2}{n - 1}$$

When you replace 100 missing rows with the exact same mean value $\mu$, their deviation from the mean is $( \mu - \mu )^2 = 0$.  
You are adding points with **zero deviation**, which artificially **deflates the overall variance**:

$$\text{Variance After Imputation} < \text{Variance Before Imputation}$$

### Problem 2: Distortion of the Distribution Shape
Piling hundreds of identical values onto the mean creates an unnatural, sharp spike in probability density:

```text
Before Imputation (Original Natural Shape):
         ___
       /     \
     /         \
____/           \____

After Mean Imputation (Artificial Concentration Spike):
          |
         |||  <-- Sharp peak created by hundreds of identical imputed values
       / ||| \
     /         \
____/           \____
```

### Problem 3: Dilution of Covariance and Correlation
When you fill missing values in feature $X$ with a constant while feature $Y$ varies naturally, the linear relationship between $X$ and $Y$ is weakened. Pearson's correlation coefficient ($r$) typically drifts closer to zero.

---

## 5. Arbitrary Value Imputation

### The Core Concept
Instead of calculating a central summary statistic (mean or median), we deliberately pick an **arbitrary, unnatural constant** that lies far outside the feature's realistic domain:

```text
NaN ──► -1       (for Age or Salary, which can never be negative)
NaN ──► 999      (for small integer counts)
NaN ──► -999     (general placeholder)
```

### Why Would Anyone Do This?
In some datasets, **missingness is not random** — the fact that a value is missing is itself an important signal:
- If a loan applicant refuses to provide their income, that refusal correlates strongly with default risk!
- If an emergency patient has no blood pressure recorded, they may have been in cardiac arrest.

By assigning `-1`, tree-based algorithms (Decision Trees, Random Forests) can create a clean split:
$$\text{Is Age} \le 0? \implies \text{Missing Record Cohort}$$

### Limitations of Arbitrary Value Imputation:
- **Terrible for Linear Models**: In Logistic Regression, a value of $-1$ or $999$ gets multiplied by a coefficient $w$, severely warping the decision boundary!
- **Terrible for Distance Models**: In KNN, an age of $999$ creates massive Euclidean distances that ruin neighbor calculations.
- **Creates Artificial Outliers**: Introduces severe distribution skewness.

---

## 6. End-of-Distribution Imputation

### The Core Concept
End-of-Distribution Imputation is a smarter, mathematically disciplined alternative to picking arbitrary numbers like `999`.

Instead of choosing a random number out of thin air, we place missing values at the **far extreme tail of the feature's actual empirical distribution**.

```text
Normal Distribution:
             Mean (μ)
                │
         ╭──────┴──────╮
        ╱   Normal      ╲
       ╱     Data        ╲
______╱                   ╲______[ μ + 3σ ] ◄── Impute missing values here!
-3σ           0           +3σ
```

### 1. For Normally Distributed Features (The 3-Sigma Rule)
From the Empirical Rule of statistics, $99.7\%$ of all data in a Normal distribution lies within $\mu \pm 3\sigma$. We place missing values just past the 3-sigma boundary:

$$\text{Upper End} = \text{Mean} + 3 \times \text{Std Dev}$$
$$\text{Lower End} = \text{Mean} - 3 \times \text{Std Dev}$$

#### Example:
If `Age` has $\text{Mean} = 30$ and $\text{Std Dev} = 14.5$:
$$\text{Upper End} = 30 + (3 \times 14.5) = 30 + 43.5 = 73.5$$
We replace all `NaN` values with $73.5$.

---

### 2. For Skewed Features (The IQR Rule)
For skewed distributions, we use the standard boxplot outlier fence formula:

$$\text{IQR} = Q_3 - Q_1$$
$$\text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$
$$\text{Extreme Upper Fence} = Q_3 + 3.0 \times \text{IQR}$$

#### Why Use End-of-Distribution?
1. It flags missingness for tree models just like arbitrary imputation.
2. It scales dynamically with the data (e.g., if salary is in millions, the imputed value naturally scales into millions).

---

## 7. The 4-Pillar Imputation Validation Checklist

After imputing missing values, never assume the job is done. Always run these 4 simple checks:

```text
                        Imputed Feature
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼                  ▼
     1. Variance        2. Distribution    3. Correlation       4. Outliers
     Check ratio:        Overlay KDE:       Check change in     Check boxplot:
     Var_new / Var_old   Look for spikes    Pearson's r         Did we add noise?
```

1. **Variance Ratio**: Compute $\frac{\text{Variance}_{\text{after}}}{\text{Variance}_{\text{before}}}$. A ratio close to $1.0$ ($0.90 - 1.05$) means variance was well-preserved.
2. **Distribution Shape**: Overlay KDE curves before vs. after. Ensure the central peak did not become artificially distorted.
3. **Correlation**: Compute correlation with target or related features ($r_{\text{before}}$ vs. $r_{\text{after}}$).
4. **Outlier Boundaries**: Inspect boxplots to confirm imputed values did not trigger false outlier flags (unless using End-of-Distribution intentionally).

---

## 8. Summary Comparison of Univariate Techniques

| Technique | Replacement Formula | Best Used When... | Major Risk |
| :--- | :--- | :--- | :--- |
| **Mean** | $\mu = \frac{\sum x}{n}$ | Data is bell-shaped (Normal) with low missingness (<5%). | Severe outlier distortion; deflates variance. |
| **Median** | 50th Percentile | Data is skewed or contains outliers with low missingness. | Deflates variance; creates central peak. |
| **Arbitrary** | User constant ($-1, 999$) | Tree-based models where missingness contains signal (MNAR). | Destroys linear models and distance calculations. |
| **End of Tail** | $\mu + 3\sigma$ or $Q_3 + 1.5\text{IQR}$ | Tree-based models needing automated missingness flags. | Alters distribution tail and inflates variance. |

---

## 9. Scikit-Learn `SimpleImputer` Workflow & Data Leakage Prevention

In production machine learning, we use Scikit-Learn's `SimpleImputer`.

> [!IMPORTANT]
> **Strict Rule Against Data Leakage:**  
> Always calculate the mean, median, or extreme boundary **strictly on the training data** (`X_train`), and use those learned values to transform both `X_train` and `X_test`!

```text
Raw X_train ──► imputer.fit(X_train) ──► imputer.transform(X_train) ──► Clean X_train
                      │ (Learns Mean/Median)
                      ▼
Raw X_test  ───────────────────────────► imputer.transform(X_test)  ──► Clean X_test
```

---

## 10. Complete Case Analysis (CCA) vs. Univariate Imputation

| Dimension | Complete Case Analysis (CCA) | Univariate Imputation |
| :--- | :--- | :--- |
| **Action on Rows** | Deletes rows containing nulls | Keeps 100% of rows |
| **Dataset Size** | Decreases ($N_{\text{clean}} < N_{\text{raw}}$) | Preserved ($N_{\text{clean}} = N_{\text{raw}}$) |
| **Production Ready** | Fails if live user inputs a null | Seamlessly fills live incoming nulls |
| **Data Preservation** | Discards valid data in other columns | Retains valid data in other columns |
| **Distribution Risk** | Can cause selection bias if not MCAR | Can reduce variance and alter distribution shape |

---

## 11. Final Mental Model

```text
                     Missing Numerical Feature
                                 │
                                 ▼
                     Is missingness meaningful?
                    (Does missingness carry signal?)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
             YES (MNAR)                      NO (MCAR / MAR)
                 │                               │
                 ▼                               ▼
       Tree-Based Model?                Check Distribution Shape
                 │                               │
        ┌────────┴────────┐              ┌───────┴───────┐
        ▼                 ▼              ▼               ▼
  End-of-Tail         Arbitrary        Normal          Skewed
(μ + 3σ or IQR)      Value (-1)         Mean           Median
```
