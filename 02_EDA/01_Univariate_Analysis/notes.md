# Module 02: Exploratory Data Analysis — Univariate Analysis

**Univariate Analysis** is the analysis of a single variable in isolation.

Its primary goal is to describe and summarize the distribution of the variable, evaluate its central tendency and dispersion, and detect unusual anomalies or outliers before studying relationships between variables.

---

## 1. Categorical Data Analysis

Categorical variables represent groups or labels rather than numerical measurements:
- **Nominal:** Categories with no inherent ordering (e.g., `Sex`: male/female; `Embarked`: S/C/Q).
- **Ordinal:** Categories with a meaningful rank or order (e.g., `Pclass`: 1st > 2nd > 3rd; Education level).

### 1.1 Frequency Distribution & `value_counts()`
A frequency distribution counts how many observations fall into each distinct category:
```python
# Raw counts
df["Sex"].value_counts()

# Proportions / Percentages
df["Sex"].value_counts(normalize=True) * 100
```

### 1.2 Categorical Visualizations

| Chart Type | Purpose & Best Practice | When to Avoid |
| :--- | :--- | :--- |
| **Count Plot** (`sns.countplot`) | Plots bars where height equals the frequency of each category. Easy to read and compare. | Avoid when there are 50+ categories (cardinality too high). |
| **Bar Chart** (`plt.bar`) | Plots custom aggregated values (e.g., percentages or survival rate per category). | When categories have no meaningful distinction. |
| **Pie Chart** (`plt.pie`) | Represents parts of a whole (must sum to 100%). | Avoid when comparing more than 3–4 categories; human eyes struggle to judge angle/area differences. |
| **Donut Chart** | A pie chart with a circular cutout in the center. Focuses attention on arc lengths rather than area. | Same limitations as pie charts. |
| **Pareto Chart** | Combines a descending bar chart of frequencies with a cumulative percentage line. Visualizes the **80/20 rule** (Pareto Principle). | For variables without actionable frequency skew. |

---

## 2. Numerical Data Analysis

Numerical variables represent quantities or counts:
- **Discrete:** Countable distinct values with no in-between fractions (e.g., `SibSp` = 0, 1, 2; `Parch` = 0, 1).
- **Continuous:** Measurable values that can take any fractional value along an interval (e.g., `Age` = 28.5; `Fare` = 71.28).

### 2.1 Histograms & Bins
A **histogram** groups continuous values into consecutive non-overlapping intervals called **bins** and counts how many data points fall into each bin.

* **Bin selection matters:** Too few bins over-smooth the distribution (hiding peaks); too many bins create a noisy, jagged plot. A good rule of thumb is Sturges' Rule: $k = \lceil 1 + \log_2(N) \rceil$ or square-root rule $k = \sqrt{N}$.

```python
import seaborn as sns

sns.histplot(df["Age"], bins=20)
```

### 2.2 Kernel Density Estimation (KDE)
**KDE** estimates the continuous probability density function (PDF) of a random variable by placing a smooth Gaussian curve over each data point and summing them.
- It produces a smooth curve independent of arbitrary bin edges.
- Area under the KDE curve equals 1.0.

```python
# Histogram + KDE combined
sns.histplot(df["Age"], kde=True, bins=20)
```

### 2.3 Box Plot & Five-Number Summary
A **box plot** (box-and-whisker plot) provides a graphical summary of data based on Tukey's five-number summary:
1. **Minimum:** Smallest value within $Q1 - 1.5 \times IQR$
2. **First Quartile ($Q1$):** 25th percentile
3. **Median ($Q2$):** 50th percentile (central line)
4. **Third Quartile ($Q3$):** 75th percentile
5. **Maximum:** Largest value within $Q3 + 1.5 \times IQR$

```text
       ┌───[Q1]───────[Median]────────[Q3]───┐
──|────│                 │                   │────|────── ○ (Outlier)
 Min   └─────────────────────────────────────┘   Max
       ◄─────────────── IQR ────────────────►
```

### 2.4 Interquartile Range (IQR) & Outlier Detection
The **IQR** measures statistical dispersion:
$$IQR = Q3 - Q1$$

Any point falling outside the following fences is flagged as a **potential outlier**:
$$\text{Lower Bound} = Q1 - (1.5 \times IQR)$$
$$\text{Upper Bound} = Q3 + (1.5 \times IQR)$$

### 2.5 Violin Plot
Combines a **box plot** and a **rotated KDE plot** on each side.
* Shows both the summary statistics (median, quartiles) and the actual multi-modal density shape of the data.

### 2.6 Empirical Cumulative Distribution Function (ECDF)
Plots the cumulative percentage of data points less than or equal to a given value:
* **Y-axis:** Fraction of data from 0.0 to 1.0 (0% to 100%).
* **Advantage:** No binning bias; shows every single data point; percentile lookup is trivial (e.g., "What % of passengers paid under £50?").

```python
sns.ecdfplot(data=df, x="Fare")
```

### 2.7 Strip / Dot Plot
Plots raw data points along a single axis with slight random horizontal jitter to avoid overplotting. Excellent for small to medium sample sizes.

---

## 3. Measures of Central Tendency

Central tendency represents the central or typical value of a distribution.

| Metric | Formula / Definition | Outlier Sensitivity | Best Used When |
| :--- | :--- | :--- | :--- |
| **Mean** | $\bar{X} = \frac{1}{N}\sum_{i=1}^N X_i$ | **Highly Sensitive** | Data is symmetric and bell-shaped (Normal distribution). |
| **Median** | 50th percentile (middle sorted value) | **Robust (Resistant)** | Data is skewed or contains extreme outliers (e.g., income, house prices, `Fare`). |
| **Mode** | Most frequently occurring value | **Resistant** | Categorical variables or finding the peak in discrete counts. |

---

## 4. Measures of Dispersion (Spread)

Dispersion quantifies how tightly grouped or widely scattered data points are around the center.

### 4.1 Range
$$\text{Range} = X_{\text{max}} - X_{\text{min}}$$
Simple but extremely sensitive to outliers.

### 4.2 Variance ($\sigma^2$ or $s^2$)
Average of squared deviations from the mean:
$$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (X_i - \bar{X})^2$$
Squared units make it non-intuitive to interpret directly.

### 4.3 Standard Deviation ($s$ or $\sigma$)
Square root of variance:
$$s = \sqrt{s^2}$$
Expressed in the exact same physical units as the original data (e.g., years for `Age`, pounds for `Fare`).

### 4.4 Interquartile Range (IQR)
$$IQR = Q3 - Q1$$
The spread of the middle 50% of the distribution. Highly robust against outliers.

---

## 5. Distribution Shape: Skewness & Kurtosis

### 5.1 Skewness
Skewness measures the **asymmetry** of a probability distribution around its mean.

* **Symmetric ($Skew \approx 0$):**
  $$\text{Mean} \approx \text{Median} \approx \text{Mode}$$
* **Positive / Right-Skewed ($Skew > 0$):**
  $$\text{Mean} > \text{Median} > \text{Mode}$$
  The tail stretches far to the right (e.g., Titanic `Fare`, household wealth).
* **Negative / Left-Skewed ($Skew < 0$):**
  $$\text{Mode} > \text{Median} > \text{Mean}$$
  The tail stretches to the left (e.g., retirement age).

### 5.2 Kurtosis
Kurtosis measures the **heaviness of tails** and sharpness of the central peak relative to a normal distribution (Normal Kurtosis = 3.0, or Excess Kurtosis = 0).

* **Mesokurtic (Excess Kurtosis $\approx 0$):** Normal bell curve.
* **Leptokurtic (Excess Kurtosis $> 0$):** Heavy tails and sharp peak; high probability of extreme outlier events ("black swan" events).
* **Platykurtic (Excess Kurtosis $< 0$):** Light tails and flat peak; fewer extreme outliers.

---

## 6. Important Conceptual Comparisons

### Comparison 1: Histogram vs. KDE
* **Histogram:** Depends on bin size and edge choices. Gives exact integer counts. Best for knowing raw sample volume.
* **KDE:** Continuous and smooth. Eliminates arbitrary bin cutoffs. Best for comparing the underlying probability density shapes across multiple subsets.

### Comparison 2: Mean vs. Median
* In symmetric data (like height or standardized test scores), Mean and Median are virtually identical.
* In heavily skewed data (like Titanic `Fare`, where max is £512 but 75% of passengers paid $\le$ £31), the **Mean (£32.20) is pulled upward** by a few wealthy passengers, while the **Median (£14.45) accurately reflects what a typical passenger paid**.

---

## Summary Checklist
- [x] Analyze categorical distributions using frequency tables, count plots, and donut/Pareto charts.
- [x] Analyze numerical distributions using histograms, KDEs, box plots, and ECDFs.
- [x] Calculate the five-number summary and flag outliers using the $1.5 \times IQR$ rule.
- [x] Choose Median over Mean whenever data exhibits strong skewness ($Skew > 1$ or $Skew < -1$).
- [x] Calculate skewness and kurtosis to guide future data transformation decisions.
