# Module 02: Exploratory Data Analysis — Bivariate Analysis

**Bivariate Analysis** investigates the statistical relationship between **two variables** simultaneously.

While univariate analysis tells us about individual distributions, bivariate analysis answers questions like:
- Does variable $X$ influence or relate to variable $Y$?
- Is the relationship linear, non-linear, or non-existent?
- How does the distribution of a numerical metric change across different categorical groups?
- Are two categorical classifications statistically independent?

We categorize bivariate analysis into three foundational combinations:
1. **Numerical + Numerical**
2. **Numerical + Categorical**
3. **Categorical + Categorical**

---

## 1. Numerical + Numerical Analysis

When both variables are continuous or discrete numbers, we examine how changes in one correspond to changes in the other.

### 1.1 Scatter Plot (`sns.scatterplot`)
A **Scatter Plot** plots points on a Cartesian plane where the X-coordinate represents feature $X$ and the Y-coordinate represents feature $Y$.

#### Types of Relationships:
1. **Positive Relationship:** As $X$ increases, $Y$ tends to increase (e.g., `Age` and ticket `Fare` in 1st class).
2. **Negative Relationship:** As $X$ increases, $Y$ tends to decrease (e.g., `Pclass` number and ticket `Fare`).
3. **No Obvious / Zero Relationship:** Points are randomly dispersed with no discernible slope (e.g., `PassengerId` and `Age`).
4. **Non-linear Relationship:** Curved relationships (U-shape, exponential, logarithmic) that standard linear correlation ($r$) fails to capture.

```python
import seaborn as sns

sns.scatterplot(data=df, x="Age", y="Fare")
```

### 1.2 Line Plot (`sns.lineplot`)
A **Line Plot** connects points in sequential order along the X-axis.
- Used when X has a natural ordering (time series, age increments, rank).
- Seaborn's `lineplot` automatically aggregates multiple observations per X-value and displays a **95% confidence interval** band.

```python
# Mean Fare across Age brackets with confidence interval
sns.lineplot(data=df, x="Age", y="Fare")
```

### 1.3 Pair Plot (`sns.pairplot`)
A **Pair Plot** creates a grid of pairwise bivariate scatter plots for all numerical features in the dataset, with univariate histograms or KDEs along the diagonal.
- Best for an exhaustive survey of multi-feature relationships in one glance.

---

## 2. Numerical + Categorical Analysis

In machine learning, this is the most common form of bivariate analysis because your target is often categorical (e.g., `Survived = 0/1`, `Churn = Yes/No`) and your features are numerical (`Age`, `Fare`).

### 2.1 Bar Plot (`sns.barplot`)
A bar plot calculates an aggregate statistic (by default, the **mean**) of the numerical feature for each category, along with a vertical error bar (representing the 95% confidence interval of the mean via bootstrapping).

```python
# Average Fare paid by Survival status
sns.barplot(data=df, x="Survived", y="Fare")
```

### 2.2 Box Plot by Category (`sns.boxplot`)
Displays the full five-number summary (median, IQR, min/max whiskers, outliers) for the numerical variable **split across categories**.

```python
# Age distribution across Passenger Classes
sns.boxplot(data=df, x="Pclass", y="Age")
```

### 2.3 Grouped KDE Plot (`sns.kdeplot(hue=...)`)
Superimposes the continuous density distribution of the numerical variable for each categorical level.

```python
# Age distribution density comparison by Survival status
sns.kdeplot(data=df, x="Age", hue="Survived", common_norm=False)
```

---

## 3. Critical Comparison: Bar Plot vs. Box Plot

Data science learners frequently confuse Bar Plots and Box Plots when comparing groups. Here is the crucial difference:

| Dimension | Bar Plot (`sns.barplot`) | Box Plot (`sns.boxplot`) |
| :--- | :--- | :--- |
| **What it shows** | A single point summary (usually the **mean**) plus confidence interval. | The **entire distribution**: median, quartiles (Q1, Q3), spread, and outliers. |
| **Skewness Sensitivity** | Highly misleading if data is heavily skewed or contains extreme outliers. | Resistant; median and IQR remain accurate regardless of outliers. |
| **Outlier Visibility** | Completely hides outliers. | Explicitly displays individual outlier points. |
| **Best Used When** | Presenting high-level expected value benchmarks to executive stakeholders. | Performing rigorous scientific EDA, data debugging, and checking distributional assumptions. |

> **Example on Titanic `Fare`:**
> A bar plot shows 1st class passengers paid an average of £84. But a box plot reveals that while the 1st class median was £60, a few luxury suites cost over £500, severely inflating the bar plot!

---

## 4. Categorical + Categorical Analysis

When both variables are categorical (e.g., `Pclass` and `Survived`, or `Sex` and `Embarked`), scatter plots cannot be used because points would stack on identical discrete grid coordinates.

### 4.1 Contingency Tables (`pd.crosstab`)
A two-way table that tabulates the joint frequency distribution of two categorical features.

```python
# Raw joint counts
pd.crosstab(df["Pclass"], df["Survived"])
```

### 4.2 Why Percentages Are More Meaningful Than Raw Counts
Looking only at raw counts can be dangerously misleading when group sizes are unequal!

* In Titanic 3rd class: **119 survived**, while **372 died**.
* In 1st class: **136 survived**, while **80 died**.

Notice that the raw count of survivors is roughly similar (119 vs 136).
However, when you normalize by row (**conditional percentage**):
$$\text{Survival Rate (1st Class)} = \frac{136}{136 + 80} = 62.96\%$$
$$\text{Survival Rate (3rd Class)} = \frac{119}{119 + 372} = 24.24\%$$

Calculating row-wise percentages reveals the true reality: a 1st class passenger had **over 2.5 times the probability of surviving** compared to a 3rd class passenger!

### 4.3 Crosstab Normalization Options
- `normalize='index'`: Row percentages (each row sums to 100%).
- `normalize='columns'`: Column percentages (each column sums to 100%).
- `normalize='all'`: Overall dataset percentages (entire table sums to 100%).

```python
# Survival percentage within each ticket class
pd.crosstab(df["Pclass"], df["Survived"], normalize="index") * 100
```

### 4.4 Crosstab Heatmaps
Visualizing the cross-tabulation using a color-encoded heatmap allows immediate pattern recognition.
```python
ct = pd.crosstab(df["Sex"], df["Survived"], normalize="index") * 100
sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues")
```

### 4.5 ClusterMap (`sns.clustermap`)
A **ClusterMap** performs **hierarchical clustering** on both rows and columns of a cross-tabulated matrix and displays dendrogram trees along the borders.
- Groups similar categories together based on Euclidean distance and correlation.
- Invaluable for high-cardinality categorical pairs (e.g., Customer Segment vs Product Category).

---

## Summary Checklist
- [x] Use scatter plots to detect linear, non-linear, or non-existent relationships between two numerical features.
- [x] Prefer box plots over bar plots when comparing numerical metrics across categories to see full distributions and outliers.
- [x] Use `pd.crosstab()` for two categorical variables, and always compute row/column percentages (`normalize='index'`) to prevent unequal group size distortion.
- [x] Visualize crosstabs with annotated heatmaps and use clustermaps to discover hierarchical groupings.
