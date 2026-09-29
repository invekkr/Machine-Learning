# Python Prerequisites: 05. Matplotlib & Seaborn for Machine Learning

Numbers and summary tables are useful, but human brains process visual patterns much faster than grids of text. Data visualization is essential during Exploratory Data Analysis (EDA) to detect outliers, understand feature distributions, and uncover relationships between variables.

*(Note: For advanced bivariate and multivariate visualization techniques, explore [02_EDA](file:///Users/shamvi/stuff/ML/02_EDA/). This module covers the foundational plotting tools.)*

---

## 1. What are Matplotlib and Seaborn?

- **Matplotlib (`matplotlib.pyplot`):** Python's foundational plotting library. It provides low-level control over every visual element (canvas size, axes, ticks, colors, labels).
- **Seaborn (`seaborn`):** A high-level visualization library built directly on top of Matplotlib. It integrates natively with Pandas DataFrames and provides beautiful, statistically oriented default themes.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean default theme
sns.set_theme(style='whitegrid')
```

---

## 2. Anatomy of a Plot

```text
 ┌───────────────────────────────────────────────────────────┐
 │ Figure (The Entire Canvas)                                │
 │                                                           │
 │     Title: Age vs. Salary Distribution                   │
 │                                                           │
 │  Y-Axis (Salary)                                          │
 │    ▲                                                      │
 │    │       *                                              │
 │    │          *     *   (Data Points)                     │
 │    │       *                                              │
 │    └─────────────────────────────► X-Axis (Age)          │
 │                                                           │
 └───────────────────────────────────────────────────────────┘
```

- **`Figure`**: The top-level window or canvas that holds everything.
- **`Axes`**: The specific plot area bounded by the X-axis and Y-axis where data points are drawn.

```python
# Standard object-oriented plotting structure
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_title("Sample Plot")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
plt.show()
```

---

## 3. The 6 Essential Plot Types in Machine Learning

### 1. Scatter Plot (Numerical vs. Numerical)
**Purpose:** Shows whether two continuous variables have a positive, negative, linear, or non-linear relationship.
```python
# Visualizing relationship between Age and Salary
plt.figure(figsize=(6, 4))
sns.scatterplot(x="Age", y="Salary", data=df, hue="Purchased")
plt.title("Age vs. Salary (Colored by Purchase)")
plt.show()
```

### 2. Line Plot (Trends Over Iterations or Time)
**Purpose:** Tracking model loss over training epochs or monitoring trends over time.
```python
epochs = [1, 2, 3, 4, 5]
loss = [0.85, 0.62, 0.45, 0.38, 0.35]

plt.plot(epochs, loss, marker="o", color="red")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()
```

### 3. Histogram & KDE (Distribution Shape of One Feature)
**Purpose:** Checking if a feature follows a normal bell curve, is skewed to one side, or has multi-modal peaks.
```python
# Histogram with Kernel Density Estimate curve
sns.histplot(df["Age"], kde=True, bins=10, color="teal")
plt.title("Distribution of Age")
plt.show()
```

### 4. Bar Plot (Categorical Counts or Means)
**Purpose:** Comparing values across distinct categories (e.g. survival rate across passenger classes).
```python
# Count plot of categories
sns.countplot(x="Gender", data=df, palette="pastel")
plt.title("Customer Count by Gender")
plt.show()
```

### 5. Box Plot (Quartiles, Spread, and Outliers)
**Purpose:** Visualizing the Five-Number Summary (Min, $Q_1$, Median, $Q_3$, Max) and detecting extreme outliers.
```python
sns.boxplot(x="Purchased", y="Salary", data=df)
plt.title("Salary Distribution by Purchase Status")
plt.show()
```

### 6. Heatmap (Correlation Matrix)
**Purpose:** Visualizing the linear correlation ($r$) between all numerical features simultaneously to spot multicollinearity.
```python
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

## 4. Connecting Plots to Model Decisions

| What You See in the Plot | What It Tells You About Preprocessing / Modeling |
| :--- | :--- |
| **Points lie along a straight line in Scatter Plot** | Linear Regression will likely perform well. |
| **Histogram is heavily right-skewed** | Might require a Logarithmic Transformation or Robust Scaling. |
| **Points isolated far above Box Plot whiskers** | Extreme outliers exist; consider RobustScaler or outlier pruning. |
| **Heatmap shows $r = 0.98$ between two features** | Severe multicollinearity; one feature is redundant and can be dropped. |
