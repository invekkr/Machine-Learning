# Module 02: Exploratory Data Analysis — Pandas Profiling (YData Profiling)

In modern machine learning and data engineering, performing manual exploratory data analysis for every basic statistic can be time-consuming.

**Pandas Profiling** (now maintained and rebranded as **`ydata-profiling`**) is an open-source library that automates standard Exploratory Data Analysis (EDA) tasks. With just a few lines of Python, it produces a comprehensive, interactive HTML report covering distributions, correlations, missing values, and dataset alerts.

---

## 1. What is Pandas Profiling / YData Profiling?

`ydata-profiling` generates interactive reports from Pandas DataFrames.
Instead of manually typing `df.describe()`, `df.isnull().sum()`, `df.corr()`, and plotting dozens of individual histograms, the library computes these metrics in a unified pass and compiles them into a self-contained HTML file.

### Installation
```bash
pip install ydata-profiling
```

### Basic Syntax
```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("data/titanic.csv")

# Generate the profile object
profile = ProfileReport(
    df, title="Titanic Dataset Comprehensive Report", explorative=True
)

# Export the report as an interactive HTML document
profile.to_file("outputs/reports/titanic_profiling_report.html")
```

---

## 2. Why Automated Profiling is Useful

1. **Rapid Baseline Triage:** In less than 60 seconds, you can identify high-level data quality issues across hundreds of columns.
2. **Standardized Reporting:** Creates an identical, reproducible diagnostic report that can be shared with data engineers, product managers, or domain stakeholders who do not write Python.
3. **Automated Alert Generation:** Flags anomalies (e.g., columns with 90% missing values, constant features, high cardinality, or severe skewness) before they corrupt model training pipelines.

---

## 3. Deep Dive: Key Sections of the Profiling Report

When you open the generated HTML report, it is divided into distinct analytical tabs:

### 3.1 Overview
* **Dataset Statistics:** Number of variables, number of observations, missing cells count and percentage, total size in memory, average record size.
* **Variable Types:** Breakdown of features categorized as Numeric, Categorical, Boolean, Date, or Unsupported.
* **Alerts / Warnings:** High-priority items requiring engineer attention.

### 3.2 Variables
Provides an interactive expandable card for every single feature in the DataFrame:
* **Quantile Statistics:** Min, Q1, Median, Q3, Max, IQR, Range.
* **Descriptive Statistics:** Mean, Standard Deviation, Variance, Skewness, Kurtosis, Mean Absolute Deviation (MAD).
* **Missing & Zeroes:** Exact counts and percentages of missing (`NaN`) and zero (`0`) values.
* **Histogram / Frequency Table:** Visual distribution of the feature.

### 3.3 Interactions
* Allows interactive bivariate scatter and 2D hexagonal binning plots for any pair of numerical variables selected via dropdowns.

### 3.4 Correlations
Unlike basic Pandas which only computes Pearson's linear correlation, YData Profiling computes multiple correlation coefficients:
* **Pearson ($r$):** Evaluates linear relationships.
* **Spearman ($\rho$):** Rank-based; evaluates monotonic relationships (even if non-linear).
* **Kendall ($\tau$):** Rank concordance; more robust against small sample noise.
* **Phik ($\phi_K$):** A modern correlation coefficient based on refined Pearson's $\chi^2$ test that captures non-linear relationships between both numerical AND categorical variables.
* **Cramér's V:** Association between categorical features.

### 3.5 Missing Values
Visualizes missing data patterns using multiple diagnostic representations:
* **Count / Bar Chart:** Total non-null counts per column.
* **Matrix Plot:** Nullity pattern across row indices (helps identify whether missingness occurs in contiguous chunks or random intervals).
* **Dendrogram:** Hierarchical clustering of columns showing which features tend to be missing together.

---

## 4. Understanding Dataset Warnings & Alerts

The **Alerts** tab is arguably the most valuable part of automated profiling. Common warnings include:

| Warning Type | Meaning | How to Handle It in ML |
| :--- | :--- | :--- |
| **High Cardinality** | Distinct count is very high (e.g., `Name`, `Ticket` have 600+ unique strings). | Do not One-Hot Encode (would cause dimensionality explosion). Extract substrings/titles or drop. |
| **High Correlation (Collinearity)** | Two features have $r > 0.90$ or high Phik association. | Multicollinearity inflates variance in linear models. Consider dropping one of the correlated pair. |
| **Missing Values** | Feature has a significant proportion of NaNs (e.g., `Cabin` has 77% missing). | Determine whether to drop, impute, or create a binary `IsMissing` indicator. |
| **Zeros** | Feature has a large percentage of 0 values (e.g., `Parch` has 76% zeros). | Verify if zero is a valid measurement (0 parents/children) or an unrecorded missing placeholder. |
| **Uniform / Constant** | Feature contains only a single unique value across all rows. | Zero variance; provides zero predictive signal. Safely drop from the dataset. |

---

## 5. Critical Insight: Why Automated EDA Does NOT Replace Understanding Your Data

While automated profiling tools are incredible accelerators, **they do not and cannot replace human exploratory data analysis**.

### 1. Lack of Domain Context & Business Logic
* The profiler reports that `Fare` has an upper fence at £66 and flags fares > £66 as outliers.
* A human with domain context knows that Titanic 1st class parlor suites genuinely cost up to £512. Deleting them as "bad data" would destroy critical predictive information about the wealthiest passengers.

### 2. Blindness to Target Leakage
* An automated tool might celebrate a feature that has a 0.99 correlation with your target.
* A human data scientist recognizes that this feature was recorded *after* the target event occurred (e.g., `HospitalDischargeDate` predicting whether a patient was admitted), which is catastrophic **data leakage**.

### 3. Blindness to Causality & Confounding
* The profiler detects correlations; it cannot discern cause, effect, or confounding variables (e.g., `Fare` correlating with `Survived` because of deck location and lifeboat protocol).

### 4. Problem Framing & Feature Engineering
* Automated profiling will never suggest combining `SibSp` + `Parch` + 1 into a `FamilySize` feature, or extracting social titles (`"Mr"`, `"Mrs"`, `"Master"`, `"Dr"`) from passenger names.

> **Conclusion:**
> Use automated profiling as your **first 10-minute diagnostic audit**, not your final understanding. Manual EDA is where deep intuition, feature engineering ideas, and modeling strategies are born.

---

## Summary Checklist
- [x] Use `ydata_profiling.ProfileReport` to generate automated HTML reports for rapid data auditing.
- [x] Inspect the Overview and Alerts tabs first to catch high cardinality, collinearity, and missingness.
- [x] Check multiple correlation metrics (Pearson, Spearman, and Phik) to detect non-linear dependencies.
- [x] Never rely solely on automated EDA: validate alerts against domain logic and watch out for target leakage.
