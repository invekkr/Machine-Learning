# Module 01: Data Handling — Working with Files

In machine learning and data science, data is rarely handed to you as a clean Python data structure. The vast majority of tabular datasets are stored in flat text files—most commonly **CSV (Comma-Separated Values)**.

Understanding how to read, inspect, and export flat files reliably is the foundational first step of any ML workflow.

---

## 1. What is a CSV File?

A **CSV** file is a plain-text tabular file where:
* Each line represents one **record** (row).
* Within each line, fields (features/columns) are separated by a delimiter, most commonly a comma (`,`).
* The first line is typically the **header**, specifying the column names.

### Example Raw CSV Snippet:
```text
PassengerId,Survived,Pclass,Name,Sex,Age
1,0,3,"Braund, Mr. Owen Harris",male,22
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38
```

> **Why CSV is ubiquitous in ML:**
> - Universal readability across operating systems and programming languages (Python, R, SQL, Excel).
> - Human-readable and lightweight for small to medium datasets (< a few GBs).
> - Easy to version-control and stream row-by-row.

---

## 2. Reading CSV Files with `pd.read_csv()`

In Python, the Pandas library provides `pd.read_csv()` to parse flat files into a DataFrame in memory.

### Basic Syntax
```python
import pandas as pd

df = pd.read_csv("data/titanic.csv")
```

### Essential Parameters You Must Know

| Parameter | Purpose | When to Use It |
| :--- | :--- | :--- |
| `filepath_or_buffer` | Path or URL to the file | Always required |
| `sep` or `delimiter` | Character separating values (default `,`) | When handling TSV (tab-separated, `\t`) or semicolon-separated (`;`) files |
| `usecols` | List of column names or indices to load | Working with massive datasets where you only need a subset of features to save RAM |
| `nrows` | Number of rows to read from the top | Quick prototyping or checking file format on gigabyte-sized files |
| `index_col` | Column to use as the row index | When your CSV already contains an index column (e.g., `index_col='PassengerId'`) |
| `na_values` | Strings to interpret as NaN | When missing values are coded as `?`, `None`, `-999`, or `N/A` |
| `dtype` | Dictionary specifying data types | To prevent Pandas from guessing types wrong (e.g., preserving leading zeros in ZIP codes) |

### Practical Example: Loading Specific Columns with Memory Optimization
```python
# Load only specific features to save memory
cols_to_keep = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "Fare"]
df_subset = pd.read_csv("data/titanic.csv", usecols=cols_to_keep)
```

---

## 3. Writing CSV Files with `df.to_csv()`

After cleaning, filtering, or engineering features, you will often export your processed DataFrame to disk.

### The Most Important Rule: `index=False`
By default, Pandas writes the row index (0, 1, 2, 3...) as the first column in the CSV file. If you re-read that file later, Pandas creates an unwanted column named `Unnamed: 0`.

```python
# BAD: Saves the row index as an extra unnamed column
df.to_csv("outputs/processed_data.csv")

# GOOD: Excludes the auto-generated integer index
df.to_csv("outputs/processed_data.csv", index=False)
```

### Useful Export Options
```python
# Export without index and format missing values as 'NA'
df.to_csv("outputs/clean_titanic.csv", index=False, na_rep="NA")
```

---

## 4. Initial Visual Inspection: `head()`, `tail()`, and `sample()`

Before computing any statistics or running machine learning algorithms, always inspect the raw rows of your DataFrame to build intuition about the data.

### 1. `df.head(n=5)`
* **What it does:** Returns the first `n` rows (defaults to 5).
* **Why use it:** Verifies that column headers were parsed correctly, checks for leading metadata artifacts, and confirms column data types look reasonable.

### 2. `df.tail(n=5)`
* **What it does:** Returns the last `n` rows.
* **Why use it:** Checks for summary rows, total rows, or corrupted trailing lines that sometimes get appended to CSV exports.

### 3. `df.sample(n=5, random_state=42)`
* **What it does:** Returns a random selection of `n` rows.
* **Why use it:** Avoids **ordering bias**. In many real-world datasets, rows are pre-sorted by date, label, class, or user ID. Inspecting only the `head()` can give a false impression of diversity.
* **Why `random_state` matters:** Setting `random_state` (seed) ensures **reproducibility**—your team or graders will see the exact same random rows every time.

```python
# Unbiased random sample of 5 rows
df.sample(n=5, random_state=42)
```

---

## Summary Checklist
- [x] Use `pd.read_csv()` to parse flat files into DataFrames.
- [x] Optimize memory on large files using `usecols` and test schemas with `nrows`.
- [x] Always pass `index=False` in `df.to_csv()` unless your index carries meaningful domain identity.
- [x] Never rely solely on `head()`; use `tail()` and `sample(random_state=...)` to check for edge cases and ordering bias.
