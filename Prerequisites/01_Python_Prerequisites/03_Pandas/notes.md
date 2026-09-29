# Python Prerequisites: 03. Pandas for Machine Learning

While NumPy provides the raw numerical engine, **Pandas** is the tool data scientists use to load, inspect, clean, filter, and prepare real-world tabular data.

*(Note: For deep-dive practical lessons, explore [01_Data_Handling](file:///Users/shamvi/stuff/ML/01_Data_Handling/) and [02_EDA](file:///Users/shamvi/stuff/ML/02_EDA/). This prerequisite module introduces the fundamental core syntax you need to get started.)*

---

## 1. What is Pandas and Why Do We Need It?

In the real world, datasets arrive as spreadsheets, SQL tables, or `.csv` files containing a mixture of column names, dates, text categories, numbers, and missing values.

NumPy arrays require all elements to have the same data type and lack named column headers.  
**Pandas** solves this by providing labeled, flexible table structures with powerful data cleaning tools.

---

## 2. The Two Core Pandas Data Structures

Pandas is built on two primary objects:

```text
1. Series (1D Labeled Column)          2. DataFrame (2D Labeled Table)
      Index    Value                          Index    Name    Age   Salary
        0      50000                            0      Alice   25    50000
        1      60000                            1      Bob     30    70000
        2      75000                            2      Charlie 28    60000
```

- **`Series`**: A single column of data with an associated index label.
- **`DataFrame`**: A 2-dimensional table made up of multiple Series sharing the same index.

```python
import pandas as pd

# Creating a Series
ages = pd.Series([25, 30, 28], name="Age")
print(ages)

# Creating a DataFrame from a dictionary
df = pd.DataFrame(
    {
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 28],
        "Salary": [50000, 70000, 60000],
    }
)
display(df)
```

---

## 3. Reading Data from CSV Files

The most common entry point for data science is loading a comma-separated values (`.csv`) file:

```python
# Reading a CSV file into a DataFrame
df = pd.read_csv('data/titanic.csv')

# Quick inspection methods
print('Shape (rows, columns):', df.shape)  # e.g., (891, 12)
display(df.head(3))  # First 3 rows
display(df.tail(2))  # Last 2 rows
```

---

## 4. Selecting Columns and Rows

### Selecting Columns:
```python
# Selecting a single column returns a Series
age_col = df["Age"]

# Selecting multiple columns (pass a LIST of names) returns a DataFrame
features = df[["Age", "Salary"]]
```

### Selecting Rows:
- **`iloc` (Integer Location):** Slices by position index (like standard Python lists).
- **`loc` (Label Location):** Slices by row label or boolean conditions.

```python
# First row by integer index
first_row = df.iloc[0]

# First 2 rows and first 2 columns
subset = df.iloc[0:2, 0:2]
```

---

## 5. Boolean Filtering (Querying Data)

Filtering rows that meet specific business conditions:

```python
# Find all employees older than 26
older_than_26 = df[df["Age"] > 26]
display(older_than_26)

# Multiple conditions (use & for AND, | for OR, and wrap in parentheses)
high_earning_seniors = df[(df["Age"] > 26) & (df["Salary"] >= 65000)]
```

---

## 6. Handling Missing Values (`NaN`)

Real-world datasets often have missing entries, represented in Pandas as `NaN` (Not a Number).

```python
# Check count of missing values per column
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Option 1: Drop rows with missing values
df_clean = df.dropna()

# Option 2: Fill missing values with a statistic (e.g. median age)
df["Age"] = df["Age"].fillna(df["Age"].median())
```

---

## 7. Descriptive Statistics and Aggregations

Pandas provides instant statistical summaries of numerical features:

```python
# High-level statistical breakdown (count, mean, std, min, 25%, 50%, 75%, max)
display(df.describe())

# Individual statistics
print("Average Salary:", df["Salary"].mean())
print("Median Age:    ", df["Salary"].median())
```

### Categorical Frequency Counts:
```python
# Count occurrences of each category
gender_distribution = df["Gender"].value_counts()
print(gender_distribution)
```

---

## 8. Grouping and Aggregating (`groupby`)

The split-apply-combine pattern allows you to compute statistics grouped by category (e.g., average salary per department or city):

```python
# Average salary by city
avg_salary_by_city = df.groupby("City")["Salary"].mean()
print(avg_salary_by_city)
```

---

## 9. How Pandas Connects to Machine Learning

In standard Scikit-Learn workflows:
1. You use **Pandas** to load the dataset: `df = pd.read_csv(...)`.
2. You use **Pandas** to separate feature columns ($X$) and target column ($y$):
   ```python
   X = df.drop(columns=["Purchased"])  # DataFrame (Features)
   y = df["Purchased"]  # Series (Target)
```
3. You pass $X$ and $y$ directly into Scikit-Learn pipelines and models.
