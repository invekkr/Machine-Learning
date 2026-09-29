# Python Prerequisites: 02. NumPy for Machine Learning

NumPy (Numerical Python) is the foundational scientific computing library in Python. Almost every data science and machine learning library—including Pandas, Scikit-Learn, SciPy, PyTorch, and TensorFlow—is built on top of NumPy.

---

## 1. What is NumPy and Why Does Machine Learning Need It?

### The Problem with Standard Python Lists:
Python lists are flexible because they can store mixed data types (integers, strings, floats all in one list). However, this flexibility comes at a severe performance cost:
- Python lists store pointers (memory addresses) to individual Python objects scattered in RAM.
- Doing math on a list of 1,000,000 numbers requires a slow Python `for` loop that checks data types one by one.

### The NumPy Solution: The `ndarray`
NumPy introduces the **`ndarray` (N-Dimensional Array)**:
- Stores numbers of the **same data type** in **contiguous blocks of memory**.
- Implemented in C, running mathematical operations at raw hardware speed.
- Supports **Vectorization**: Performing math on entire collections of numbers simultaneously without writing `for` loops.

```python
import numpy as np

# A simple 1D NumPy array
arr = np.array([10, 20, 30])
print(arr)  # [10 20 30]
print(type(arr))  # <class 'numpy.ndarray'>
```

---

## 2. 1D vs. 2D Arrays in Machine Learning

In Machine Learning:
- A **1D Array** represents a single vector or target column ($y$).
- A **2D Array** represents a feature matrix ($X$), where rows are samples and columns are features.

```python
# 1D Array (Vector - e.g., Target Labels)
y = np.array([0, 1, 1, 0])

# 2D Array (Matrix - e.g., Feature Matrix with 3 rows and 2 columns)
X = np.array([[25, 50000], [30, 70000], [28, 45000]])
```

---

## 3. Essential Array Attributes: `ndim`, `shape`, `size`, `dtype`

Every NumPy array has 4 critical properties you must check constantly:

| Attribute | What It Tells You | Small Example (`X`) | Plain English Meaning |
| :--- | :--- | :--- | :--- |
| **`ndim`** | Number of array dimensions (axes) | `X.ndim` $\longrightarrow 2$ | It is a 2D table (rows and columns). |
| **`shape`** | Tuple of lengths along each dimension | `X.shape` $\longrightarrow (3, 2)$ | 3 rows (samples) and 2 columns (features). |
| **`size`** | Total number of elements | `X.size` $\longrightarrow 6$ | There are 6 total numbers stored. |
| **`dtype`** | Data type of the stored elements | `X.dtype` $\longrightarrow int64$ | All elements are 64-bit integers. |

```python
X = np.array([[10, 20], [30, 40], [50, 60]])

print("Dimensions:", X.ndim)  # 2
print("Shape:     ", X.shape)  # (3, 2)
print("Total size:", X.size)  # 6
print("Data type: ", X.dtype)  # int64
```

---

## 4. Array Indexing and Slicing

Slicing in 2D arrays follows the universal rule: **`array[rows, columns]`**.

```text
       Column 0   Column 1
Row 0: [  10,        20   ]
Row 1: [  30,        40   ]
Row 2: [  50,        60   ]
```

- **`X[0, 1]`**: Row 0, Column 1 $\longrightarrow$ `20`.
- **`X[:, 0]`**: All rows (`:`), Column 0 $\longrightarrow$ `[10, 30, 50]` *(extracting a single feature)*.
- **`X[0:2, :]`**: First two rows (0 and 1), all columns $\longrightarrow$ `[[10, 20], [30, 40]]`.

---

## 5. Reshaping Arrays (`reshape`)

In ML algorithms, you often need to change an array's dimensions (e.g. converting a flat 1D array of 4 numbers into a 2D column vector of shape `(4, 1)`):

```python
arr = np.array([10, 20, 30, 40])
print("Original shape:", arr.shape)  # (4,)

# Reshape into 2 rows and 2 columns
matrix = arr.reshape(2, 2)
print("Reshaped (2, 2):\\n", matrix)

# Reshape into a 2D column vector (crucial for Scikit-Learn single features)
col_vec = arr.reshape(-1, 1)
print("Column vector shape:", col_vec.shape)  # (4, 1)
```
*(The `-1` tells NumPy: "Calculate this dimension automatically so the total size matches.")*

---

## 6. Vectorized Arithmetic & Element-Wise Operations

In standard Python, adding 5 to every item in a list requires a loop: `[x + 5 for x in lst]`.  
In NumPy, you apply the math directly to the array:

```python
salaries = np.array([50000, 60000, 70000])

# Add 5,000 bonus to everyone
salaries_bonus = salaries + 5000
print(salaries_bonus)  # [55000 65000 75000]

# Element-wise operations between two arrays
actual = np.array([100, 200, 300])
predicted = np.array([90, 210, 290])

errors = actual - predicted
print("Errors:", errors)  # [ 10 -10  10]

squared_errors = errors**2
print("Squared Errors:", squared_errors)  # [100 100 100]
```

---

## 7. Statistical & Aggregation Functions

NumPy provides fast built-in mathematical summary functions:

```python
data = np.array([10, 20, 30, 40, 50])

print("Sum:               ", np.sum(data))  # 150
print("Mean (Average):    ", np.mean(data))  # 30.0
print("Median:            ", np.median(data))  # 30.0
print("Minimum:           ", np.min(data))  # 10
print("Maximum:           ", np.max(data))  # 50
print("Variance:          ", np.var(data))  # 200.0
print("Standard Deviation:", np.std(data))  # 14.14
```

### Aggregating Along Axes (`axis=0` vs `axis=1`):
In a 2D matrix:
- **`axis=0`**: Computes **down columns** (vertical) $\longrightarrow$ Calculates feature statistics.
- **`axis=1`**: Computes **across rows** (horizontal) $\longrightarrow$ Calculates per-sample statistics.

```python
X = np.array([[10, 100], [20, 200], [30, 300]])

# Feature means (Mean of Column 0, Mean of Column 1)
feature_means = np.mean(X, axis=0)
print("Column Means:", feature_means)  # [ 20. 200.]
```

---

## 8. Random Numbers & Reproducibility (`random_state`)

Machine learning algorithms use randomness (e.g. shuffling data, initializing neural network weights). To ensure your experiments produce identical results every time you run them, set a **random seed**:

```python
# Setting a seed guarantees reproducible results
np.random.seed(42)

# Generate 3 random floats between 0 and 1
random_weights = np.random.rand(3)
print("Weights:", np.round(random_weights, 3))  # [0.375 0.951 0.732]
```

---

## 9. Matrix Basics: The Dot Product

Linear models calculate predictions by multiplying input features by learned weights and summing them up:
$$\hat{y} = w_1 x_1 + w_2 x_2$$

In linear algebra, this is simply the **Dot Product** between two vectors:

```python
features = np.array([2.0, 3.0])  # [x1, x2]
weights = np.array([0.5, 1.5])  # [w1, w2]

# Dot product: (2.0 * 0.5) + (3.0 * 1.5) = 1.0 + 4.5 = 5.5
prediction = np.dot(features, weights)
print("Dot Product (Prediction):", prediction)  # 5.5

# Modern Python operator (@) does the exact same thing
print("Using @ operator:        ", features @ weights)  # 5.5
```

---

## 10. Why NumPy is Indispensable in Machine Learning

| ML Task | How NumPy Powers It |
| :--- | :--- |
| **Linear Regression** | Matrix multiplication solves the Normal Equation $\hat{\beta} = (X^T X)^{-1} X^T y$. |
| **Feature Scaling** | Centering ($x - \mu$) and scaling ($\div \sigma$) in one vectorized line. |
| **Neural Networks** | Forward pass is a series of matrix dot products: $Z = XW + b$. |
| **Loss Evaluation** | Calculating Mean Squared Error: `np.mean((y_true - y_pred)**2)`. |
| **PCA** | Computing covariance matrices and singular value decompositions. |
