# Mathematics Prerequisites: 06. Linear Algebra for Machine Learning

Linear Algebra is the mathematical language of multi-dimensional data. Modern datasets contain millions of rows and hundreds of features. Without Linear Algebra, a computer would have to loop through numbers one by one. With Linear Algebra, entire datasets are stored compactly as matrices, and transformations happen in lightning-fast parallel operations.

---

## 1. Scalars, Vectors, and Matrices

```text
Scalar (0D)              Vector (1D)                 Matrix (2D)
  [ 5 ]                  ┌───┐                      ┌───────┬───────┐
  Single                 │ 2 │                      │ 1   2 │
  number                 │ 3 │                      │ 3   4 │
                         └───┘                      └───────┴───────┘
                     List of numbers              Grid of numbers
                     (Magnitude & Direction)      (Rows & Columns)
```

- **Scalar ($c$):** A single plain number (e.g. $c = 5$).
- **Vector ($\mathbf{v}$):** An ordered list of numbers. In ML, a vector represents either:
  - A single row with multiple features: $\mathbf{x} = [\text{Age}, \text{Salary}]$.
  - A model's weight vector: $\mathbf{w} = [w_1, w_2]$.
- **Matrix ($A$ or $X$):** A two-dimensional rectangular grid of numbers organized into rows and columns. In ML, the **Feature Matrix $X$** has $N$ rows (samples) and $D$ columns (features).

---

## 2. Vector Operations

### 1. Vector Addition
Add corresponding components:
$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 1 + 3 \\ 2 + 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

### 2. Scalar Multiplication
Multiply every component by the scalar:
$$2 \times \begin{bmatrix} 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 2 \times 3 \\ 2 \times 5 \end{bmatrix} = \begin{bmatrix} 6 \\ 10 \end{bmatrix}$$

### 3. Vector Dot Product (The Weighted Sum)
Multiply matching elements and sum the results:
$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n$$

**Tiny Example:**
$$\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$$
$$\mathbf{x} \cdot \mathbf{w} = (2 \times 4) + (3 \times 5) = 8 + 15 = \mathbf{23}$$

**Why ML Uses It:** Linear Regression predictions are simply the dot product between the feature vector and the weight vector: $\hat{y} = \mathbf{x} \cdot \mathbf{w} + b$.

### 4. Vector Magnitude (Norm / Length) — $\|\mathbf{v}\|$
How long is the vector in space?
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2}$$
For $\mathbf{v} = [3, 4]$:
$$\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5}$$

---

## 3. Matrix Operations

### 1. Matrix Dimensions
A matrix with $M$ rows and $N$ columns is an **$M \times N$ matrix**.
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad \text{Shape: } 3 \times 2$$

### 2. Matrix Multiplication ($A \times B$)
To multiply two matrices, **the inner dimensions must match**:
$$(M \times K) \times (K \times N) \longrightarrow \text{Resulting Shape: } (M \times N)$$

**Tiny Step-by-Step Example ($2 \times 2$ matrix multiplied by $2 \times 1$ column vector):**
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

- **Row 0 dot product:** $(1 \times 5) + (2 \times 6) = 5 + 12 = 17$
- **Row 1 dot product:** $(3 \times 5) + (4 \times 6) = 15 + 24 = 39$

$$A \mathbf{x} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 3. Matrix Transpose ($A^T$)
Swaps rows and columns (Row 0 becomes Column 0, Row 1 becomes Column 1):
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix}$$

---

### 4. Identity Matrix ($I$)
A square matrix with $1$s on the main diagonal and $0$s everywhere else:
$$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$
It acts like the number **1** in arithmetic: multiplying any matrix by $I$ leaves it completely unchanged ($A \cdot I = A$).

---

### 5. Matrix Inverse ($A^{-1}$)
In regular arithmetic, dividing by $5$ is the same as multiplying by $\frac{1}{5}$ (its reciprocal: $5 \times 5^{-1} = 1$).  
In linear algebra, you cannot "divide" by a matrix. Instead, you multiply by its **Inverse ($A^{-1}$)**:
$$A \cdot A^{-1} = I$$

**ML Context:** In Ordinary Least Squares (OLS) Linear Regression, the weights that minimize squared errors are calculated analytically via the Normal Equation:
$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

---

## 4. Covariance Matrix

A **Covariance Matrix** summarizes how multiple features vary together:

```text
                  Feature 1 (Age)     Feature 2 (Salary)
Feature 1 (Age)   [ Variance(Age)      Covariance(Age, Salary) ]
Feature 2 (Salary)[ Covariance(Salary, Age)   Variance(Salary) ]
```
- **Main Diagonal:** Contains the variance of each individual feature.
- **Off-Diagonals:** Contains the covariance between pairs of features.
- If covariance is positive: As Age increases, Salary tends to increase.
- If covariance is near zero: Age and Salary have no linear relationship.

---

## 5. Eigenvectors and Eigenvalues (Demystified!)

Do not let the German word *"eigen"* (meaning "characteristic" or "own") intimidate you. The visual intuition is simple.

### Intuition: Transformation Without Rotation
When you multiply a vector $\mathbf{v}$ by a matrix $A$, the matrix usually does two things:
1. It **rotates** the vector into a new direction.
2. It **stretches or shrinks** its length.

```text
Most Vectors:             Rotates and changes direction
   v ──► [ Matrix A ] ──► Av (Points in a totally different direction!)
```

However, for any given matrix, there are a few **very special directions** where the vector **DOES NOT ROTATE AT ALL!**  
The vector stays pointing in the exact same line—it is **only stretched or squashed by a scalar factor**!

```text
Special Eigenvectors:     STAYS on the exact same axis! Only stretched!
   v ──► [ Matrix A ] ──► Av = λv
```

### The Famous Equation:
$$A \mathbf{v} = \lambda \mathbf{v}$$

### Symbol Breakdown:
- **$A$**: The square transformation matrix (e.g. the Covariance Matrix of our data).
- **$\mathbf{v}$**: The **Eigenvector** (the special direction that does not rotate).
- **$\lambda$**: The **Eigenvalue** (the Greek letter *lambda*, representing the stretch/scaling factor).

### Why ML Uses Eigenvectors & Eigenvalues: Principal Component Analysis (PCA)
In high-dimensional datasets with 100 features, many features are redundant.  
How do we find the most important axes of information?
1. We compute the **Covariance Matrix** $A$ of our features.
2. We find its **Eigenvectors ($\mathbf{v}$)**: These vectors point in the **exact directions where data has the greatest variance / spread**.
3. We check their **Eigenvalues ($\lambda$)**: The eigenvalue tells us **how much information (variance) is captured** along that direction!
4. We keep the top 2 or 3 eigenvectors with the largest eigenvalues, compressing 100 features down to 2 dimensions with minimal information loss!

---

## 6. Python Verification

```python
import numpy as np

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
print("A @ x =", A @ x)  # [17, 39]

# Eigenvalues and Eigenvectors
cov_matrix = np.array([[4, 2], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:  ", np.round(eigenvalues, 2))
print("Eigenvectors:\n", np.round(eigenvectors, 2))
```
