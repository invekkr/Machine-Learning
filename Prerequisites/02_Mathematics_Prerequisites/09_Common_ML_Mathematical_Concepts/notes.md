# Mathematics Prerequisites: 09. Common Mathematical Concepts in Machine Learning

As you progress through your 100 Days of Machine Learning, you will notice certain mathematical equations reappearing across dozens of different algorithms. This guide serves as your universal reference for these recurring mathematical building blocks.

---

## 1. Distance Metrics: Measuring Similarity

In machine learning, "similarity" between data points is measured as geometric distance. If two customers have similar ages, salaries, and purchase histories, their points sit close together in feature space.

```text
               (x₂, y₂) = (4, 6)
                  ▲
                 /│
  Euclidean     / │ Vertical distance = |6 - 2| = 4
  (Hypotenuse) /  │
              /   │
             /____│
(x₁, y₁) = (1, 2)
   Horizontal distance = |4 - 1| = 3
```

### 1. Euclidean Distance ("As the Crow Flies")
The straight-line geometric distance between two points:
$$d_{\text{Euclidean}}(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{D} (p_i - q_i)^2} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

**Tiny Step-by-Step Calculation:**
Points: $P_1 = (1, 2)$ and $P_2 = (4, 6)$
1. Horizontal difference: $(4 - 1) = 3 \implies 3^2 = 9$
2. Vertical difference: $(6 - 2) = 4 \implies 4^2 = 16$
3. Sum and square root: $\sqrt{9 + 16} = \sqrt{25} = \mathbf{5.0}$

---

### 2. Manhattan Distance ("Taxicab Distance")
The distance a taxicab travels along grid-like city streets (sum of absolute horizontal and vertical lengths):
$$d_{\text{Manhattan}}(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{D} |p_i - q_i| = |x_2 - x_1| + |y_2 - y_1|$$

**Tiny Calculation:**
$$|4 - 1| + |6 - 2| = 3 + 4 = \mathbf{7.0}$$

**ML Algorithms Using Distance:**
- **K-Nearest Neighbors (KNN):** Finds the $K$ closest training points to classify a new point.
- **K-Means Clustering:** Assigns points to the nearest cluster centroid.
- **Support Vector Machines (SVM):** Measures margins between support vectors and decision boundaries.

---

## 2. Regression Error Metrics: MAE, MSE, and RMSE

In regression problems, a model predicts a continuous number $\hat{y}$ (e.g. house price).  
The **Residual Error** is:
$$e = y - \hat{y} \quad (\text{Actual} - \text{Predicted})$$

Let's test three house predictions:
- **Actual Prices ($y$):** $[10, 20, 30]$
- **Predicted Prices ($\hat{y}$):** $[12, 18, 31]$
- **Residual Errors ($y - \hat{y}$):** $[10 - 12, 20 - 18, 30 - 31] = [-2, +2, -1]$

---

### 1. Mean Absolute Error (MAE)
Takes the average of the absolute errors (linear penalty):
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i| = \frac{|-2| + |+2| + |-1|}{3} = \frac{2 + 2 + 1}{3} = \frac{5}{3} \approx \mathbf{1.67}$$

---

### 2. Mean Squared Error (MSE)
Squares each error before averaging (heavily punishes large mistakes):
$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \frac{(-2)^2 + (+2)^2 + (-1)^2}{3} = \frac{4 + 4 + 1}{3} = \frac{9}{3} = \mathbf{3.00}$$

---

### 3. Root Mean Squared Error (RMSE)
Takes the square root of MSE to return the error back to the original units:
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{3.00} \approx \mathbf{1.73}$$

---

## 3. The Sigmoid Function ($\sigma$)

In classification, we want our model to output a **clean probability between $0.0$ and $1.0$**.  
A linear equation ($w \cdot x + b$) can produce any number from $-\infty$ to $+\infty$. How do we squash that into $[0, 1]$?

We pass it through the **Sigmoid Function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```text
       Probability P(y=1)
            ▲
        1.0 │                 . - - - - - - (Approaches 1.0 as z -> +inf)
            │               /
        0.5 │..............o (At z = 0, P = 0.50!)
            │             /
        0.0 │ - - - - - -' (Approaches 0.0 as z -> -inf)
            └─────────────────────────────► Raw Linear Output (z)
              -6  -4  -2   0   2   4   6
```

### Tiny Calculations:
- When $z = 0$: $\sigma(0) = \frac{1}{1 + e^{-0}} = \frac{1}{1 + 1} = \frac{1}{2} = \mathbf{0.50}$ *(The exact decision threshold!)*
- When $z = +2$: $\sigma(2) = \frac{1}{1 + e^{-2}} \approx \frac{1}{1 + 0.135} \approx \mathbf{0.88}$ *(High confidence)*
- When $z = -2$: $\sigma(-2) = \frac{1}{1 + e^{2}} \approx \frac{1}{1 + 7.389} \approx \mathbf{0.12}$ *(Low confidence)*

**ML Context:** This is the core mathematical engine of **Logistic Regression** and Neural Network activations.

---

## 4. Entropy and Information Gain (Decision Trees)

In Decision Trees, the algorithm must choose which feature to split on at each step. How does it know which feature is best?  
It chooses the feature that **reduces uncertainty the most**.

### 1. Entropy ($H$) — "Measure of Impurity"
Entropy measures the chaos or uncertainty in a group of labels:
$$H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
- **Pure Group (All Yes, 0 No):** Entropy is **$0.0$** (Zero uncertainty!).
- **Completely Mixed Group (50% Yes, 50% No):** Entropy is **$1.0$** (Maximum chaos!).

### 2. Information Gain — "Reduction in Chaos"
$$\text{Information Gain} = \text{Entropy before split} - \text{Weighted Entropy after split}$$
The feature with the **highest Information Gain** becomes the next decision branch!

---

## 5. Regularization: $L_1$ (Lasso) vs. $L_2$ (Ridge)

When models have too many features, they risk **overfitting** (memorizing training noise).  
We add a penalty to the loss function to keep weights small:

$$\text{Total Cost} = \text{MSE} + \text{Penalty}$$

```text
L1 Regularization (Lasso)                   L2 Regularization (Ridge)
Penalty = λ * sum(|w_i|)                    Penalty = λ * sum(w_i²)
- Uses ABSOLUTE values                      - Uses SQUARED values
- Drives uninformative weights              - Shrinks weights smoothly
  EXACTLY TO ZERO                           - Solves multicollinearity without
- Acts as automated feature selection!        discarding features!
```

---

## 6. Python Verification

```python
import math
import numpy as np


# 1. Euclidean vs Manhattan Distance
def euclidean(p1, p2):
  return np.sqrt(np.sum((p1 - p2) ** 2))


def manhattan(p1, p2):
  return np.sum(np.abs(p1 - p2))


p1 = np.array([1, 2])
p2 = np.array([4, 6])
print("Euclidean Distance:", euclidean(p1, p2))  # 5.0
print("Manhattan Distance:", manhattan(p1, p2))  # 7.0


# 2. Sigmoid Function
def sigmoid(z):
  return 1 / (1 + np.exp(-z))


print("\nSigmoid(0) :", sigmoid(0))  # 0.50
print("Sigmoid(+2):", round(sigmoid(2), 2))  # 0.88
print("Sigmoid(-2):", round(sigmoid(-2), 2))  # 0.12
```
