# Mathematics Prerequisites: 01. Basic Mathematics for Machine Learning

Machine Learning algorithms are built on top of high school arithmetic and algebra. If you understand basic operations like exponents, fractions, square roots, and logarithms, you have all the tools needed to understand how algorithms calculate errors, evaluate probabilities, and update weights.

---

## 1. Numbers, Fractions, and Negative Numbers

### Positive and Negative Numbers:
In machine learning, numbers represent feature measurements or directions:
- Positive numbers ($+5$) represent values above zero or increases.
- Negative numbers ($-5$) represent values below zero, decreases, or errors where predictions exceeded the true value.
- **Rules of Signs:**
  - $(+) \times (+) = (+)$ e.g., $3 \times 4 = 12$
  - $(-) \times (-) = (+)$ e.g., $(-3) \times (-4) = 12$ *(Crucial in squared errors: squaring negative numbers always gives positive numbers!)*
  - $(+) \times (-) = (-)$ e.g., $3 \times (-4) = -12$

### Fractions, Decimals, and Percentages:
These are three different ways of writing the exact same number:
$$\text{Fraction: } \frac{1}{4} \quad = \quad \text{Decimal: } 0.25 \quad = \quad \text{Percentage: } 25\%$$

- **Machine Learning Context:** Probabilities in models are expressed as decimals between $0.0$ and $1.0$. A predicted probability of $0.85$ means an $85\%$ chance.

---

## 2. Absolute Value ($|x|$)

### Intuition:
Absolute value measures the **magnitude or distance from zero**, completely ignoring the direction or sign.

### Small Example:
- $|5| = 5$
- $|-5| = 5$

### Why ML Uses It:
In regression, if a house is priced at \$300,000 and your model predicts \$280,000, your error is $300,000 - 280,000 = +20,000$.  
If the model predicts \$320,000, your error is $300,000 - 320,000 = -20,000$.  
By taking the absolute value $|-20,000| = 20,000$, we measure the **magnitude of the mistake** without letting positive and negative mistakes cancel each other out. This is the foundation of **Mean Absolute Error (MAE)** and **Lasso Regularization ($L_1$)**.

---

## 3. Powers and Exponents ($x^n$)

### Intuition:
An exponent indicates how many times a number is multiplied by itself.

### Small Examples:
- $2^2 = 2 \times 2 = 4$ *(Two squared)*
- $2^3 = 2 \times 2 \times 2 = 8$ *(Two cubed)*
- $10^3 = 10 \times 10 \times 10 = 1,000$

### Why ML Uses It:
1. **Squared Errors:** In Mean Squared Error (MSE), we square errors: $(-3)^2 = 9$. Squaring heavily penalizes large mistakes (an error of 10 becomes 100, while an error of 2 becomes only 4).
2. **Euclidean Distance:** The distance between points uses squares: $(x_2 - x_1)^2$.

---

## 4. Square Roots ($\sqrt{x}$)

### Intuition:
A square root asks the reverse question of squaring:  
*"What number multiplied by itself gives $x$?"*

### Small Examples:
- $\sqrt{4} = 2$ because $2 \times 2 = 4$.
- $\sqrt{9} = 3$ because $3 \times 3 = 9$.
- $\sqrt{25} = 5$ because $5 \times 5 = 25$.

### Why ML Uses It:
When we compute Mean Squared Error (MSE), our error is in squared units (e.g., "squared dollars" or "squared years"). Taking the square root gives the **Root Mean Squared Error (RMSE)**, which brings the error metric back to the original units (actual dollars).

---

## 5. Logarithms ($\log_b(x)$) — Demystifying the Math

Beginners are often intimidated by logarithms, but the concept is very simple.

### Intuition:
A logarithm asks:  
*"How many times must I multiply the base number to reach this target?"*

```text
Exponents:    10² = 100
                 │
                 ▼
Logarithm:   log₁₀(100) = 2
```

### Small Examples:
1. **Base 10:**
   - $10^1 = 10 \implies \log_{10}(10) = 1$
   - $10^2 = 100 \implies \log_{10}(100) = 2$
   - $10^3 = 1,000 \implies \log_{10}(1,000) = 3$
2. **Base 2 (Binary Logarithm, used in Decision Trees):**
   - $2^1 = 2 \implies \log_2(2) = 1$
   - $2^2 = 4 \implies \log_2(4) = 2$
   - $2^3 = 8 \implies \log_2(8) = 3$

### Key Properties of Logarithms:
- $\log(1) = 0$ (because any number to the power of 0 equals 1: $10^0 = 1$).
- $\log(A \times B) = \log(A) + \log(B)$ *(Turns slow, complex multiplications into simple additions!)*
- Logarithms squash massive numbers down to human-manageable scales.

---

## 6. Euler's Number ($e$) and the Natural Logarithm ($\ln$)

### What is $e$?
$e$ is a famous mathematical constant approximately equal to:
$$e \approx 2.71828$$
It represents continuous growth and is the natural base of exponential systems.

### The Natural Logarithm ($\ln(x)$):
When a logarithm uses base $e$, it is called the **Natural Logarithm** ($\ln$):
$$\ln(x) = \log_e(x)$$
If $e^2 \approx 7.389$, then $\ln(7.389) \approx 2$.

### Why ML Uses $e$ and $\ln$:
- **Logistic Regression:** Uses the **Sigmoid function** $\sigma(z) = \frac{1}{1 + e^{-z}}$ to squash predictions into clean probabilities between $0$ and $1$.
- **Cross-Entropy Loss (Log-Loss):** Measures classification error using natural logs: $-\ln(p)$.

---

## 7. Python Verification

```python
import math
import numpy as np

# Exponents
print("2^3 =", 2**3)  # 8

# Square roots
print("sqrt(25) =", math.sqrt(25))  # 5.0

# Base-10 Logarithm
print("log10(100) =", math.log10(100))  # 2.0

# Base-2 Logarithm (Used in Entropy & Decision Trees)
print("log2(8) =", math.log2(8))  # 3.0

# Natural Logarithm (ln) and Euler's constant (e)
print("e =", math.e)  # 2.71828...
print("ln(e) =", math.log(math.e))  # 1.0
```
