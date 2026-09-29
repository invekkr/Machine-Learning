# Mathematics Prerequisites: 04. Statistics for Machine Learning

Statistics is the discipline of collecting, summarizing, interpreting, and drawing conclusions from data. In Machine Learning, statistical concepts tell us what our features look like, how spread out they are, whether outliers are present, and how to scale them before training.

---

## 1. Measures of Central Tendency: Where is the Center?

When looking at a collection of numbers, the first question is:  
*"What is a typical or central value?"*

### 1. Mean (Arithmetic Average) — $\mu$ or $\bar{x}$
Sum all numbers and divide by the total count:
$$\mu = \frac{\sum_{i=1}^{N} x_i}{N}$$

**Symbol Breakdown:**
- $\mu$: Greek letter *mu* representing the mean.
- $\sum$: Greek letter *sigma* meaning "add all up".
- $x_i$: Each individual data value.
- $N$: Total number of values.

**Tiny Calculation:**
Values: $10, 20, 30$
$$\mu = \frac{10 + 20 + 30}{3} = \frac{60}{3} = \mathbf{20}$$
*Interpretation:* $20$ is the mathematical center balance point.

---

### 2. Median (The Middle Value)
Sort the numbers in ascending order and pick the value exactly in the center:
- **Odd Count:** $10, \mathbf{20}, 30 \implies \text{Median} = \mathbf{20}$.
- **Even Count:** $10, \mathbf{20}, \mathbf{30}, 40 \implies \frac{20 + 30}{2} = \mathbf{25}$.

**The Outlier Superpower of Median:**  
Suppose an extreme salary of \$1,000,000 enters our sample:
- Sample: $10, 20, 30, 40, 1000$
- Mean: $\frac{10 + 20 + 30 + 40 + 1000}{5} = \frac{1100}{5} = \mathbf{220}$ *(Corrupted by the outlier!)*
- Median: $\mathbf{30}$ *(Completely robust and unaffected!)*

---

### 3. Mode (Most Frequent Value)
The value that appears most often.
- Sample: $25, 30, 25, 40, 25 \implies \text{Mode} = \mathbf{25}$.
- Essential for **imputing missing values in categorical data** (`SimpleImputer(strategy='most_frequent')`).

---

## 2. Measures of Spread: How Dispersed is the Data?

Two datasets can have the exact same mean of $20$, yet look completely different:
- Dataset A: $20, 20, 20 \longrightarrow \text{Mean} = 20$ (Zero spread, all identical)
- Dataset B: $0, 20, 40 \longrightarrow \text{Mean} = 20$ (Wide spread)

We need metrics to quantify **spread**.

---

### 1. Range
$$\text{Range} = \text{Maximum} - \text{Minimum}$$
For Dataset B: $40 - 0 = 40$.

---

### 2. Variance ($\sigma^2$) — "Average Squared Distance from Mean"
Variance measures how far numbers fluctuate around their average:
$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

**Step-by-Step Manual Calculation:**
Values: $10, 20, 30 \quad (\mu = 20)$
1. Subtract mean from each value:
   - $10 - 20 = -10$
   - $20 - 20 = 0$
   - $30 - 20 = +10$
2. Square each difference (so negative signs don't cancel!):
   - $(-10)^2 = 100$
   - $(0)^2 = 0$
   - $(+10)^2 = 100$
3. Compute the average of these squared differences:
   $$\sigma^2 = \frac{100 + 0 + 100}{3} = \frac{200}{3} \approx \mathbf{66.67}$$

---

### 3. Standard Deviation ($\sigma$) — "The Typical Distance"
Notice that variance is in **squared units** ("squared dollars" or "squared years").  
To bring the spread back to the **original units**, we take the square root of variance:
$$\sigma = \sqrt{\sigma^2}$$

$$\sigma = \sqrt{66.67} \approx \mathbf{8.16}$$
*Interpretation:* A typical observation in this dataset is about $8.16$ units away from the average ($20$).

---

### 4. Interquartile Range (IQR) & Percentiles
- **$25^{\text{th}}$ Percentile ($Q_1$):** 25% of the data falls below this value.
- **$50^{\text{th}}$ Percentile ($Q_2$ / Median):** Half the data falls below this value.
- **$75^{\text{th}}$ Percentile ($Q_3$):** 75% of the data falls below this value.
- **$\text{IQR} = Q_3 - Q_1$**: The spread of the middle 50% of the dataset.

**Outlier Detection (Tukey's Rule):**
Any point falling outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$ is flagged as an outlier.

---

## 3. The Z-Score (Standardization Foundation)

How many standard deviations is a single data point above or below average?
$$z = \frac{x - \mu}{\sigma}$$

If class average exam score is $70$ with $\sigma = 10$:
- A student scoring $90$ has $z = \frac{90 - 70}{10} = \mathbf{+2.0}$ *(2 standard deviations above average!)*
- A student scoring $60$ has $z = \frac{60 - 70}{10} = \mathbf{-1.0}$ *(1 standard deviation below average)*

This is the exact mathematical foundation of Scikit-Learn's **`StandardScaler`**.

---

## 4. Covariance and Correlation: How Features Move Together

- **Covariance ($\text{Cov}(X, Y)$):** Measures whether two variables increase together (positive) or move in opposite directions (negative).
- **Correlation ($r$):** Standardizes covariance to lie strictly between **$-1.0$ and $+1.0$**:
  - $r = +1.0$: Perfect positive linear relationship (as $X$ rises, $Y$ rises).
  - $r = 0.0$: No linear relationship.
  - $r = -1.0$: Perfect negative linear relationship (as $X$ rises, $Y$ falls).

---

## 5. Python Verification

```python
import numpy as np

data = np.array([10, 20, 30])

print("Mean:              ", np.mean(data))  # 20.0
print("Median:            ", np.median(data))  # 20.0
print("Variance (pop):    ", np.var(data))  # 66.67
print("Standard Deviation:", np.std(data))  # 8.16
```
