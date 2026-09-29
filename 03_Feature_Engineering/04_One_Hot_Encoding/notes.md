# Module 03: Feature Engineering — One-Hot Encoding (OHE)

In machine learning, algorithms do not understand text labels. They do not know what `"Red"`, `"Blue"`, or `"Green"` mean. They are mathematical functions that calculate weighted sums, matrix dot products, and geometric distances.

To feed categorical data into these algorithms, we must convert words into numbers. 

However, **how** we convert them matters immensely. If we do it carelessly, we inadvertently teach the model false mathematical relationships that ruin its predictions.

**One-Hot Encoding (OHE)** is the primary, mathematically sound technique used to convert nominal categorical data into numerical form.

---

## 1. What is One-Hot Encoding?

### Understand It First

Consider a dataset containing a feature named `Color`:

```text
Color
-----
Red
Blue
Green
Red
```

These values are **categorical labels**. 
If you try to pass them into a machine learning algorithm like Linear Regression, Support Vector Machines, or Neural Networks:
$$\hat{y} = w_1 \times \text{Color} + b$$
The computer immediately fails because it cannot multiply a mathematical weight $w_1$ by the string `"Red"`.

### The Naive Approach: Assigning Arbitrary Integers
A natural first thought is: *"Why not just assign each color an integer?"*
```text
Red   → 0
Blue  → 1
Green → 2
```

### The Serious Flaw in Integer Encoding:
The moment you write $0, 1, 2$, the mathematical model assumes arithmetic relationships that do not exist:
1. **False Hierarchy:** The model assumes $\text{Green} (2) > \text{Blue} (1) > \text{Red} (0)$.
2. **False Distance:** The model calculates that the distance between Green and Red ($2 - 0 = 2$) is **twice as large** as the distance between Blue and Red ($1 - 0 = 1$).
3. **False Arithmetic:** The model assumes $\text{Red} (0) + \text{Green} (2) = \text{Green} (2)$ and $\text{Blue} (1) + \text{Blue} (1) = \text{Green} (2)$.

Colors have **no natural ordering**. Green is not "greater" than Red, nor is Mumbai "twice" Chennai. 
This naive approach introduces an **artificial numerical relationship** that biases and corrupts your model.

---

### The Solution: One-Hot Encoding
Instead of cramming all categories into a single column with ordered numbers, **we create a new binary column for every unique category**:

| Color | Color_Red | Color_Blue | Color_Green |
| :--- | :---: | :---: | :---: |
| **Red** | **1** | 0 | 0 |
| **Blue** | 0 | **1** | 0 |
| **Green** | 0 | 0 | **1** |
| **Red** | **1** | 0 | 0 |

### How It Works:
- Each unique category receives its own dedicated binary column.
- **`1`** indicates that the category is **present** for that observation.
- **`0`** indicates that the category is **absent**.
- Because every category is now represented as an independent axis, **no artificial hierarchy or ordering is introduced**.

> **Proper Definition:**  
> **One-Hot Encoding** is a feature engineering technique that converts nominal categorical variables into multiple binary (0/1) features, with one binary column representing each distinct category.

---

## 2. Why Do We Need One-Hot Encoding?

Nominal categorical data has **no inherent or meaningful order**. 

When we evaluate distances between categories:
- In integer encoding:
  $$\text{Distance}(\text{Red}, \text{Green}) = |0 - 2| = 2$$
  $$\text{Distance}(\text{Red}, \text{Blue}) = |0 - 1| = 1$$
  *(Red and Blue appear artificially closer than Red and Green!)*

- In One-Hot Encoding:
  $$\text{Red} = [1, 0, 0]$$
  $$\text{Blue} = [0, 1, 0]$$
  $$\text{Green} = [0, 0, 1]$$

  Now, compute the Euclidean distance between any pair:
  $$\text{Distance}(\text{Red}, \text{Blue}) = \sqrt{(1-0)^2 + (0-1)^2 + (0-0)^2} = \sqrt{1 + 1 + 0} = \sqrt{2} \approx 1.414$$
  $$\text{Distance}(\text{Red}, \text{Green}) = \sqrt{(1-0)^2 + (0-0)^2 + (0-1)^2} = \sqrt{1 + 0 + 1} = \sqrt{2} \approx 1.414$$
  $$\text{Distance}(\text{Blue}, \text{Green}) = \sqrt{(0-0)^2 + (1-0)^2 + (0-1)^2} = \sqrt{0 + 1 + 1} = \sqrt{2} \approx 1.414$$

**Every category is now perfectly equidistant from every other category.** 
The model sees them as distinct, equal options without any unintended bias.

### Comparison Table:

| Encoding Method | Representation | The Problem |
| :--- | :--- | :--- |
| **Integer / Label Encoding** | $0, 1, 2, \dots$ | Introduces artificial mathematical ranking and unequal distances. |
| **One-Hot Encoding** | Multiple $0/1$ binary columns | Preserves category independence with zero artificial ordering. |

---

## 3. How One-Hot Encoding Works in Practice

Let's trace a small example with a feature named `City`:

```text
City:
Delhi
Mumbai
Chennai
```

### Step 1: Identify Unique Categories
There are 3 unique categories: `Delhi`, `Mumbai`, and `Chennai`.

### Step 2: Create 3 Binary Indicator Columns
We generate three new columns:
1. `City_Delhi`
2. `City_Mumbai`
3. `City_Chennai`

### Step 3: Populate Binary Values
- For a passenger from Delhi: `[1, 0, 0]`
- For a passenger from Mumbai: `[0, 1, 0]`
- For a passenger from Chennai: `[0, 0, 1]`

| Original City | City_Delhi | City_Mumbai | City_Chennai |
| :--- | :---: | :---: | :---: |
| **Delhi** | **1** | 0 | 0 |
| **Mumbai** | 0 | **1** | 0 |
| **Chennai** | 0 | 0 | **1** |

### Why is it Called "One-Hot"?
The term comes from digital electronics: in a digital circuit with multiple lines, a state where **exactly one line is high (1 / "hot") while all others are low (0 / "cold")** is called *one-hot*.
In our table, for every single row, **exactly one column is 1 ("hot")** and the rest are 0.

---

## 4. OHE vs. Label Encoding vs. Ordinal Encoding

It is vital not to mix up these three preprocessing methods:

```text
                           Categorical Feature
                                    │
                  Does it have a real, meaningful order?
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                   YES                              NO
                    │                               │
             Ordinal Data                     Nominal Data
                    │                               │
             Ordinal Encoding                One-Hot Encoding
         (High School=0, UG=1, PG=2)           (Delhi=[1,0,0], Mumbai=[0,1,0])
```

- **Nominal Data (Colors, Cities, Blood Groups):** Has no order $\longrightarrow$ **Use One-Hot Encoding**.
- **Ordinal Data (Education, Ratings, Sizes: S/M/L):** Has real hierarchical progression $\longrightarrow$ **Use Ordinal Encoding**.
- **Target Variable $y$ (Pass/Fail, Cat/Dog):** Classification label $\longrightarrow$ **Use Label Encoding**.

---

## 5. The Dummy Variable Trap

Now we encounter a critical mathematical issue that every machine learning engineer must understand: **The Dummy Variable Trap**.

Consider a feature named `Gender` with two categories: `Male` and `Female`.

If we perform standard One-Hot Encoding:

| Row | Gender | Gender_Male | Gender_Female |
| :---: | :--- | :---: | :---: |
| 1 | Male | 1 | 0 |
| 2 | Female | 0 | 1 |
| 3 | Female | 0 | 1 |
| 4 | Male | 1 | 0 |

### Notice the Perfect Mathematical Dependency:
Look at the two generated columns:
$$\text{Gender\_Female} = 1 - \text{Gender\_Male}$$

- If you know that $\text{Gender\_Male} = 1$, you **know with 100% certainty** that $\text{Gender\_Female} = 0$.
- If you know that $\text{Gender\_Male} = 0$, you **know with 100% certainty** that $\text{Gender\_Female} = 1$.

The second column provides **zero new information**. It is completely redundant.

### Why This Breaks Models (Multicollinearity):
In statistical models that include a constant intercept (like Ordinary Least Squares Linear Regression or Logistic Regression):
$$y = w_0 + w_1 \times \text{Male} + w_2 \times \text{Female}$$

Because $\text{Male} + \text{Female} = 1$, the features are linearly dependent. 
Mathematically, the matrix $(X^T X)$ becomes non-invertible (singular), causing:
- Infinite or unstable weight coefficients ($w_1, w_2$).
- Inability to interpret feature importance.
- Inflated standard errors.

> **Formal Definition:**  
> The **Dummy Variable Trap** is a scenario in which the one-hot encoded dummy variables are highly correlated (perfect multicollinearity), allowing one feature to be perfectly predicted from the others.

---

## 6. How to Avoid the Dummy Variable Trap: $N$ Categories $\rightarrow N - 1$ Columns

The solution is simple and elegant:
> **If a categorical variable has $N$ distinct categories, create only $N - 1$ dummy columns.**

### Example 1: Gender ($N = 2$)
We have 2 categories: `Male` and `Female`.
We drop one column (e.g. drop `Female`) and keep only `Male`:

| Person | Gender_Male | Meaning |
| :--- | :---: | :--- |
| **Person 1** | **1** | Person is **Male** |
| **Person 2** | **0** | Person is **Female** (the dropped baseline) |

We did not lose a single shred of information! When `Gender_Male == 0`, we know the person is Female.
The dropped category is called the **reference category** (or baseline category).

---

### Example 2: Colors ($N = 3$)
We have 3 categories: `Red`, `Blue`, `Green`.
Instead of creating 3 columns, we drop one (e.g., drop `Green`) and keep 2 columns:

| Color | Color_Red | Color_Blue | How It is Interpreted |
| :--- | :---: | :---: | :--- |
| **Red** | **1** | 0 | Red is present |
| **Blue** | 0 | **1** | Blue is present |
| **Green** | **0** | **0** | Both are 0 $\longrightarrow$ It **must be Green**! |

Notice how `[0, 0]` uniquely and unambiguously identifies the dropped category (`Green`).

> **The Rule:**  
> Dropping one column eliminates perfect multicollinearity while preserving 100% of the categorical information.

---

## 7. Implementation with Pandas: `pd.get_dummies()`

Pandas provides a quick, convenient function called `pd.get_dummies()`.

### Basic Usage:
```python
import pandas as pd

df = pd.DataFrame({"Color": ["Red", "Blue", "Green", "Red"]})

# Basic One-Hot Encoding (creates N columns)
df_ohe = pd.get_dummies(df, columns=["Color"], dtype=int)
```

Output:
```text
   Color_Blue  Color_Green  Color_Red
0           0            0          1
1           1            0          0
2           0            1          0
3           0            0          1
```

### Avoiding the Dummy Variable Trap with `drop_first=True`:
To automatically drop the first category ($N - 1$ columns):
```python
# Drops the first category alphabetically ('Color_Blue')
df_ohe_dropped = pd.get_dummies(
    df, columns=["Color"], drop_first=True, dtype=int
)
```

Output:
```text
   Color_Green  Color_Red
0            0          1  -> Red
1            0          0  -> Blue (both are 0, so it's the dropped baseline!)
2            1          0  -> Green
3            0          1  -> Red
```

---

## 8. Implementation with Scikit-Learn: `OneHotEncoder`

While `pd.get_dummies()` is great for quick data exploration, **it should not be used in machine learning production pipelines**.

### Why `pd.get_dummies()` Fails in Production ML:
1. It does not "remember" the training categories.
2. If your test set has different categories, or categories in a different order, `pd.get_dummies()` produces misaligned columns.
3. If an unseen category appears in the test set, `pd.get_dummies()` creates an unexpected column that breaks model prediction.

### Why Scikit-Learn's `OneHotEncoder` is the Gold Standard:
- It adheres to Scikit-Learn's `fit()` / `transform()` architecture.
- It learns the categories from `X_train` during `fit()` and freezes that structure.
- It guarantees that `X_test` will have the exact same columns in the exact same order.
- It provides built-in handling for unseen categories (`handle_unknown='ignore'`).

```python
from sklearn.preprocessing import OneHotEncoder

# Initialize encoder
# sparse_output=False returns a dense NumPy array instead of a sparse matrix
# drop='first' avoids the dummy variable trap
encoder = OneHotEncoder(sparse_output=False, drop="first")

# Fit and transform
X_train_encoded = encoder.fit_transform(X_train[["Color"]])
```

---

## 9. Understanding `fit()`, `transform()`, and Data Leakage

Understanding the difference between `fit()` and `transform()` is critical:

```text
Training Data (X_train)                  Test Data (X_test)
         │                                       │
      fit()                                      │
         │                                       │
Learns unique categories                         │
(e.g., ['Delhi', 'Mumbai', 'Chennai'])          │
         │                                       │
    transform()                               transform()
         │                                       │
Encoded X_train                         Encoded X_test
(Using training categories)             (Using the SAME training categories)
```

- **`fit(X_train)`**: Scans the training data and memorizes the unique categories.
- **`transform(X)`**: Encodes the rows into binary columns based on the categories memorized during `fit()`.
- **`fit_transform(X_train)`**: A combined convenience method that runs `fit()` then `transform()` in one step.

### Why You Must Split BEFORE Encoding (Preventing Data Leakage):
If you run `encoder.fit(X)` on your entire dataset before splitting into train and test:
1. The encoder uses categories and frequencies that exist only in the test set.
2. Information from the test set leaks into the preprocessing stage (**Data Leakage**).
3. Your test set is no longer a true simulation of unseen real-world data.

> **Rule:**  
> Always split your dataset into `X_train` and `X_test` first.  
> Call `fit_transform()` on `X_train`, and call **only** `transform()` on `X_test`.

---

## 10. The Real-World Challenge: Unseen / Unknown Categories

What happens when your model is deployed in production and receives a category it has never seen before?

### The Problem:
- **Training data:** `City` contains `Delhi`, `Mumbai`.
- **New incoming test record:** A customer from `Bangalore`!

By default, Scikit-Learn's `OneHotEncoder` will raise an immediate exception:
```text
ValueError: Found unknown categories ['Bangalore'] in column 0 during transform
```

### The Solution: `handle_unknown='ignore'`
When initializing `OneHotEncoder`, pass `handle_unknown='ignore'`:

```python
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
```

When an unseen category (`"Bangalore"`) appears during `transform(X_test)`:
- The encoder does **not crash**.
- It safely encodes the unknown category as **all zeros** (`[0, 0]`).
- The model treats the unknown category as having zero contribution from the known city indicators.

*(Note: When `handle_unknown='ignore'`, Scikit-Learn requires keeping all $N$ columns, because an all-zero vector is reserved for unseen values rather than a dropped baseline).*

---

## 11. High Cardinality & Dimensionality Explosion

One-Hot Encoding is powerful, but it has one major vulnerability: **High Cardinality**.

> **Cardinality** refers to the number of distinct unique categories in a feature.
> - `Gender`: 2 unique values $\longrightarrow$ Low cardinality.
> - `US State`: 50 unique values $\longrightarrow$ Moderate cardinality.
> - `City` / `ZIP Code`: 10,000 unique values $\longrightarrow$ **High cardinality**.

### What Happens if You One-Hot Encode High Cardinality Features?
If a dataset has 10,000 unique cities, One-Hot Encoding creates **10,000 new columns**!
This causes **Dimensionality Explosion**:
1. **Memory Exhaustion:** Dataset size balloons from megabytes to gigabytes.
2. **Computational Slowness:** Training time increases dramatically.
3. **The Curse of Dimensionality:** Data points become extremely sparse in high-dimensional space, leading to severe overfitting.

---

## 12. Handling High Cardinality: Rare Category Grouping

How do we prevent dimensionality explosion when dealing with many categories?
One practical, widely used technique is **Rare Category Grouping** (also known as creating an `"Other"` category).

### Intuition:
In many datasets, a small handful of top categories account for 90%+ of all observations, while hundreds of tiny categories appear only 1 or 2 times each.

```text
City Breakdown:
Delhi        → 50,000 records
Mumbai       → 40,000 records
Bangalore    → 35,000 records
SmallTownA   → 3 records
SmallTownB   → 2 records
SmallTownC   → 1 record
... (1,000 other small towns)
```

Instead of creating 1,000 separate columns for towns that appear once:
1. Identify categories that appear less than a chosen threshold.
2. Replace all rare town names with the label `"Other"`.
3. Apply One-Hot Encoding to the grouped column.

```text
Resulting Categories:
- Delhi
- Mumbai
- Bangalore
- Other
(Only 4 columns instead of 1,000!)
```

### Is There a Universal Threshold?
**NO.** There is no universal rule like *"count < 100"* or *"frequency < 5%"*.
The appropriate cutoff depends on:
- Total dataset size (a count of 50 in a 1,000-row dataset is large; in a 10-million-row dataset, it's negligible).
- Number of categories.
- Domain context and business requirements.
- Percentage cutoffs (e.g. grouping categories representing $< 1\%$ or $< 2\%$ of the data) are common practical choices.

---

## 13. When to Use vs. When NOT to Use One-Hot Encoding

### ✅ When to Use One-Hot Encoding:
1. The categorical feature is **nominal** (no natural order).
2. Cardinality is **low to moderate** (typically $< 15$ to $20$ unique categories).
3. Using linear models, logistic regression, SVMs, KNN, or neural networks that rely on geometric distance or linear weight coefficients.

---

### ❌ When NOT to Use One-Hot Encoding:
1. **The feature is ordinal:** (e.g. Education, Ratings, T-shirt sizes).  
   $\longrightarrow$ **Use Ordinal Encoding** to preserve the hierarchy.
2. **The feature has high cardinality:** (e.g. 5,000 product IDs, user IDs, or URLs).  
   $\longrightarrow$ One-Hot Encoding will cause dimensionality explosion.  
   $\longrightarrow$ Use alternative techniques:
   - **Target Encoding / Mean Encoding**
   - **Frequency / Count Encoding**
   - **Feature Hashing (Hashing Trick)**
   - **Learned Entity Embeddings** (in deep learning)
3. **Using Tree-Based Models on High-Cardinality Data:**  
   Tree models (Random Forests, Gradient Boosting) struggle with wide one-hot encoded data because individual trees must make dozens of one-hot splits to isolate groups. Integer encoding or native categorical handling (like in LightGBM/CatBoost) is often more efficient.

---

## 14. Summary Comparison: Encoding Techniques

| Feature | Category Type | Recommended Encoder | Reason |
| :--- | :--- | :--- | :--- |
| **`Education`** (High School, UG, PG) | Ordinal | **`OrdinalEncoder`** | Categories have an ordered ranking that must be preserved. |
| **`Customer Rating`** (1 star to 5 star) | Ordinal | **`OrdinalEncoder`** | 5 stars is quantitatively higher than 1 star. |
| **`City`** (Delhi, Mumbai, Chennai) | Nominal (Low Cardinality) | **`OneHotEncoder`** | No hierarchy; prevents artificial numerical relationships. |
| **`Gender`** (Male, Female) | Nominal | **`OneHotEncoder(drop='first')`** | Avoids the dummy variable trap. |
| **`ZIP Code`** (10,000 distinct codes) | Nominal (High Cardinality) | **Target Encoding / Hashing** | Avoids dimensionality explosion. |
| **`Loan Status`** (Approved, Denied) | Classification Target ($y$) | **`LabelEncoder`** | Target labels only; maps classes to $0, 1$. |

---

## 15. Complete Decision Flowchart

```text
                           Categorical Feature
                                    │
                         Is it the Target (y)?
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
                   YES                              NO
                    │                               │
              LabelEncoder                Is there a meaningful
             (Pass=1, Fail=0)                     order?
                                                    │
                                    ┌───────────────┴───────────────┐
                                    ▼                               ▼
                                   YES                              NO
                                    │                               │
                              Ordinal Data                     Nominal Data
                                    │                               │
                              OrdinalEncoder                        │
                        (categories=[...])                          │
                                                           Is cardinality high?
                                                            (> 20 categories)
                                                                    │
                                                    ┌───────────────┴───────────────┐
                                                    ▼                               ▼
                                                   YES                              NO
                                                    │                               │
                                          Rare Grouping /                    OneHotEncoder
                                          Target Encoding                  (drop='first' or
                                                                        handle_unknown='ignore')
```

---

## 16. Final Important Takeaways

- **One-Hot Encoding** is designed specifically for **nominal** categorical data.
- It transforms each category into an independent binary ($0/1$) column.
- **$1$** means the category is present; **$0$** means it is absent.
- It eliminates false numerical hierarchies and ensures categories are equidistant.
- $N$ categories produce $N$ columns.
- Dropping one column ($N - 1$ columns) prevents the **Dummy Variable Trap** (perfect multicollinearity) in linear models.
- **Pandas:** `pd.get_dummies(df, drop_first=True)` is great for quick analysis.
- **Scikit-Learn:** `OneHotEncoder(sparse_output=False)` is mandatory for robust ML pipelines.
- Always fit the encoder on `X_train` and only transform `X_test` to prevent **data leakage**.
- Use `handle_unknown="ignore"` to safely handle unseen categories in production.
- High cardinality causes **dimensionality explosion**; mitigate it using **Rare Category Grouping** (`"Other"`).
- **Golden Rule:** Ordinal $\longrightarrow$ `OrdinalEncoder`, Nominal $\longrightarrow$ `OneHotEncoder`, Target $\longrightarrow$ `LabelEncoder`.
