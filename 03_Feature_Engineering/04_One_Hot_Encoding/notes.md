# Module 03: Feature Engineering — One-Hot Encoding (OHE)

In machine learning, algorithms do not understand text labels. They do not know what `"Red"`, `"Blue"`, or `"Green"` mean. They are mathematical optimization engines that calculate weighted sums, matrix dot products, and geometric distances.

To feed categorical data into these algorithms, we must convert words into numbers. 

However, **how** we convert them matters immensely. If we do it carelessly, we inadvertently teach the model false mathematical relationships that corrupt its predictions.

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
- Because every category is now represented as an independent orthogonal axis, **no artificial hierarchy or ordering is introduced**.

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
The model sees them as distinct, equal options without any unintended geometric bias.

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
The term comes from digital circuit electronics: in a digital bus with multiple lines, a state where **exactly one line is high (1 / "hot") while all others are low (0 / "cold")** is called *one-hot*.
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

## 5. The Dummy Variable Trap (Why Redundant Columns Confuse Models)

Now let's understand a common issue in categorical encoding: **The Dummy Variable Trap**.

Consider a feature named `Gender` with two categories: `Male` and `Female`.

If we create a column for both:

| Row | Gender | Gender_Male | Gender_Female |
| :---: | :--- | :---: | :---: |
| 1 | Male | 1 | 0 |
| 2 | Female | 0 | 1 |
| 3 | Female | 0 | 1 |
| 4 | Male | 1 | 0 |

### Notice the Obvious Redundancy:
- If someone is **Male** (`Gender_Male = 1`), you already know they are **not Female** (`Gender_Female = 0`).
- If someone is **not Male** (`Gender_Male = 0`), you already know they **must be Female** (`Gender_Female = 1`).

In simple mathematical terms:
$$\text{Gender\_Female} = 1 - \text{Gender\_Male}$$

The second column gives us **zero new information**. It is 100% predictable from the first column!

When two or more columns in a dataset are perfectly predictable from each other, statisticians call it **multicollinearity**.

### Why This Confuses Machine Learning Models:
In linear models (like Linear Regression and Logistic Regression), the model tries to learn an individual weight for each column.
When two columns carry the exact same information, the model cannot decide how much weight belongs to `Gender_Male` versus `Gender_Female`. This creates numerical instability and makes the model weights unreliable.

This problem is called the **Dummy Variable Trap**.

---

## 6. How to Avoid the Dummy Variable Trap: Use $N - 1$ Columns

The fix is very simple:
> **If a categorical variable has $N$ distinct categories, create only $N - 1$ dummy columns.**

### Example 1: Gender ($N = 2$)
We have 2 categories: `Male` and `Female`.
We drop one column (e.g., drop `Female`) and keep only `Male`:

| Person | Gender_Male | Meaning |
| :--- | :---: | :--- |
| **Person 1** | **1** | Person is **Male** |
| **Person 2** | **0** | Person is **Female** (the dropped baseline) |

We did not lose any information! When `Gender_Male == 0`, we know with certainty the person is Female.
The dropped category is called the **reference (or baseline) category**.

---

### Example 2: Colors ($N = 3$)
We have 3 categories: `Red`, `Blue`, `Green`.
Instead of creating 3 columns, we drop one (e.g., drop `Green`) and keep 2 columns:

| Color | Color_Red | Color_Blue | How It is Interpreted |
| :--- | :---: | :---: | :--- |
| **Red** | **1** | 0 | Red is present |
| **Blue** | 0 | **1** | Blue is present |
| **Green** | **0** | **0** | Both are 0 $\longrightarrow$ It **must be Green**! |

Notice how `[0, 0]` uniquely and clearly identifies the dropped category (`Green`).

> **The Rule:**  
> Dropping one column eliminates redundancy while preserving 100% of the information.

---

## 7. When Does Dropping a Column Matter?

Do all machine learning algorithms require you to drop a column?

- **Linear Models (Linear Regression, Logistic Regression):**
  **YES.** You should drop one column (`drop='first'`). These models are sensitive to redundant columns and can become unstable if you keep all $N$ columns.
- **Tree-Based Models (Decision Trees, Random Forests):**
  **NO.** Trees evaluate one feature at a time, so having redundant columns does not break them.
- **General Practice for Beginners:**
  Dropping the first column is a standard, safe habit when working with linear models to keep your features clean and independent.

---

## 8. Implementation with Pandas: `pd.get_dummies()`

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

## 9. Implementation with Scikit-Learn: `OneHotEncoder`

While `pd.get_dummies()` is convenient for quick exploratory data analysis, **it should not be used in machine learning production pipelines**.

### Why `pd.get_dummies()` Fails in Production ML:
1. **No Memory:** It does not "remember" the training categories.
2. **Column Misalignment:** If your test set has different categories, or categories in a different order, `pd.get_dummies()` produces misaligned columns that crash your model.
3. **Unseen Categories:** If an unseen category appears in the test set, `pd.get_dummies()` silently creates an unexpected column that breaks model ingestion.

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

## 10. Understanding `fit()`, `transform()`, and Data Leakage

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

## 11. Handling Unseen Categories (`handle_unknown='ignore'`)

What happens if your training data only contains three cities:
`Delhi`, `Mumbai`, and `Chennai`.

Then, a new row appears in your test data with the city **`Kolkata`**?

- **By default:** Scikit-Learn will throw an error and crash because it does not recognize `Kolkata`.
- **With `handle_unknown='ignore'`:** Instead of crashing, Scikit-Learn represents the unseen city with all zeros (`[0, 0, 0]`). This allows your code to run smoothly even when unexpected categories show up in new data.

```python
# Safe encoder for production data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
```

---

## 12. High Cardinality & Dimensionality Explosion

One-Hot Encoding is powerful, but it has one major vulnerability: **High Cardinality**.

> **Cardinality** refers to the number of distinct unique categories in a feature.
> - `Gender`: 2 unique values $\longrightarrow$ Low cardinality.
> - `US State`: 50 unique values $\longrightarrow$ Moderate cardinality.
> - `City` / `ZIP Code` / `User ID`: 10,000 unique values $\longrightarrow$ **High cardinality**.

### What Happens if You One-Hot Encode High Cardinality Features?
If a dataset has 10,000 unique cities, One-Hot Encoding creates **10,000 new columns**!
This causes **Dimensionality Explosion**:
1. **Memory Exhaustion:** Dataset size balloons from megabytes to gigabytes.
2. **Computational Slowness:** Training time increases quadratically for many models.
3. **The Curse of Dimensionality:** Data points become extremely sparse in high-dimensional space, leading to severe overfitting.
4. **Tree Degradation:** Decision trees must split dozens of times across one-hot columns to isolate groups, destroying tree depth efficiency.

---

## 13. Handling High Cardinality: Rare Category Grouping & Alternatives

How do we prevent dimensionality explosion when dealing with many categories?

### 1. Rare Category Grouping (The `"Other"` Strategy)
In many real-world datasets, a small handful of top categories account for 90%+ of all observations, while hundreds of tiny categories appear only 1 or 2 times each:

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
The cutoff depends on:
- Total dataset size (a count of 50 in a 1,000-row dataset is large; in a 10-million-row dataset, it's negligible).
- Number of categories.
- Domain context and business requirements.
- Percentage cutoffs (e.g. grouping categories representing $< 1\%$ or $< 2\%$ of the data) are common practical choices.

### 2. Modern Scikit-Learn: `min_frequency` Parameter
Modern Scikit-Learn provides built-in rare grouping directly inside `OneHotEncoder`:
```python
# Automatically groups categories appearing in less than 2% of rows into 'infrequent_sklearn'
encoder = OneHotEncoder(sparse_output=False, min_frequency=0.02)
```

### 3. Alternative Encodings for Extreme Cardinality ($> 50$ Categories):
When cardinality exceeds 50–100 categories, One-Hot Encoding should be replaced by:
- **Target Encoding (Mean Encoding):** Replaces each category with the average target value of that category (requires regularization/smoothing to prevent leakage).
- **Frequency / Count Encoding:** Replaces each category with its occurrence count.
- **Feature Hashing (The Hashing Trick):** Hashes categories into a fixed number of bins (e.g., 32 or 64).
- **Entity Embeddings:** Learns dense continuous vectors for each category in deep learning architectures.

---

## 14. Sparse vs. Dense Output (`sparse_output`)

When you One-Hot Encode categorical data, most of the values in the new columns will be `0`.

- **Dense Output (`sparse_output=False`):** Returns a standard array / table where every `0` and `1` is visible. This is easy to read and inspect.
- **Sparse Output (`sparse_output=True`):** A memory-saving format that only remembers where the `1`s are, skipping the zeros. This is helpful for huge datasets with many columns.
- **For our study:** We set `sparse_output=False` so we can clearly see and print the encoded numbers.

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
- Dropping one column ($N - 1$ columns) prevents the **Dummy Variable Trap** (perfect multicollinearity) in unregularized linear models.
- **Pandas:** `pd.get_dummies(df, drop_first=True)` is great for quick analysis.
- **Scikit-Learn:** `OneHotEncoder(sparse_output=False)` is mandatory for robust ML pipelines.
- Always fit the encoder on `X_train` and only transform `X_test` to prevent **data leakage**.
- Use `handle_unknown="ignore"` to safely handle unseen categories in production.
- High cardinality causes **dimensionality explosion**; mitigate it using **Rare Category Grouping** (`"Other"`).
- **Golden Rule:** Ordinal $\longrightarrow$ `OrdinalEncoder`, Nominal $\longrightarrow$ `OneHotEncoder`, Target $\longrightarrow$ `LabelEncoder`.
