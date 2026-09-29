# Module 03: Feature Engineering — Encoding Categorical Data

Machine learning algorithms are fundamentally mathematical optimization engines. They compute dot products ($w^T x$), matrix inversions, and geometric distances ($\sqrt{\sum (x_i - y_i)^2}$). 

Because mathematical operations cannot be performed directly on text strings like `"Delhi"`, `"Female"`, or `"Undergraduate"`, we must convert categorical variables into numbers.

However, **how** we convert them determines whether our model learns valid patterns or nonsensical artificial relationships.

---

## 1. What is Categorical Data?

### Understand It First

Look at a sample employee dataset:

| Name | Education | City | Experience (Years) |
| :--- | :--- | :--- | :---: |
| **Rahul** | Graduate | Delhi | 2 |
| **Amit** | Postgraduate | Mumbai | 4 |
| **Neha** | Graduate | Delhi | 3 |
| **Priya** | Doctorate | Chennai | 6 |

Notice the difference between the columns:
- **`Experience`**: A **numerical** variable. You can add them, take an average, or say that 4 years is twice as much as 2 years.
- **`Education` & `City`**: **Categorical** variables. They contain text labels representing categories or groups, not direct numerical counts.

### Why Do We Need Encoding?

If you try to pass text strings into an algorithm like Linear Regression, Support Vector Machines, or Neural Networks:
$$\text{Salary} = w_1 \times \text{Education} + w_2 \times \text{Experience} + b$$

The computer throws an immediate error:
$$\text{TypeError: can't multiply sequence by non-int of type 'float'}$$

We must translate these words into numbers.

> **Proper Definition:**  
> **Categorical Encoding** is the process of converting categorical values into numerical representations so that machine learning algorithms can process them without introducing misleading mathematical relationships.

---

## 2. Types of Categorical Data: Nominal vs. Ordinal

Categorical data is not homogeneous. It falls into two fundamentally different families:

```text
Categorical Data
      ↓
 ┌───────────┐
 ↓           ↓
Nominal     Ordinal
(No Order)  (Meaningful Order)
```

### 2.1 Nominal Data

#### Understand It First
Consider the `City` column:
$$\text{Delhi},\; \text{Mumbai},\; \text{Chennai},\; \text{Kolkata}$$

Can you say that *Delhi > Mumbai*? Or that *Chennai is twice Kolkata*?
**No.** There is no inherent rank, hierarchy, or natural sequence. Every category is just a distinct qualitative label.

Other real-world examples:
- **Gender:** Male, Female, Other
- **Blood Group:** A+, B+, O+, AB+
- **Department:** Sales, Marketing, Engineering, HR
- **Color:** Red, Green, Blue
- **Country:** India, USA, Germany, Japan

> **Proper Definition:**  
> **Nominal data** is categorical data whose categories have **no inherent, quantitative, or meaningful order**.

---

### 2.2 Ordinal Data

#### Understand It First
Now consider the `Education` column:
$$\text{High School} \longrightarrow \text{Undergraduate} \longrightarrow \text{Postgraduate} \longrightarrow \text{Doctorate}$$

Or customer `Feedback`:
$$\text{Poor} \longrightarrow \text{Average} \longrightarrow \text{Good} \longrightarrow \text{Excellent}$$

Or customer `Tier`:
$$\text{Bronze} \longrightarrow \text{Silver} \longrightarrow \text{Gold} \longrightarrow \text{Platinum}$$

Here, there is an unmistakable, universally understood **rank and progression**:
- Postgraduate is higher than Undergraduate.
- Excellent is better than Good, which is better than Poor.

> **Proper Definition:**  
> **Ordinal data** is categorical data whose categories have a **meaningful, ordered, and hierarchical relationship**.

---

## 3. Why the Distinction Dictates the Encoding Method

> **The Golden Principle of Categorical Encoding:**  
> **We cannot assign numbers arbitrarily.** The numerical representation must preserve the true nature of the data:
> - If the data has an order, the numbers **must reflect that order**.
> - If the data has no order, the numbers **must NOT create a fake order**.

---

## 4. Ordinal Encoding

### Understand It First
When categories possess a natural sequence, we assign consecutive integers that preserve that exact hierarchy:

$$\text{High School} \longrightarrow 0$$
$$\text{Undergraduate} \longrightarrow 1$$
$$\text{Postgraduate} \longrightarrow 2$$
$$\text{Doctorate} \longrightarrow 3$$

Because $3 > 2 > 1 > 0$, an algorithm can recognize that someone with a Doctorate has a higher educational qualification than someone with a High School diploma.

> **Proper Definition:**  
> **Ordinal Encoding** converts ordinal categorical values into numerical integers according to their predefined, meaningful rank order.

---

## 5. Why Explicit Order Matters in Ordinal Encoding

Look at what happens if you assign numbers arbitrarily to customer ratings:

| Rating | Correct Ordinal Mapping | Arbitrary / Scrambled Mapping |
| :--- | :---: | :---: |
| **Poor** | **0** | **3** |
| **Average** | **1** | **0** |
| **Good** | **2** | **2** |
| **Excellent** | **3** | **1** |

In the scrambled mapping:
- The algorithm sees $\text{Poor} = 3$ and $\text{Excellent} = 1$.
- It concludes that **Poor is three times higher than Excellent**!
- Any model trained on this will learn completely backwards patterns.

> **Key Rule:**  
> With ordinal data, **you must explicitly specify the category order** to the encoder. Never let an algorithm guess the hierarchy alphabetically!

---

## 6. OrdinalEncoder in Scikit-Learn

In Scikit-Learn, we use `OrdinalEncoder` and pass the explicit order using the `categories` parameter:

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame(
    {
        "Education": [
            "High School",
            "Undergraduate",
            "Postgraduate",
            "Undergraduate",
        ]
    }
)

# Explicitly specify the lowest-to-highest hierarchy
education_order = [["High School", "Undergraduate", "Postgraduate"]]

encoder = OrdinalEncoder(categories=education_order)
df["Education_encoded"] = encoder.fit_transform(df[["Education"]])
```

### Why the `categories` Parameter is Essential:
By default, Scikit-Learn sorts categories alphabetically:
`["High School", "Postgraduate", "Undergraduate"]`
Alphabetically, Postgraduate comes before Undergraduate! Passing `categories=[...]` forces the encoder to use the real-world domain hierarchy instead of dictionary sorting.

---

## 7. Label Encoding

### Understand It First
Beginners often confuse `LabelEncoder` with `OrdinalEncoder`. 

`LabelEncoder` was designed for one specific purpose: **encoding the target variable ($y$) in classification problems**, not the input features ($X$).

### Example: Target Classification
Suppose you are predicting loan approvals:
$$\text{Denied} \longrightarrow 0$$
$$\text{Approved} \longrightarrow 1$$

Or medical diagnosis:
$$\text{Benign} \longrightarrow 0$$
$$\text{Malignant} \longrightarrow 1$$

> **Proper Definition:**  
> **Label Encoding** converts categorical target labels ($y$) into integer class identifiers ($0, 1, 2, \dots, K-1$).

```python
from sklearn.preprocessing import LabelEncoder

y = pd.Series(["Denied", "Approved", "Approved", "Denied"])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(y_encoded)  # Output: [1, 0, 0, 1]
print(label_encoder.classes_)  # Output: ['Approved', 'Denied']
```

---

## 8. Critical Comparison: Ordinal Encoding vs. Label Encoding

| Dimension | Ordinal Encoding (`OrdinalEncoder`) | Label Encoding (`LabelEncoder`) |
| :--- | :--- | :--- |
| **Intended Target** | **Input Features ($X$)** (multi-column 2D array) | **Target Variable ($y$)** (single 1D array) |
| **Input Shape** | 2D: `df[['col1', 'col2']]` | 1D: `df['target']` or `Series` |
| **Data Nature** | **Ordinal categories** with meaningful hierarchy | **Class labels** for classification |
| **Order Control** | User **explicitly defines** the sequence | Encoder assigns integers automatically (alphabetical) |
| **Meaning of Numbers** | Quantitative rank ($2 > 1 > 0$ carries meaning) | Pure identity label ($0$ and $1$ are just category IDs) |

---

## 9. Why You Must NEVER Use LabelEncoder on Nominal X Features

This is one of the most common mistakes in machine learning.

Suppose you have a nominal input feature: `City` with values `Chennai`, `Delhi`, `Mumbai`.
A beginner uses `LabelEncoder` on `X['City']` because it is quick:
$$\text{Chennai} \longrightarrow 0$$
$$\text{Delhi} \longrightarrow 1$$
$$\text{Mumbai} \longrightarrow 2$$

### The Hidden Catastrophe:
To a machine learning model:
$$2 > 1 > 0 \implies \text{Mumbai} > \text{Delhi} > \text{Chennai}$$
$$\text{Mumbai} - \text{Delhi} = 2 - 1 = 1$$
$$\text{Mumbai} - \text{Chennai} = 2 - 0 = 2$$

The model mathematically assumes:
1. **Mumbai has twice the magnitude of Delhi.**
2. **Mumbai is further away from Chennai than from Delhi.**
3. **$\text{Chennai} + \text{Delhi} \approx \text{Delhi}$.**

None of these relationships are true in the real world! They are purely an artifact of arbitrary alphabetical labeling. Linear Regression, KNN, SVM, and Neural Networks will be corrupted by these fake linear distances.

> **Golden Rule:**  
> **Never use `LabelEncoder` for nominal input features ($X$).**  
> For nominal features, you must use **One-Hot Encoding**.

---

## 10. One-Hot Encoding

### Understand It First
How do we encode nominal categories (like `Delhi`, `Mumbai`, `Chennai`) without creating a false ordering?

Instead of squeezing all categories into a single column, we give **each category its own separate binary column**:

| City | City_Chennai | City_Delhi | City_Mumbai |
| :--- | :---: | :---: | :---: |
| **Delhi** | 0 | **1** | 0 |
| **Mumbai** | 0 | 0 | **1** |
| **Chennai** | **1** | 0 | 0 |

### Why This Completely Solves the Problem:
1. Every category is represented by a binary switch ($1 = \text{Present}, 0 = \text{Absent}$).
2. Every category coordinate is equidistant from every other category:
   $$\text{Distance}(\text{Delhi}, \text{Mumbai}) = \sqrt{(0-0)^2 + (1-0)^2 + (0-1)^2} = \sqrt{2} \approx 1.414$$
   $$\text{Distance}(\text{Delhi}, \text{Chennai}) = \sqrt{(0-1)^2 + (1-0)^2 + (0-0)^2} = \sqrt{2} \approx 1.414$$
3. There is **zero artificial ranking**. No city is greater than another.

> **Proper Definition:**  
> **One-Hot Encoding** represents each distinct category as a separate binary feature (dummy variable), where $1$ indicates the presence of that category and $0$ indicates its absence.

---

## 11. OneHotEncoder in Scikit-Learn

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({"City": ["Delhi", "Mumbai", "Chennai", "Delhi"]})

# Initialize encoder with dense array output
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform
city_encoded = ohe.fit_transform(df[["City"]])

# Extract clean column names
column_names = ohe.get_feature_names_out(["City"])

# Build readable DataFrame
encoded_df = pd.DataFrame(city_encoded, columns=column_names)
```

Result:
```text
   City_Chennai  City_Delhi  City_Mumbai
0           0.0         1.0          0.0
1           0.0         0.0          1.0
2           1.0         0.0          0.0
3           0.0         1.0          0.0
```

---

## 12. The Practical Decision Tree

When you encounter any categorical variable in a dataset, use this decision framework:

```text
               Is it an Input Feature (X) or Target Label (y)?
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
    Input Feature (X)                                   Target Label (y)
            │                                                   │
  Does it have meaningful order?                                ▼
       ┌────┴────┐                                        LabelEncoder
       ▼         ▼
      YES        NO
       ▼         ▼
    Ordinal   Nominal
       ▼         ▼
OrdinalEncoder OneHotEncoder
```

### Examples:
- **`Customer Rating` (Low, Medium, High):** Input feature, meaningful order $\longrightarrow$ **OrdinalEncoder**.
- **`Marital Status` (Single, Married, Divorced):** Input feature, no order $\longrightarrow$ **OneHotEncoder**.
- **`State` (CA, NY, TX, FL):** Input feature, no order $\longrightarrow$ **OneHotEncoder**.
- **`Churned` (True, False):** Target label $y$ $\longrightarrow$ **LabelEncoder**.

---

## 13. Train/Test Split and Preventing Data Leakage

Just as with feature scaling, **encoding must respect the train/test split**.

```text
Full Dataset
     ↓
Train / Test Split  <--- Split FIRST!
     ↓
Fit Encoder on X_train ONLY  <--- Learn unique categories from training data
     ↓
Transform X_train using training encoder
     ↓
Transform X_test using the SAME training encoder
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1. Split raw data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Fit encoder ONLY on training data
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_train_encoded = encoder.fit_transform(X_train[["City"]])

# 3. Transform test data using the fitted encoder
X_test_encoded = encoder.transform(X_test[["City"]])
```

### Why `handle_unknown='ignore'` Matters:
What if the test set contains a new city (e.g. `"Bangalore"`) that never appeared in the training set?
- By default, Scikit-Learn will raise an error.
- Setting `handle_unknown='ignore'` tells the encoder to safely encode unseen test categories as all zeros (`[0, 0, 0]`), preventing runtime crashes in production!

---

## 14. Real-World Case Study: The Titanic Dataset

In the Titanic dataset (`data/titanic.csv`), we find three distinct categorical variables:

| Column | Type | Values | Why We Choose Its Encoder |
| :--- | :--- | :--- | :--- |
| **`Sex`** | Nominal Input ($X$) | `male`, `female` | No hierarchy $\longrightarrow$ **OneHotEncoder** (or binary dummy). |
| **`Embarked`** | Nominal Input ($X$) | `S`, `C`, `Q` | Ports have no intrinsic rank $\longrightarrow$ **OneHotEncoder**. |
| **`Pclass`** | Ordinal Input ($X$) | `1`, `2`, `3` | Ticket class has a clear hierarchy: 1st Class > 2nd Class > 3rd Class $\longrightarrow$ **OrdinalEncoder**. |
| **`Survived`** | Target Label ($y$) | `0`, `1` | Classification outcome $\longrightarrow$ Already binary encoded. |

---

## 15. Final Comparison Table

| Data Type | Example Features | Recommended Encoder | Reason |
| :--- | :--- | :--- | :--- |
| **Nominal Input ($X$)** | `City`, `Gender`, `Color`, `Embarked` | **`OneHotEncoder`** | Prevents false numerical hierarchy; keeps categories equidistant. |
| **Ordinal Input ($X$)** | `Education`, `Customer Rating`, `Pclass` | **`OrdinalEncoder`** | Preserves true rank order via user-defined `categories=[...]`. |
| **Target Label ($y$)** | `Survived`, `Churn`, `Disease Diagnosis` | **`LabelEncoder`** | Maps text classes into integer identifiers ($0, 1, \dots$). |

> **Final Note:**  
> These are established best practices, not rigid dogmas. For high-cardinality nominal variables (e.g. 5,000 ZIP codes), One-Hot Encoding creates thousands of sparse columns. In such advanced scenarios, techniques like Target Encoding or Frequency Encoding are used. But for standard nominal features, One-Hot Encoding is the undisputed gold standard.
