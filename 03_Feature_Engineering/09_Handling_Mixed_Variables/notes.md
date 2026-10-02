# Module 03: Feature Engineering — Handling Mixed Variables

In real-world data science, raw datasets rarely arrive with features neatly partitioned into pure numerical measurements or clean categorical labels.

Frequently, a single column packages **multiple disparate types of information together into a single string**. These are called **Mixed Variables**.

In this module, we study how to diagnose, unpack, and transform mixed variables into clean, model-ready numerical and categorical components.

---

## 1. What are Mixed Variables?

### The Core Concept

A **mixed variable** is a feature where each observation contains **two or more distinct pieces of information** bundled together—most commonly a mixture of:
- **Categorical information** (letters, codes, prefixes, abbreviations)
- **Numerical information** (serial numbers, room numbers, counts, years)

### Classic Real-World Examples:

| Column | Example Values | What is Mixed Inside? |
| :--- | :--- | :--- |
| **`Cabin`** (Titanic) | `C85`, `E46`, `B78` | Letter `C` (Deck section) + Number `85` (Cabin room number) |
| **`Ticket`** (Titanic) | `A/5 21171`, `PC 17599` | Text `A/5` (Ticketing agency / class) + Number `21171` (Serial number) |
| **Product SKU** | `ELEC-2024-998` | Department (`ELEC`) + Release Year (`2024`) + Item ID (`998`) |
| **Vehicle Registration** | `MH-12-DE-1433` | State (`MH`) + District (`12`) + Series (`DE`) + Vehicle ID (`1433`) |
| **Flight Code** | `BA0178` | Airline Code (`BA` = British Airways) + Flight Route (`178`) |

---

## 2. Why Mixed Variables are a Problem in Machine Learning

Machine learning algorithms are mathematical functions. They cannot naturally parse human string conventions.

If you leave a mixed column as a raw string and feed it into an algorithm, you face two catastrophic issues:

### 1. High Cardinality & Dimensionality Explosion
Consider the `Ticket` column on the Titanic:
- Total passengers: **891**
- Unique ticket strings: **681**

If you apply One-Hot Encoding directly to `Ticket`:
- Your feature matrix explodes by **681 sparse binary columns**!
- Almost every ticket column has a frequency of only $1$ or $2$ passengers.
- The model memorizes individual rows rather than generalizable patterns, causing severe **overfitting**.

### 2. Destruction of Latent Domain Signals
When treated as an atomic string:
- `C85` and `C123` are treated as two completely distinct, unrelated categories.
- The algorithm has **zero awareness** that both passengers were located on **Deck C**!
- On a sinking ship, deck location directly dictated physical distance to lifeboats and evacuation priority. Packaging the deck together with the room number blinds the model to this critical signal.

```text
Without Extraction:
"C85"  ──► Category 1 (Independent token)
"C123" ──► Category 2 (Completely unrelated token)
(The model learns NO shared Deck C relationship!)

With Component Extraction:
"C85"  ──► Deck: 'C'  | Room: 85
"C123" ──► Deck: 'C'  | Room: 123
(The model easily learns that Deck 'C' passengers shared high survival odds!)
```

---

## 3. The Main Strategy: Splitting the Column

The foundational strategy for handling mixed variables is **Component Extraction**:

```text
                  Raw Mixed Column
                         │
                         ▼
             [ Identify Components ]
                         │
                         ▼
             [ Extraction / Splitting ]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
Categorical Component             Numerical Component
(e.g., Cabin_Deck = 'C')          (e.g., Cabin_Number = 85)
        │                                 │
        ▼                                 ▼
Handle Missing Categories         Handle Missing Numbers
('Missing' / 'Unknown')           (Contextual Imputation)
        │                                 │
        ▼                                 ▼
Categorical Encoding              Feature Scaling
(OneHotEncoder / Ordinal)         (StandardScaler / RobustScaler)
        │                                 │
        └────────────────┬────────────────┘
                         │
                         ▼
             Clean Model Feature Matrix
```

### Tabular Transformation Example:

| Raw `Cabin` | $\implies$ | `Cabin_Deck` (Categorical) | `Cabin_Number` (Numerical) |
| :--- | :---: | :---: | :---: |
| `C85` | $\implies$ | `C` | `85` |
| `C123` | $\implies$ | `C` | `123` |
| `E46` | $\implies$ | `E` | `46` |
| `B78` | $\implies$ | `B` | `78` |
| `NaN` | $\implies$ | `Missing` | `0` (or imputed) |

Now we have two pure, well-behaved features that can be preprocessed using our standard toolkit (`OneHotEncoder`, `StandardScaler`).

---

## 4. Why Splitting Works: Information Unpacking

By separating the components, we transform unstructured text into structured domain features:

1. **`Cabin_Deck`**:
   - Compresses 147 unique cabin strings into just **8 deck levels**: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `T`.
   - Dramatically reduces cardinality while retaining the primary physical driver of survival.
2. **`Cabin_Number`**:
   - Isolates the numerical room position along the ship's corridor.
   - Even/Odd numbers on ships often denote port (left) vs. starboard (right) side cabins!

---

## 5. Handling Missing Values After Splitting

Splitting mixed variables frequently reveals or propagates missing values. For instance, in the Titanic dataset, over 77% of passengers had `Cabin = NaN`.

When we split `NaN`, both extracted features are null:
```text
Cabin = NaN  ──►  Cabin_Deck = NaN, Cabin_Number = NaN
```

We must handle each component according to its statistical and physical meaning:

### A. The Categorical Component (`Cabin_Deck`)
- **Best Practice**: Treat missingness as an explicit, informative category:
  ```python
  df['Cabin_Deck'] = df['Cabin_Deck'].fillna('Missing')
  ```
- **Why?** Missing cabin records were not random data entry omissions; 3rd class passengers were simply not assigned private numbered cabins. The fact that the cabin is missing is itself a powerful predictive signal!

### B. The Numerical Component (`Cabin_Number`)
- **Option 1: Meaningful Constant (0)**:
  - Setting `Cabin_Number = 0` can be valid **if and only if** `0` has a distinct domain interpretation (e.g. `0` explicitly represents "no cabin assigned").
- **Option 2: Median / Mean Imputation with Missing Indicator**:
  - Impute the missing numbers with the median room number and add a binary indicator column:
  ```python
  df['Cabin_Number_Imputed'] = df['Cabin_Number'].fillna(median_val)
  df['Cabin_Number_Was_Missing'] = df['Cabin_Number'].isnull().astype(int)
  ```

> **The Golden Imputation Rule:**  
> Never automatically replace missing numerical values with `0` unless `0` carries an authentic physical meaning in your problem domain.

---

## 6. Regular Expressions (Regex) as an Extraction Powerhouse

When mixed values have clean, consistent structures (like `C85`), simple string indexing (`x[0]` and `x[1:]`) works.

However, real-world strings are often messy, inconsistent, and irregular:
```text
A/5 21171
PC 17599
STON/O2. 3101282
349909
```
Notice the challenges:
- Some tickets have prefixes with slashes (`/`) and dots (`.`).
- Some tickets have multiple spaces.
- Some tickets have **no prefix at all** and consist purely of numbers (`349909`).

Fixed string indexing breaks down on irregular data. This is where **Regular Expressions (Regex)** become essential.

### Core Regex Cheat Sheet for Feature Engineering:

| Token | Meaning | What It Matches | Example |
| :--- | :--- | :--- | :--- |
| `\d` | Any digit (0-9) | `'5'`, `'9'` | `\d` matches `8` in `"C85"` |
| `\d+` | One or more digits | `'85'`, `'21171'` | Extracts entire multi-digit numbers |
| `[a-zA-Z]` | Any alphabet letter | `'C'`, `'p'` | Matches individual letters |
| `[a-zA-Z]+` | One or more letters | `'PC'`, `'STON'` | Matches whole text prefixes |
| `\s` | Whitespace character | Spaces, tabs | Matches space between prefix and number |
| `^` | Start of the string | Position anchor | `^[A-Z]` matches a capital letter at the start |
| `$` | End of the string | Position anchor | `\d+$` matches digits at the very end |
| `()` | **Capture Group** | Isolates the exact substring you want to extract | `r'([A-Za-z]+)'` returns only the matched letters |

---

## 7. Practical Regex Patterns for Mixed Variables

### Pattern 1: Extracting Leading Letter and Trailing Number (`Cabin = C85`)
```python
import re

# Extracts: group 1 = Deck letter ('C'), group 2 = Room number ('85')
pattern = r'([A-Za-z])(\d+)'
```

### Pattern 2: Extracting Trailing Number from Inconsistent String (`Ticket`)
In tickets like `"A/5 21171"` or `"349909"`, the serial number is always the digits at the end:
```python
# Extracts the final contiguous group of digits
number_pattern = r'(\d+)$'
```

### Pattern 3: Extracting Prefix from Inconsistent String (`Ticket`)
The prefix is everything preceding the final number:
```python
# Extracts all text before the final digits, stripped of extra spaces/dots
prefix_pattern = r'^(.*?)\s*\d+$'
```

---

## 8. Summary Comparison: Raw String vs. Extracted Features

| Aspect | Raw Mixed Variable (`Ticket`, `Cabin`) | After Feature Extraction |
| :--- | :--- | :--- |
| **Data Type** | Unstructured `object` string | Clean categorical + Clean numerical |
| **Cardinality** | Hundreds of unique levels (e.g., 681 tickets) | Low cardinality (8 decks, ~20 prefixes) |
| **Model Ingestion** | Requires massive, sparse one-hot matrices | Compact, dense, and interpretable |
| **Generalization** | High risk of overfitting on rare IDs | Strong generalization across shared prefixes |
| **Missingness** | Nulls hide whether category or number was absent | Explicit categorical bucket (`'Missing'`) |

---

## 9. Complete End-to-End Workflow Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   MIXED VARIABLE PROCESSING PIPELINE                   │
└────────────────────────────────────────────────────────────────────────┘
                               │
                      Raw Tabular Dataset
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
      Raw 'Cabin'                            Raw 'Ticket'
     ['C85', NaN]                      ['A/5 21171', '349909']
            │                                     │
            ▼                                     ▼
     Regex Extraction                      Regex Extraction
  r'([A-Za-z])' & r'(\d+)'              r'^(.*?)\s*\d+$' & r'(\d+)$'
            │                                     │
   ┌────────┴────────┐                   ┌────────┴────────┐
   ▼                 ▼                   ▼                 ▼
Cabin_Deck      Cabin_Number        Ticket_Prefix     Ticket_Num
  ['C']             [85]               ['A/5']         [21171]
   [NaN]           [NaN]              ['NONE']         [349909]
   │                 │                   │                 │
   ▼                 ▼                   ▼                 ▼
Categorical:      Numerical:          Categorical:      Numerical:
Fill 'Missing'   Fill 0/Median       Fill 'NONE'       Scale / Log
   │                 │                   │                 │
   ▼                 ▼                   ▼                 ▼
OneHotEncoder    StandardScaler      OneHotEncoder     StandardScaler
   │                 │                   │                 │
   └─────────────────┴─────────┬─────────┴─────────────────┘
                               │
                               ▼
                   [ ColumnTransformer ]
                               │
                               ▼
                   [ Final Estimator (Model) ]
```

---

## 10. Important Points & Common Pitfalls

### Critical Guidelines:
1. **Don't automatically treat the whole string as categorical**: High-cardinality strings dilute your model's capacity and encourage overfitting.
2. **Context dictates numerical imputation**: Only fill missing numbers with `0` if zero conveys "absence of attribute" (e.g. no cabin). Otherwise, use median imputation.
3. **Use Regex for messy real-world strings**: Plain `.split(' ')` crashes when values lack spaces or contain multiple delimiters.
4. **Always standardize text prefixes**: Convert prefixes to uppercase and strip punctuation (e.g., `'A/5.'` and `'A/5'` both represent the same agency `'A/5'`).

---

## Quick Revision Table

| Concept | Meaning | Practical Implementation |
| :--- | :--- | :--- |
| **Mixed Variable** | A column combining distinct information types | `C85` (Deck + Room), `A/5 21171` (Prefix + Serial) |
| **Extraction** | Unpacking components into separate columns | `str.extract(r'([A-Za-z])')`, `str.extract(r'(\d+)')` |
| **Prefix Handling** | Categorical component extraction | Low-cardinality nominal encoding (`OneHotEncoder`) |
| **Number Handling** | Continuous component extraction | Convert with `pd.to_numeric()`, scale with `StandardScaler` |
| **Missing Treatment** | Contextual null resolution | Categorical $\to$ `'Missing'`; Numerical $\to$ Median / 0 |
| **Regex Power** | Pattern-based extraction engine | `\d+` (digits), `[A-Za-z]+` (letters), `()` (capture) |

---

### 🧠 Final Mental Model

> **Raw String $\implies$ Identify Sub-Signals $\implies$ Regex Extraction $\implies$ Categorical + Numerical Channels $\implies$ Contextual Imputation $\implies$ Encode / Scale $\implies$ Model**
