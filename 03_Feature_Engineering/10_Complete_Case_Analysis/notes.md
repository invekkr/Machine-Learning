# Module 03: Feature Engineering — Complete Case Analysis (CCA)

In real-world data science, missing values are unavoidable. Sensors malfunction, survey respondents skip questions, records are lost in transmission, and legacy databases contain unrecorded fields.

Before diving into complex mathematical imputation techniques (like Mean/Median, KNN, or MICE), we must study the most straightforward, intuitive, and widely used initial baseline: **Complete Case Analysis (CCA)**, also known as **Listwise Deletion**.

---

## 1. What is Complete Case Analysis?

### The Core Concept

**Complete Case Analysis (CCA)** is a missing data handling technique where we **discard the entire row (observation) if it contains even one missing value** in the features being analyzed or trained.

Because we only retain rows with complete data across all required columns, it is formally known as **Listwise Deletion**.

```text
Raw Observation:  [ Age: 25,  Salary: 50,000,  City: 'Delhi' ]   ──► Retain row
Raw Observation:  [ Age: 30,  Salary: NaN,     City: 'Mumbai' ]  ──► DISCARD ENTIRE ROW
Raw Observation:  [ Age: NaN, Salary: 70,000,  City: 'Delhi' ]   ──► DISCARD ENTIRE ROW
Raw Observation:  [ Age: 28,  Salary: 60,000,  City: 'Gurgaon']  ──► Retain row
```

### Simple Tabular Example

Suppose we have an employee table:

| Row ID | Age | Salary (₹) | City | Status in CCA |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 25 | 50,000 | Delhi | Retained |
| 2 | 30 | **NaN** | Mumbai | **Deleted** (Salary is missing) |
| 3 | **NaN** | 70,000 | Delhi | **Deleted** (Age is missing) |
| 4 | 28 | 60,000 | Gurgaon | Retained |

#### Result After CCA:

| Row ID | Age | Salary (₹) | City |
| :---: | :---: | :---: | :---: |
| 1 | 25 | 50,000 | Delhi |
| 4 | 28 | 60,000 | Gurgaon |

Rows 2 and 3 have been completely removed. The remaining dataset contains **zero missing values**.

---

## 2. When Can We Use CCA? The MCAR Assumption

The foundational statistical assumption required to justify Complete Case Analysis is:

> ### **MCAR — Missing Completely At Random**

### What Does MCAR Mean?
In simple terms:
> The probability of a value being missing is **purely random** and has **no relationship** with any observed feature (like gender, class, or salary) or any unobserved feature.

### Intuitive Real-World Scenarios:
- **True MCAR (Safe for CCA):**  
  A laboratory technician accidentally drops and breaks 2 out of 1,000 test tubes. The breakage had nothing to do with what was inside the tube. Removing those 2 observations does not distort the experiment.
- **Not MCAR (Unsafe for CCA):**  
  In a voluntary income survey, high-earning executives systematically decline to disclose their salary. The missingness is directly correlated with wealth. Discarding rows with missing salaries deletes the highest-earning cohort, creating an artificially poor sample!

### Summary of the 3 Missing Data Mechanisms:

| Mechanism | Full Name | Intuitive Meaning | Can We Safely Use CCA? |
| :--- | :--- | :--- | :--- |
| **MCAR** | Missing Completely At Random | Missingness is pure chance. Nothing predicts whether data is missing. | **Yes** (if data loss is small) |
| **MAR** | Missing At Random | Missingness depends on *other observed features* (e.g. 3rd class passengers have more missing ages). | **Risky** (introduces selection bias) |
| **MNAR** | Missing Not At Random | Missingness depends directly on the *unobserved value itself* (e.g. high earners hiding income). | **Dangerous** (severely distorts distribution) |

---

## 3. Why Does MCAR Matter? Population Shift & Selection Bias

When missingness is **not** completely random, removing rows alters the population structure of your dataset:

```text
BEFORE CCA (Original Population):
┌───────────────────────────────┬───────────────────────────────┐
│ Young People (60%)            │ Older People (40%)            │
└───────────────────────────────┴───────────────────────────────┘

Suppose missing values occur disproportionately among older people.
If you apply CCA (delete all rows with missing values):

AFTER CCA (Distorted Sample):
┌───────────────────────────────────────────────┬───────────────┐
│ Young People (80%)                            │ Older (20%)   │
└───────────────────────────────────────────────┴───────────────┘
```

### The Statistical Consequence:
Your machine learning algorithm is no longer trained on a representative sample of reality. It trains on a **distorted, biased sub-population**:
- Predictions on older people will suffer from high variance and error.
- Feature relationships (e.g., correlation between age and purchase behavior) will be distorted.

> [!IMPORTANT]
> **The Golden Mental Model:**  
> **Random missingness (MCAR)** $\implies$ Removing rows retains representative sample proportions.  
> **Patterned missingness (MAR / MNAR)** $\implies$ Removing rows introduces severe selection bias.

---

## 4. How Much Missing Data? The 5% Rule of Thumb

A standard guideline taught across data science is:

> **Rule of Thumb:** Complete Case Analysis is generally acceptable when the total proportion of observations removed is small, often **under 5%**.

### Missing Data Ratio Formula:
$$\text{Missing Percentage} = \left(\frac{\text{Number of Rows Lost}}{\text{Total Rows in Dataset}}\right) \times 100$$

### Why 5% is a Rule of Thumb, NOT a Universal Law:
Do not treat 5% as a rigid mathematical cutoff. You must always balance **statistical power (sample size)** against **representativeness**:

```text
Case 1: Large Dataset, Low Missingness
100,000 total rows
  2,000 rows contain missing values (2% lost)
Remaining: 98,000 complete rows.
Verdict: CCA is completely acceptable if MCAR is plausible.

Case 2: Large Dataset, High Missingness
100,000 total rows
 40,000 rows contain missing values (40% lost)
Remaining: 60,000 rows.
Verdict: Discarding 40,000 observations wastes massive amounts of expensive training data.

Case 3: Small Dataset, Low Missingness
80 total rows
 4 rows contain missing values (5% lost)
Remaining: 76 rows.
Verdict: In tiny medical trials or small sample regimes, losing even 4 rows might reduce statistical power significantly!
```

---

## 5. What If a Column Has Extremely High Missingness?

Consider a dataset with three features:

| Feature | Missing Percentage | Action Consideration |
| :--- | :---: | :--- |
| **`Feature A`** | **0.2%** | Excellent candidate for CCA (drop the 0.2% rows). |
| **`Feature B`** | **3.5%** | Candidate for CCA if distribution is preserved. |
| **`Feature C`** | **85.0%** | **DO NOT drop 85% of rows!** Drop the *column*, or engineer an informative missing indicator. |

### Dropping the Column vs. Dropping the Row:
- If a column is missing in **>70–80%** of rows, applying CCA would delete 70–80% of your entire dataset!
- Instead of deleting the observations, evaluate whether `Feature C` can be:
  1. **Dropped entirely** as an uninformative feature.
  2. **Transformed into an indicator flag** (e.g., `Feature_C_Was_Present = 1/0`), as we did with the Titanic `Cabin` feature in [Module 03 Topic 09](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/09_Handling_Mixed_Variables/).

---

## 6. Advantages of Complete Case Analysis

| Advantage | Why It Matters |
| :--- | :--- |
| **1. Maximum Simplicity** | Conceptually obvious: no mathematical models, no distance metrics, and no complex algorithms. |
| **2. One-Line Implementation** | In pandas, it requires a single call: `df.dropna()`. |
| **3. Preserves Natural Relationships** | Unlike mean or median imputation, CCA never injects artificial constant numbers that distort variance or feature correlations. |
| **4. Valid Under Pure MCAR** | If missingness is truly random, the remaining sample is an unbiased representation of the original population. |

---

## 7. Disadvantages & Production Bottlenecks

### 1. Permanent Loss of Data & Statistical Power
Deleting rows throws away valid information contained in the *other* columns of that row. If an observation has 20 valid measurements and only 1 missing value, CCA discards all 20 valid measurements!

### 2. Risk of Introducing Severe Bias
If data is not MCAR, CCA skews population distributions, altering means, variances, and category proportions.

### 3. The Production / Inference Dilemma
> [!WARNING]
> **The Critical Production Flaw of CCA:**  
> In training, you can drop incomplete rows. But in a live production environment:  
> **A user submits a form with a missing optional field.**  
> If your model was trained only on complete rows without an imputation strategy, Scikit-Learn will throw:  
> `ValueError: Input contains NaN`  
> You cannot simply "drop" the customer! You must produce a prediction. Therefore, production pipelines almost always require an explicit imputation strategy.

---

## 8. Verifying Distribution Preservation (Before vs. After CCA)

Before approving CCA in your pipeline, you **must statistically verify** that dropping rows did not distort your data.

### Diagnostic Checklist:

```text
                 Raw Dataset (Before CCA)
                            │
                   [ Perform CCA ]
                            │
                 Clean Dataset (After CCA)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    Categorical Features            Numerical Features
            │                               │
            ▼                               ▼
 Compare Category Proportions       Compare Descriptive Stats:
 (Percentages of each class)        - Mean & Median
                                    - Standard Deviation & Variance
                                    - Min, Max, Quantiles (IQR)
                                    - Histogram / KDE Shape
```

### Categorical Distribution Check Example:
If `City` before CCA has Delhi (40%), Mumbai (35%), Gurgaon (25%), and after CCA has Delhi (41%), Mumbai (34%), Gurgaon (25%), the distribution is preserved $\implies$ **CCA is safe**.

If after CCA Delhi becomes 65% and Mumbai drops to 20%, selection bias was introduced $\implies$ **CCA is unsafe**.

---

## 9. Numerical Distribution Check Metrics

For continuous numerical variables (like `Age` or `Fare`), compare:

$$\text{Variance Ratio} = \frac{\text{Variance}_{\text{After}}}{\text{Variance}_{\text{Before}}}$$

- **Preserved Distribution:** $\text{Mean}_{\text{After}} \approx \text{Mean}_{\text{Before}}$ and $\text{Variance Ratio} \approx 1.0 \pm 0.05$.
- **Distorted Distribution:** Mean shifts significantly or variance shrinks/expands by $>10\%$.

---

## 10. Complete Case Analysis vs. Imputation

| Comparison Dimension | Complete Case Analysis (CCA) | Imputation (Mean, Median, KNN) |
| :--- | :--- | :--- |
| **Action on Rows** | Deletes rows containing nulls | Retains all original rows |
| **Dataset Size** | Decreases ($N_{\text{clean}} < N_{\text{raw}}$) | Unchanged ($N_{\text{clean}} = N_{\text{raw}}$) |
| **Complexity** | Extremely simple (`dropna()`) | Requires choosing strategy & fitting imputers |
| **Artificial Data** | Introduces **zero** synthetic values | Fills cells with synthetic constants or estimates |
| **Variance Impact** | Preserves true distribution shape if MCAR | Often artificially reduces feature variance |
| **Production Ready** | Fails if incoming inference row has nulls | Seamlessly handles incoming nulls at inference |
| **Primary Requirement** | MCAR + Small loss of rows ($<5\%$) | Suitable across MCAR, MAR, and MNAR |

---

## 11. Practical Implementation in Pandas

### Dropping All Rows With Any Missing Value:
```python
df_clean = df.dropna()
```

### Targeted Dropping on Specific Columns:
When only specific features have low, random missingness, use `subset`:
```python
# Drops rows ONLY if 'Embarked' is null, preserving rows where 'Cabin' is null
df_clean = df.dropna(subset=['Embarked'])
```

---

## 12. Important Corrections & Myths to Avoid

1. **Myth:** *"CCA is always prohibited if missingness exceeds 5%."*  
   **Reality:** 5% is an empirical guideline. If a dataset has 500,000 rows and 6% missingness that is proven MCAR, keeping 470,000 rows is statistically sound.
2. **Myth:** *"Imputation is always superior to CCA."*  
   **Reality:** Imputation replaces unknown reality with artificial estimates. When missingness is negligible (<0.5%) and MCAR, CCA is cleaner, faster, and avoids injecting synthetic noise.

---

## 13. Summary Revision & Final Mental Model

```text
                      Dataset With Missing Values
                                  │
                                  ▼
                    Evaluate Missing Percentage
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
      > 50–70%                                          < 5–10%
 (Feature is mostly empty)                         (Small missingness)
         │                                                 │
         ▼                                                 ▼
Drop Column / Create Missing Flag               Is Missingness MCAR?
                                                           │
                                        ┌──────────────────┴──────────────────┐
                                        ▼                                     ▼
                                    Yes (MCAR)                         No (MAR / MNAR)
                                        │                                     │
                                        ▼                                     ▼
                               Check Distributions                    Use Imputation
                                  Before vs After                     (Median, KNN)
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                    Preserved?                      Altered?
                         │                             │
                         ▼                             ▼
                   Safe for CCA!                Use Imputation!
```
