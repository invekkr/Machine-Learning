# Module 03: Feature Engineering — Advanced Missing Data Techniques

In previous topics, we explored the foundations of handling missing data:
- **Deletion**: [Complete Case Analysis (CCA)](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/10_Complete_Case_Analysis/) (deleting incomplete rows).
- **Simple Numerical Imputation**: [Mean, Median, Arbitrary, and End-of-Distribution](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/11_Handling_Missing_Numerical_Data/).
- **Simple Categorical Imputation**: [Mode and Missing Category](file:///Users/shamvi/stuff/ML/03_Feature_Engineering/12_Handling_Missing_Categorical_Data/).

Now we take the next step and address three critical questions:
1. **How can we fill missing values without shrinking the natural spread of our data?** $\implies$ **Random Sample Imputation**
2. **How can we prevent the model from forgetting that a value was originally missing?** $\implies$ **Missing Indicator**
3. **How do we scientifically choose the best imputation strategy instead of guessing?** $\implies$ **GridSearchCV Imputation Tuning**

---

## 1. Random Sample Imputation

### The Core Idea
Instead of calculating a mathematical summary (like the mean or median), we **randomly select an existing, observed value from the same column** and use it to replace the missing entry.

```text
Raw Column:             [ 20, 25, 30, 35, NaN ]
Observed Values:        { 20, 25, 30, 35 }
Randomly Pick One:      Suppose we randomly draw 30
Imputed Result:         [ 20, 25, 30, 35, 30 ]
```

Every missing row receives an authentic, real-world value drawn directly from the feature's empirical distribution.

---

## 2. Why Not Just Use the Mean or Median?

Consider a feature with 4 observed values and 3 missing values:
$$[20, \ 25, \ 30, \ 35, \ \text{NaN}, \ \text{NaN}, \ \text{NaN}]$$

### If You Use Mean Imputation:
- Mean: $\frac{20 + 25 + 30 + 35}{4} = 27.5$
- Result: $[20, \ 25, \ 30, \ 35, \ \mathbf{27.5}, \ \mathbf{27.5}, \ \mathbf{27.5}]$
- **The Issue**: You have created **3 completely identical values**. The data artificially bunches up at $27.5$, creating an unnatural spike and shrinking the feature's variance.

### If You Use Random Sample Imputation:
- We randomly draw 3 values from $\{20, 25, 30, 35\}$: say, $30, 20, 35$.
- Result: $[20, \ 25, \ 30, \ 35, \ \mathbf{30}, \ \mathbf{20}, \ \mathbf{35}]$
- **The Benefit**: The imputed numbers look like natural, genuine measurements. The empirical spread is preserved!

```text
Original Distribution Spread:
20           25           30           35

After Mean Imputation (Artificially Bunched):
20           25   [27.5  27.5  27.5]   30           35

After Random Sample Imputation (Preserves Spread):
[20  20]     25          [30   30]     [35   35]
```

---

## 3. What Does Random Sample Imputation Preserve? (And What Does It Break?)

### What It Preserves: Univariate Spread & Variance
Because replacement values are sampled directly from observed observations, the feature's **mean, variance, and overall distribution shape remain virtually unchanged**:

$$\text{Variance}_{\text{imputed}} \approx \text{Variance}_{\text{original}}$$

---

### The Big Catch: It Can Disturb Relationships Between Columns
While Random Sample Imputation is wonderful for a **single column in isolation**, it ignores all other columns in that row:

| Passenger / Employee | Age | Salary (₹) | Problem |
| :---: | :---: | :---: | :--- |
| 1 | 20 | 30,000 | Realistic |
| 2 | 25 | 35,000 | Realistic |
| 3 | 30 | 50,000 | Realistic |
| 4 | 35 | 60,000 | Realistic |
| 5 | **NaN** | **70,000** | Senior earner missing Age |

If we randomly sample from $\{20, 25, 30, 35\}$ and accidentally draw `20`, we create:
$$\text{Age} = 20, \quad \text{Salary} = ₹70,000$$

A 20-year-old junior entry-level hire is now paired with an executive ₹70k salary! This distorts the covariance and correlation between `Age` and `Salary`.

> [!IMPORTANT]
> **Key Rule of Random Sample Imputation:**  
> It preserves the internal spread of the **single column**, but can weaken or distort the **relationships between columns**.

---

## 4. Missing Indicator: Preserving the "Missing" Signal

Now let's examine a completely different, complementary concept.

### The Fundamental Flaw of Simple Imputation
Suppose we have:
$$[20, \ 25, \ \text{NaN}, \ 30]$$

We fill `NaN` with the median ($25$):
$$[20, \ 25, \ \mathbf{25}, \ 30]$$

The algorithm now sees two rows with $\text{Age} = 25$:
- Row 1: The passenger was **genuinely 25 years old**.
- Row 2: The passenger's age was **unknown**, and we guessed 25.

The model cannot tell the difference! We have erased the information that the value was absent.

---

### The Solution: An Extra Binary Flag Column
A **Missing Indicator** adds a companion binary column indicating whether the original value was present or absent:

| Original Age | Imputed Age | `Age_Missing` (Indicator) | Meaning |
| :---: | :---: | :---: | :--- |
| 20 | 20 | **0** | Original value was available |
| 25 | 25 | **0** | Original value was available |
| **NaN** | 25 | **1** | **Value was missing and imputed!** |
| 30 | 30 | **0** | Original value was available |

Now the machine learning model has full clarity:
- $\text{Age} = 25, \ \text{Age\_Missing} = 0 \implies \text{True 25-year-old}.$
- $\text{Age} = 25, \ \text{Age\_Missing} = 1 \implies \text{Missing value estimated with median}.$

---

### Why Does Missingness Matter? (The Informative Missingness Signal)
In many real-world domains, **missingness is not an accident** — it reflects user behavior or business status:

```text
Loan Applicant Table:
Income: ₹50k,  Default: No
Income: ₹60k,  Default: No
Income: NaN,   Default: YES  <-- Applicant hid income because of distress!
Income: ₹70k,  Default: No
Income: NaN,   Default: YES  <-- Applicant hid income because of distress!
```

If you simply replace `NaN` with the median income ($₹60\text{k}$), you hide the delinquency signal.  
With an indicator (`Income_Missing = 1`), the algorithm can easily learn:
$$\text{If Income\_Missing} = 1 \implies \text{Higher Default Risk}$$

> [!NOTE]
> **Crucial Clarification:**  
> A Missing Indicator **does NOT fill the missing value**.  
> It only records the flag ($0/1$). It must be paired with an imputation method (e.g., Median Imputation + Missing Indicator).

---

## 5. Automated Imputation Tuning with GridSearchCV

In practice, data scientists often wonder:
- *"Should I use Mean or Median?"*
- *"Should I add a Missing Indicator (`True` vs. `False`)?"*

Instead of guessing, we can make Scikit-Learn **test all combinations systematically using Cross-Validation**.

---

### What is GridSearchCV?
**GridSearchCV** (Grid Search Cross-Validation) tests a "grid" of different preprocessing and model settings, trains each on validation folds, and reports which setting achieved the highest validation score.

```text
                     GridSearchCV Pipeline
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       Option A: Mean                  Option B: Median
               │                               │
               ▼                               ▼
      Train on 5 Folds                Train on 5 Folds
               │                               │
               ▼                               ▼
        Accuracy: 78.2%                 Accuracy: 80.5%
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                   WINNER: Median (80.5%)
```

---

### Step-by-Step Code Structure

#### 1. Define the Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('classifier', RandomForestClassifier(random_state=42)),
])
```

#### 2. Define the Parameter Grid:
Notice the double underscore (`__`) syntax connecting the step name to its parameter:
```python
param_grid = {
    # Test both mean and median
    'imputer__strategy': ['mean', 'median'],
    # Test with and without the missing indicator flag!
    'imputer__add_indicator': [True, False],
}
```

#### 3. Run GridSearchCV with 5-Fold Cross-Validation:
```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Inspect best combination
print("Best Strategy:", grid.best_params_)
print(f"Best CV Accuracy: {grid.best_score_ * 100:.2f}%")
```

---

## 6. What Do the Key Parameters Mean?

### What Does `'imputer__strategy'` Mean?
In Scikit-Learn pipelines, double underscores (`__`) tell `GridSearchCV` which step inside the pipeline to adjust:
$$\underbrace{\text{imputer}}_{\text{Step Name}} \ \text{\_\_} \ \underbrace{\text{strategy}}_{\text{Parameter of SimpleImputer}}$$

### What Does `cv=5` Mean?
**5-fold cross-validation**:
The training data is split into 5 equal subsets (folds). The pipeline is trained on 4 folds and tested on the 5th fold. This is repeated 5 times so every fold acts as a validation test once. The average of all 5 runs determines the score.

---

## 7. Big-Picture Summary: When to Use What?

| Scenario | Recommended Approach | Why? |
| :--- | :--- | :--- |
| **Small missingness ($< 5\%$) in symmetric data** | **Mean Imputation** | Simple, fast, and does not alter the center. |
| **Data has outliers or is skewed** | **Median Imputation** | Robust against extreme values. |
| **Want to preserve the spread / variance** | **Random Sample Imputation** | Replaces nulls with genuine observed values. |
| **Missingness itself carries predictive meaning** | **Missing Indicator (`add_indicator=True`)** | Tells the model which values were estimated. |
| **Unsure which method works best** | **GridSearchCV** | Lets cross-validation find the empirical winner. |
| **Categorical data with small nulls** | **Most Frequent (Mode)** | Fills with the dominant class. |
| **Categorical data with large nulls ($> 10\%$)** | **`'Missing'` Category** | Prevents mode inflation and preserves signal. |

---

## 8. Common Mistakes to Avoid

1. **Forgetting `random_state` in Random Sample Imputation**:
   Without a fixed seed, your imputed values change every time you run the code, producing unstable model scores.
2. **Thinking a Missing Indicator replaces the missing value**:
   Remember: An indicator only creates a `0/1` column. You must still fill the `NaN` values in the original column!
3. **Running GridSearchCV on the test set**:
   Always fit `GridSearchCV` strictly on `X_train`. Use `X_test` only for final unbiased evaluation.
4. **Blindly trusting Random Sample with correlated features**:
   Remember that random sampling can create unrealistic combinations (like a 20-year-old CEO).

---

## 9. Final Mental Model

```text
                        Missing Value Encountered (NaN)
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        "How do I fill it?"    "Does missingness     "Which method
                                  carry signal?"        is best?"
                │                      │                      │
                ▼                      ▼                      ▼
       Mean / Median /         Add a Missing             Use
        Random Sample            Indicator           GridSearchCV
        (Replace NaN)          (0/1 Flag)            (Tune Choice)
```
