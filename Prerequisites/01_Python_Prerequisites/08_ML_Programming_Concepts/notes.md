# Python Prerequisites: 08. Core Machine Learning Programming Patterns

As you study different Machine Learning algorithms, you will notice that certain programming patterns and terminology repeat across every single codebase. Understanding these universal patterns will make reading and writing ML code intuitive.

---

## 1. Array Dimensions: The `(N, D)` Convention

In almost every data science paper, textbook, and library:
- **$N$**: The number of **samples** (data rows / instances).
- **$D$**: The number of **dimensions** or **features** (data columns).

```text
Feature Matrix X: Shape (N, D)
                  Columns (D Features)
                  ┌───────┬───────┬───────┐
         Row 0    │  x00  │  x01  │  x02  │
Rows     Row 1    │  x10  │  x11  │  x12  │
(N)      Row 2    │  x20  │  x21  │  x22  │
         Row 3    │  x30  │  x31  │  x32  │
                  └───────┴───────┴───────┘
Target Vector y:  Shape (N,) or (N, 1)
```

### The Scikit-Learn Dimensionality Rule:
- **`X` MUST be a 2-Dimensional array or DataFrame** of shape `(N, D)`. Even if you only have one single feature, it must be shaped as `(N, 1)`.
- **`y` is typically a 1-Dimensional vector** of shape `(N,)`.

---

## 2. Model Parameters vs. Hyperparameters

This is one of the most common points of confusion for beginners:

```text
                           MODEL CONFIGURATION
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
         PARAMETERS                               HYPERPARAMETERS
  (Learned by the computer)                  (Set manually by YOU)
  - Learned during model.fit()               - Set when instantiating the model
  - Computed from data                       - Governs how the model learns
  - Examples:                                - Examples:
    * Weights (w) & Bias (b) in Linear Reg     * n_neighbors in KNN
    * Decision split thresholds in Trees       * max_depth in Decision Trees
    * Cluster centroids in K-Means             * learning_rate in Gradient Descent
```

---

## 3. Reproducibility and `random_state`

Many algorithms involve stochastic (random) processes:
- `train_test_split` randomly shuffles rows.
- Random Forests randomly pick subsets of features.
- K-Means randomly initializes cluster centroids.

If you don't control the random generator, your code will produce slightly different accuracy scores every time you run it.

```python
# random_state acts as a fixed seed (e.g. 42)
# Any integer works; 42 is an industry-standard convention.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
Setting `random_state=42` guarantees that your colleague or instructor running your code gets the **exact same split and results**.

---

## 4. The 4-Step Object-Oriented ML Blueprint

No matter how advanced the model is, the code follows the exact same 4-step rhythm:

```python
# Step 1: Instantiate the model with hyperparameters
model = KNeighborsClassifier(n_neighbors=5)

# Step 2: Fit (train) the model on training data
model.fit(X_train, y_train)

# Step 3: Predict on unseen test data
y_pred = model.predict(X_test)

# Step 4: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

---

## 5. Summary of Universal ML Programming Terms

| Term | What It Means | Code Example |
| :--- | :--- | :--- |
| **Epoch / Iteration** | One complete pass through the training data | `for epoch in range(100):` |
| **Loss Function** | Measures error for a single sample | $\text{Error} = (y - \hat{y})^2$ |
| **Cost Function** | Average error across all $N$ training samples | $\text{MSE} = \frac{1}{N}\sum(y_i - \hat{y}_i)^2$ |
| **Overfitting** | Model memorizes training data but fails on test data | High train score, Low test score |
| **Underfitting** | Model is too simple to learn even training patterns | Low train score, Low test score |
| **Metric** | Human-interpretable score of success | Accuracy, Precision, Recall, $R^2$ |
