# Machine Learning Prerequisites: The Foundational Toolkit

Welcome to the **Prerequisites** section of your Machine Learning journey!

Before building complex models like Linear Regression, Decision Trees, Support Vector Machines, or Neural Networks, every machine learning practitioner needs two foundational pillars:
1. **Programming Foundations** (Python, NumPy, Pandas, Scikit-Learn basics) — the tools used to manipulate data and execute algorithms.
2. **Mathematical Foundations** (Basic Math, Algebra, Functions, Statistics, Probability, Linear Algebra, Calculus, and Optimization) — the language used to understand *how* algorithms learn from data.

---

## 🧭 Who Should Study This?

- **Complete Beginners:** If you are new to programming or haven't touched mathematics since school, this section was built specifically for you.
- **Self-Taught Developers:** If you know how to write Python code but want to understand the intuition behind statistical formulas and gradient descent without feeling overwhelmed.
- **Experienced Practitioners Needing a Refresher:** If you need a quick, intuitive reference for concepts like eigenvalues, Bayes' theorem, covariance, or partial derivatives.

---

## 🎯 Our Pedagogical Philosophy: Intuition First, Zero Jargon

You do **NOT** need a degree in advanced mathematics to master Machine Learning. 

Every single topic in this prerequisite guide follows a strict 7-step learning progression:

$$\text{Plain English Intuition} \longrightarrow \text{Tiny Real-World Numbers} \longrightarrow \text{Formal Definition} \longrightarrow \text{Step-by-Step Formula} \longrightarrow \text{Manual Calculation} \longrightarrow \text{Python Code} \longrightarrow \text{Why ML Uses It}$$

- We never jump directly into abstract equations.
- We never assume you already know technical words like "eigenvector", "log-likelihood", or "orthogonality".
- We calculate everything with simple numbers first (like $10, 20, 30$) before showing the equivalent Scikit-Learn or NumPy code.

---

## 🗂️ Prerequisite Architecture

```text
Prerequisites/
├── README.md                              # Main overview, roadmap, and algorithm mapping
│
├── 01_Python_Prerequisites/               # Programming foundations
│   ├── README.md
│   ├── 01_Python_Basics/                  # Variables, data types, control flow, functions, lists/dicts
│   ├── 02_NumPy/                          # Arrays, dimensions, shapes, slicing, vector arithmetic, dot products
│   ├── 03_Pandas/                         # Series, DataFrames, indexing, filtering, missing values, groupbys
│   ├── 04_Data_Handling/                  # Features vs targets, train/val/test splits, leakage concepts
│   ├── 05_Matplotlib_Seaborn/             # Scatter, line, bar, histogram, boxplot, heatmap visualizations
│   ├── 06_Scikit_Learn/                   # Estimator API: fit, transform, predict, Pipeline architecture
│   ├── 07_Jupyter_Notebook/               # Cells, kernel execution, interactive workflow tips
│   └── 08_ML_Programming_Concepts/       # Dimensions, random_state, hyperparameters, metrics
│
└── 02_Mathematics_Prerequisites/          # Mathematical foundations
    ├── README.md
    ├── 01_Basic_Mathematics/              # Exponents, square roots, logarithms, order of operations
    ├── 02_Algebra/                        # Variables, linear & quadratic equations, solving for unknowns
    ├── 03_Functions_and_Graphs/           # Inputs/outputs, f(x), slope, intercept, linear vs non-linear
    ├── 04_Statistics/                     # Mean, median, mode, variance, standard deviation, IQR, skewness
    ├── 05_Probability/                    # Events, sample space, conditional probability, Bayes' Theorem
    ├── 06_Linear_Algebra/                 # Scalars, vectors, matrices, dot products, covariance, eigenvalues
    ├── 07_Calculus/                       # Limits, derivatives as speed/slope, polynomial rules, partial derivatives
    ├── 08_Optimization/                   # Minima, maxima, loss functions, gradient descent intuition
    └── 09_Common_ML_Mathematical_Concepts/# Distance metrics (Euclidean/Manhattan), MSE/MAE, Sigmoid, Entropy
```

---

## 🗺️ Recommended Study Order

```text
Step 01: Python Basics ────────► Step 02: NumPy ────────► Step 03: Pandas
                                                               │
Step 06: Basic Mathematics ◄── Step 05: Data Handling ◄───────┘
         │
         ▼
Step 07: Algebra & Functions ──► Step 08: Statistics ──► Step 09: Probability
                                                               │
Step 12: ML Math Concepts ◄── Step 11: Calculus & Opt ◄───────┘
         │
         ▼
Ready for Main Curriculum:
[01_Data_Handling] ──► [02_EDA] ──► [03_Feature_Engineering] ──► [ML Models]
```

---

## 🔗 Algorithm $\longrightarrow$ Prerequisite Mapping Table

When you advance into the 100 Days of Machine Learning curriculum, here is the exact foundational map of what each algorithm depends on:

| Machine Learning Algorithm / Topic | Core Mathematical Prerequisites | Core Programming Prerequisites |
| :--- | :--- | :--- |
| **Linear Regression** | Slope, Intercept, Equations, Mean, Residual Errors, MSE, Matrix Dot Product | `NumPy` 2D arrays, `pandas` DataFrame, `Scikit-Learn` LinearRegression |
| **Polynomial Regression** | Exponents, Non-linear functions, Powers, Overfitting intuition | `PolynomialFeatures`, `Scikit-Learn` Pipeline |
| **Ridge & Lasso Regression** | Loss functions, Absolute values ($L_1$), Squared terms ($L_2$), Optimization | Regularization parameter `alpha` in Scikit-Learn |
| **Logistic Regression** | Probability, Logarithms, Euler's $e$, Sigmoid function, Log-Loss | Binary classification targets, `predict_proba` |
| **K-Nearest Neighbors (KNN)** | Distance metrics (Euclidean, Manhattan), Distance scaling | `StandardScaler`, Vector distances, `KNeighborsClassifier` |
| **Naive Bayes** | Sample space, Conditional probability, Independent events, Bayes' Theorem | Categorical encoding, frequency tables, `GaussianNB` |
| **Decision Trees** | Proportions, Logarithms ($\log_2$), Entropy, Information Gain, Gini Impurity | Nested if-else logic, recursion intuition, `DecisionTreeClassifier` |
| **Random Forest & Ensembles** | Sampling with replacement (Bootstrap), Aggregation, Variance reduction | Random state seeds, loop iterations, `RandomForestClassifier` |
| **Gradient Boosting & XGBoost** | Residual errors, Derivatives, Learning rates, Sequential optimization | Loss gradient stepping, sequential pipelines |
| **Support Vector Machines (SVM)**| Vectors, Dot products, Hyperplanes, Geometric margins, Convex optimization | Feature scaling (`StandardScaler`), matrix operations |
| **K-Means Clustering** | Centroids, Mean, Euclidean distance, Iterative optimization | 2D distance calculation, unsupervised clustering API |
| **Principal Component Analysis (PCA)**| Variance, Covariance matrix, Orthogonal axes, Eigenvalues & Eigenvectors | Matrix multiplication, SVD, `sklearn.decomposition.PCA` |
| **Gradient Descent** | Function slopes, Derivatives, Partial derivatives ($\partial f / \partial x$), Gradient vector ($\nabla$) | Numerical parameter updates, learning rate multipliers |
| **Feature Scaling (Z-Score)** | Mean ($\mu$), Standard Deviation ($\sigma$), Bell-curve spread | `StandardScaler`, `fit_transform` vs `transform` |
| **Min-Max Normalization** | Minimum, Maximum, Ratios, Value spans | `MinMaxScaler`, bounding intervals $[0, 1]$ |

---

## 🚀 How to Use These Notes

1. Pick the topic you feel least comfortable with.
2. Read [`notes.md`](file:///Users/shamvi/stuff/ML/Prerequisites/01_Python_Prerequisites/01_Python_Basics/notes.md) to understand the conceptual *why* and walk through the simple numerical examples.
3. Open the accompanying Jupyter Notebook (`*.ipynb`) to run the simple code cells, experiment with values, and verify the outputs.
4. Move seamlessly into the main repository modules with confidence!
