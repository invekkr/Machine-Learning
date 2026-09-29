# Mathematics & Statistics Prerequisites for Machine Learning

Mathematics is often treated as a gatekeeper to Machine Learning, filled with intimidating Greek symbols and abstract theorems. 

In this repository, we reject that approach.

**In Machine Learning, mathematics is simply a practical toolkit for problem-solving:**
- **Algebra & Functions** define how models express relationships ($y = mx + b$).
- **Linear Algebra** provides the data structure to store thousands of features compactly as matrices and vectors.
- **Calculus** provides the compass that guides models during learning (calculating which direction reduces error).
- **Probability & Statistics** allow models to quantify uncertainty and evaluate performance on unseen data.

---

## 🗂️ Module Overview

| Module | Core Question Answered | Key Topics Covered |
| :--- | :--- | :--- |
| [**01_Basic_Mathematics**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/01_Basic_Mathematics/notes.md) | *"What arithmetic foundations does ML rely on?"* | Positive/negative numbers, fractions, ratios, exponents ($2^3 = 8$), square roots ($\sqrt{9}=3$), logarithms ($\log_{10}(100)=2$), Euler's constant ($e$). |
| [**02_Algebra**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/02_Algebra/notes.md) | *"How do we solve for unknown model parameters?"* | Variables, constants, expressions, linear equations ($x + 5 = 10$), quadratic equations, rearranging equations, substitution. |
| [**03_Functions_and_Graphs**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/03_Functions_and_Graphs/notes.md) | *"What is an ML model mathematically?"* | Inputs $\to$ Function $\to$ Outputs, $f(x) = 2x + 1$, 2D coordinate planes, slope ($m$), intercept ($b$), linear vs non-linear curves. |
| [**04_Statistics**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/04_Statistics/notes.md) | *"How do we summarize and describe data?"* | Central tendency (Mean, Median, Mode), Dispersion (Range, Variance, Standard Deviation, IQR), Skewness, Outliers, Z-scores. |
| [**05_Probability**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/05_Probability/notes.md) | *"How do models quantify uncertainty?"* | Experiments, outcomes, sample spaces, probabilities $[0, 1]$, independent vs dependent events, conditional probability, **Bayes' Theorem**. |
| [**06_Linear_Algebra**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/06_Linear_Algebra/notes.md) | *"How do models process tabular data in bulk?"* | Scalars, vectors, dot products, matrices, matrix multiplication, transpose, identity, inverse, covariance matrix, **Eigenvalues & Eigenvectors**. |
| [**07_Calculus**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/07_Calculus/notes.md) | *"How do models know how to improve?"* | Limits, rate of change (speed/slope), derivatives ($\frac{dy}{dx}$ of $x^2$), polynomial power rules, **Partial Derivatives** ($\frac{\partial f}{\partial x}$), **Gradients** ($\nabla$). |
| [**08_Optimization**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/08_Optimization/notes.md) | *"How do models find the best possible weights?"* | Minima, maxima, objective functions, loss vs cost functions, **Gradient Descent intuition** ($\theta = \theta - \alpha \nabla J$). |
| [**09_Common_ML_Mathematical_Concepts**](file:///Users/shamvi/stuff/ML/Prerequisites/02_Mathematics_Prerequisites/09_Common_ML_Mathematical_Concepts/notes.md) | *"What formulas appear everywhere in ML?"* | Euclidean & Manhattan distance, MSE, MAE, Sigmoid function, Log-Loss, Entropy, Information Gain. |

---

## 🎯 Our Step-by-Step Learning Formula

For every single mathematical concept in this folder:
1. **Plain English Intuition:** What does this concept mean in daily life?
2. **Tiny Number Example:** Simple calculation with numbers like $10, 20, 30$.
3. **Formula & Symbol Breakdown:** We define every single letter so nothing is mysterious.
4. **Step-by-step Manual Arithmetic:** We show every intermediate step.
5. **Python Verification:** We confirm the result with 2 lines of Python.
6. **Why Machine Learning Uses It:** We link it directly to future algorithms (e.g. PCA, Linear Regression, KNN).
