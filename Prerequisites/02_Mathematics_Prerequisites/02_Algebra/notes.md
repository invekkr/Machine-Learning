# Mathematics Prerequisites: 02. Algebra for Machine Learning

Algebra is the language of machine learning. When we say an algorithm "learns", what it is actually doing is **solving an algebraic equation to find the best unknown numbers (weights)** that match the data.

---

## 1. What is Algebra?

Arithmetic is working with known numbers ($2 + 3 = 5$).  
**Algebra** is arithmetic when some numbers are unknown or can vary. We use letters (like $x, y, w, b$) as placeholders for those unknown values.

---

## 2. Variables, Constants, and Expressions

- **Constant:** A number that never changes (e.g. $5, -12, 0.5$).
- **Variable:** A letter representing a quantity that can change or an unknown we want to find (e.g. $x$ represents Age, $y$ represents Salary).
- **Coefficient:** A number placed directly before and multiplying a variable (in $3x$, $3$ is the coefficient).
- **Expression:** A combination of variables and constants without an equals sign (e.g. $2x + 3$).
- **Equation:** A mathematical statement that two expressions are equal (e.g. $2x + 3 = 11$).

---

## 3. Solving Linear Equations (The Balance Scale Rule)

Think of an equation as a perfectly balanced scale. Whatever operation you perform on the left side, you **must perform on the right side** to keep it balanced.

### Example 1: Addition & Subtraction
$$x + 5 = 10$$
To isolate $x$, subtract $5$ from both sides:
$$x + 5 - 5 = 10 - 5$$
$$x = 5$$

### Example 2: Multiplication & Division
$$2x = 8$$
To isolate $x$, divide both sides by $2$:
$$\frac{2x}{2} = \frac{8}{2}$$
$$x = 4$$

### Example 3: Two-Step Equation
$$3x + 4 = 19$$
1. Subtract $4$ from both sides:
   $$3x = 15$$
2. Divide both sides by $3$:
   $$x = 5$$

---

## 4. Rearranging Formulas (Isolating Variables)

In machine learning, you will often need to rearrange a formula to solve for a specific parameter.

Take the classic equation of a straight line:
$$y = mx + b$$

Suppose we want to rearrange this to solve for $x$:
1. Subtract $b$ from both sides:
   $$y - b = mx$$
2. Divide both sides by $m$:
   $$x = \frac{y - b}{m}$$

---

## 5. Systems of Equations and Substitution

What if you have two equations with two unknowns?
1. $x + y = 10$
2. $y = 2x + 1$

### Method of Substitution:
Substitute the expression for $y$ from Equation 2 into Equation 1:
$$x + (2x + 1) = 10$$
$$3x + 1 = 10$$
$$3x = 9 \implies x = 3$$

Now substitute $x = 3$ back into Equation 2:
$$y = 2(3) + 1 = 7$$
Solution: $(x = 3, y = 7)$.

**ML Context:** When models find where decision boundaries intersect or solve linear regression with multiple features, they are solving systems of linear equations.

---

## 6. Quadratic Equations and Parabolic Curves

A linear equation produces a straight line ($y = 2x + 1$).  
A **Quadratic Equation** contains a squared variable:
$$y = ax^2 + bx + c$$

### The Visual Shape: A Parabola (U-Shape)
When $a > 0$, the graph forms a smooth **U-shaped curve** that has a single **lowest point (minimum)**.

```text
  Error (Loss)
       ▲
       │   \       /
       │    \     /
       │     \___/  ◄── Global Minimum (Zero or lowest error!)
       └────────────────► Model Weight (w)
```

**Why This Is Central to Machine Learning:**  
In regression, our loss function is Mean **Squared** Error ($E = (y - \hat{y})^2$). Because it is squared, the error surface forms a parabola! Optimization algorithms search for the bottom of this bowl to find the best possible model weights.

---

## 7. Connecting Algebra to Machine Learning

| Algebraic Concept | Machine Learning Application |
| :--- | :--- |
| **$y = mx + b$** | **Simple Linear Regression:** $\hat{y} = w_1 x_1 + b$ where $w_1$ is the weight (slope) and $b$ is the bias (intercept). |
| **$y = w_1 x_1 + w_2 x_2 + b$** | **Multiple Linear Regression:** Predicting house price ($y$) from size ($x_1$) and bedrooms ($x_2$). |
| **Rearranging Equations** | Deriving closed-form analytical solutions (the OLS Normal Equation). |
| **Inequalities ($w \cdot x + b \ge 0$)** | **Classification Decision Boundaries:** If result $\ge 0$, predict Class 1; else predict Class 0. |
