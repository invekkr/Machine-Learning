# Mathematics Prerequisites: 03. Functions and Graphs for Machine Learning

At its core, **every Machine Learning model is simply a mathematical function**. Understanding what functions are, how they are represented graphically, and what slope and intercept mean gives you the blueprint for Linear Regression, Logistic Regression, and Neural Networks.

---

## 1. What is a Function?

### Intuition: The Processing Machine
Think of a function as an automated kitchen appliance:
- You feed in raw ingredients (**Input $x$**).
- The machine applies a fixed recipe (**The Rule $f$**).
- It outputs a finished meal (**Output $y$ or $f(x)$**).

```text
       Input (x)  ──►  [ Function: f(x) = 2x + 1 ]  ──►  Output (y)
          1       ──►            2(1) + 1           ──►       3
          2       ──►            2(2) + 1           ──►       5
          3       ──►            2(3) + 1           ──►       7
```

### Mathematical Notation:
- **$f(x)$**: Pronounced *"f of x"*, meaning the output value produced when $x$ is passed into function $f$.
- **$y = f(x)$**: Means the output $y$ is determined by the input $x$.
- **Domain:** The set of all possible valid inputs $x$.
- **Range:** The set of all possible outputs $y$.

---

## 2. The 2D Coordinate Plane and Graphing

To visualize how a function behaves, we plot $(x, y)$ coordinate pairs onto a two-dimensional grid:
- **Horizontal Axis (X-Axis):** The independent input variable.
- **Vertical Axis (Y-Axis):** The dependent output variable.

```text
  y (Output)
  ▲
  7 │                 • (3, 7)
  6 │
  5 │           • (2, 5)
  4 │
  3 │     • (1, 3)
  2 │
  1 │ • (0, 1)  ◄── Y-Intercept (x = 0)
    └─────────────────────► x (Input)
      0   1   2   3   4
```

Connecting these points creates a **straight line**.

---

## 3. The Anatomy of a Linear Function: Slope and Intercept

The standard equation of any straight line is:
$$y = mx + b$$

### Part 1: The Slope ($m$) — "Rate of Change"
Slope measures **how steep the line is** and **how much $y$ changes when $x$ increases by 1 unit**.

$$\text{Slope } m = \frac{\Delta y}{\Delta x} = \frac{\text{Change in } y}{\text{Change in } x} = \frac{\text{Rise}}{\text{Run}} = \frac{y_2 - y_1}{x_2 - x_1}$$

Using our two points $(1, 3)$ and $(2, 5)$:
$$m = \frac{5 - 3}{2 - 1} = \frac{2}{1} = 2$$
**Meaning:** For every 1 unit increase in $x$, $y$ increases by $2$ units.

- If $m > 0$: The line slopes **upward** (positive relationship).
- If $m < 0$: The line slopes **downward** (negative relationship).
- If $m = 0$: The line is completely **flat horizontal** (no relationship).

### Part 2: The Y-Intercept ($b$) — "The Starting Baseline"
The intercept is the value of $y$ when $x = 0$.
In $f(x) = 2x + 1$:
$$f(0) = 2(0) + 1 = 1$$
The line crosses the vertical axis at point $(0, 1)$.

---

## 4. Linear vs. Non-Linear Functions

```text
Linear Function (y = 2x + 1)            Non-Linear Function (y = x²)
         ▲                                       ▲
         │       /                               │  \       /
         │      /                                │   \     /
         │     /                                 │    \___/
         └──────────►                            └──────────►
   Constant Slope Everywhere                 Slope Changes at Every Point!
```

- **Linear Functions:** The rate of change (slope) is identical everywhere. A unit increase in $x$ always produces the exact same change in $y$.
- **Non-Linear Functions:** The slope changes depending on where you are on the curve. Examples:
  - Quadratic curves ($y = x^2$): U-shaped loss surfaces.
  - Sigmoid curve ($y = \frac{1}{1 + e^{-x}}$): S-shaped probability curve.
  - Logarithmic curve ($y = \ln(x)$): Rapid initial rise followed by leveling off.

---

## 5. The Grand Connection to Machine Learning

> **A Machine Learning Model IS a Function!**

When you build a model:
$$\hat{y} = f(X)$$
- **Features ($X$):** The inputs.
- **Model Parameters ($w, b$):** The slope and intercept inside the function.
- **Prediction ($\hat{y}$):** The output.

In **Simple Linear Regression**:
$$\hat{y} = w_1 x_1 + b$$
- $w_1$ is the **learned slope** (weight). It tells us how much the target changes per unit change in feature $x_1$.
- $b$ is the **learned intercept** (bias). It represents the baseline expectation when $x_1 = 0$.

Training a machine learning algorithm simply means: **adjusting $w$ and $b$ until the function's line passes as close to the training points as possible!**
