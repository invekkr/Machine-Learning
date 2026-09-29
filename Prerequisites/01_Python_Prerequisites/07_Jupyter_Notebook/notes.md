# Python Prerequisites: 07. Jupyter Notebooks for Machine Learning

Jupyter Notebooks (`.ipynb`) are the universal laboratory workbench of data scientists and machine learning engineers. They allow you to combine executable code, rich text explanations, mathematical equations, and interactive charts in a single, shareable document.

---

## 1. What is a Jupyter Notebook?

In traditional software development, you write a Python script in a `.py` file and execute the entire program from start to finish via terminal. If a bug occurs at line 95, the program crashes, and you have to run everything from line 1 again.

In Machine Learning, training a model or loading a 10 GB dataset can take minutes or hours. You cannot afford to restart from zero every time you want to plot a graph.

**Jupyter Notebooks** solve this by breaking code into independent, interactive blocks called **cells**.

---

## 2. The Two Types of Cells

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Markdown Cell (Documentation & Theory)                   │
│    # Heading 1                                              │
│    This section explains the intuition behind our model.     │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ 2. Code Cell (Executable Python)                            │
│    [1]: import pandas as pd                                 │
│         df = pd.read_csv('data.csv')                        │
│         df.head(2)                                          │
└─────────────────────────────────────────────────────────────┘
```

- **Markdown Cells:** Formatted text using Markdown syntax (headings, bullet points, LaTeX formulas like $y = mx + b$, and links).
- **Code Cells:** Executable Python code. When you press `Shift + Enter`, only that specific cell executes, and its output (table, text, or plot) appears directly beneath it.

---

## 3. The Kernel: The Invisible Engine

Behind every Jupyter Notebook sits a **Kernel**—a running Python process in your computer's memory.

```text
  Jupyter Web Interface (Browser)  ◄───►  Python Kernel (Background RAM)
  [Cell 1: x = 10]                         Stores: x = 10
  [Cell 2: print(x + 5)]  ──► Output: 15
```

### The Out-of-Order Execution Trap (Crucial Warning!):
The Kernel remembers variables in the **order you executed the cells**, NOT the physical order they appear on the page!

If you write:
- Cell 1: `x = 10` (Executed first $\longrightarrow `[1]`$)
- Cell 2: `x = 20` (Executed second $\longrightarrow `[2]`$)
- Then you scroll back up and run Cell 1 again, `x` is now `10`!
- The bracket numbers `[1]`, `[2]`, `[3]` show the exact chronological order of execution.

> **Golden Rule of Reproducibility:**  
> Before sharing your notebook or concluding an experiment, always click:  
> **Kernel $\longrightarrow$ Restart & Run All Cells**.  
> If your notebook runs from top to bottom without errors, your code is reproducible.

---

## 4. Useful Keyboard Shortcuts

| Shortcut | Mode | Action |
| :--- | :--- | :--- |
| **`Shift + Enter`** | Any | Run current cell and select next cell |
| **`Ctrl + Enter`** | Any | Run current cell in-place |
| **`Esc` then `A`** | Command | Insert a new cell **Above** |
| **`Esc` then `B`** | Command | Insert a new cell **Below** |
| **`Esc` then `M`** | Command | Convert cell to **Markdown** |
| **`Esc` then `Y`** | Command | Convert cell to **Code** |
| **`Esc` then `D, D`** | Command | **Delete** current cell |
