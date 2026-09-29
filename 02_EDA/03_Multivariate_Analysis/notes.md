# Module 02: Exploratory Data Analysis — Multivariate Analysis

**Multivariate Analysis** involves the simultaneous examination of **three or more variables**.

While univariate analysis focuses on individual distributions and bivariate analysis explores pairwise relationships, real-world systems are rarely governed by pairs of variables in isolation. Features interact, compound, and confound each other.

---

## 1. Why Multivariate Analysis is Necessary

In real-world data science and machine learning:
1. **Confounding Variables:** A strong bivariate relationship might actually be driven by a third unobserved or ignored variable.
2. **Interaction Effects:** The impact of feature $A$ on target $Y$ often depends on the level of feature $B$.
3. **Simpson's Paradox:** A statistical phenomenon where a trend appears in different groups of data but disappears or reverses when these groups are combined.

> **Titanic Motivation:**
> If you only look at `Sex` vs. `Survived` (bivariate), you conclude: *"Females had a high survival rate (74%), males had a low survival rate (19%)."*
> But what happened to females in 3rd class compared to 1st class?
> And what happened to children when conditioned on both travel class and gender?
> Only multivariate analysis reveals the full historical reality.

---

## 2. Multi-Channel Visual Encodings

When plotting on a 2D screen, how do we represent 3, 4, or 5 variables without getting confused?
We use visual encodings:

| Visual Channel | Best Variable Type | Seaborn Parameter | Practical Example |
| :--- | :--- | :--- | :--- |
| **X-Position** | Continuous / Ordered | `x="Age"` | Horizontal placement |
| **Y-Position** | Continuous | `y="Fare"` | Vertical placement |
| **Color (Hue)** | Categorical (Target) | `hue="Survived"` | Green = Survived, Red = Perished |
| **Shape (Style)** | Categorical (Low Cardinality) | `style="Sex"` | Circle = Male, Cross = Female |
| **Size** | Continuous or Ordinal | `size="Pclass"` or `size="SibSp"` | Larger marker = Higher value |

### 2.1 Multivariate Scatter Plot Example
```python
import seaborn as sns

sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived",  # 3rd variable (color)
    style="Sex",  # 4th variable (marker shape)
    size="Pclass",  # 5th variable (marker size)
    sizes=(40, 140),
    alpha=0.7,
)
```

---

## 3. Multivariate Faceting (`FacetGrid` & `sns.catplot`)

When visual channels on a single plot become cluttered, **faceting** (small multiples) splits the data into a grid of distinct subplots conditioned on one or two categorical features.

### 3.1 `sns.catplot`
Allows generating grouped box plots, bar plots, or violin plots across rows and columns of categories:

```python
# Survival rate across Sex (x), Pclass (col), and Embarked (row)
sns.catplot(
    data=df,
    x="Sex",
    y="Survived",
    col="Pclass",
    kind="bar",
    errorbar=None,
    palette="muted",
)
```

### 3.2 Faceted Distribution Grids (`FacetGrid`)
```python
g = sns.FacetGrid(df, col="Pclass", row="Sex", hue="Survived", margin_titles=True)
g.map(sns.histplot, "Age", bins=20, alpha=0.6)
g.add_legend()
```

---

## 4. Multi-Variable Aggregation: Pivot Tables & Heatmaps

When analyzing interactions between 3 variables (e.g., 2 categorical inputs and 1 numerical target), **`pd.pivot_table()`** combined with a **Heatmap** provides an exceptionally clear summary.

### Syntax
```python
# Compute average survival rate conditioned on both Sex and Pclass
pivot = df.pivot_table(
    index="Sex", columns="Pclass", values="Survived", aggfunc="mean"
)

# Render as an annotated percentage heatmap
sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn")
```

---

## 5. The Titanic Multivariate Revelation

Let us observe the interaction of **`Sex` + `Pclass` + `Survived`**:

| Gender | 1st Class Survival | 2nd Class Survival | 3rd Class Survival |
| :--- | :--- | :--- | :--- |
| **Female** | **96.8%** | **92.1%** | **50.0%** |
| **Male** | **36.9%** | **15.7%** | **13.5%** |

### What Multivariate Analysis Uncovers:
1. **1st Class Females:** Survival was virtually guaranteed (nearly 97%).
2. **3rd Class Females:** Despite the "women and children first" maritime protocol, half (50%) of 3rd class women perished due to cabin locations deep in the ship and locked steerage gates.
3. **1st Class Males:** Survived at a higher rate (36.9%) than 2nd or 3rd class men, and significantly higher than overall male survival (18.9%).
4. **Children across Classes:** In 1st and 2nd class, virtually 100% of children survived. In 3rd class, child survival dropped below 40%.

Bivariate analysis could never tell this complete story.

---

## Summary Checklist
- [x] Use `hue`, `style`, and `size` to map up to 5 variables on a single scatter plot.
- [x] Use `sns.pairplot(hue=...)` to check pairwise feature separations conditioned on the target label.
- [x] Deploy `sns.catplot` with `col` and `row` faceting when single plots become too crowded.
- [x] Create multi-variable pivot tables and heatmaps to reveal non-linear group interaction effects.
- [x] Always look beyond pairwise correlations to detect hidden confounding variables.
