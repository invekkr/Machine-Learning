# Practical Machine Learning Study Repository

Welcome to your practical, hands-on Machine Learning study repository. This project is structured as a self-study course designed to bridge the gap between **mathematical/statistical intuition** and **practical Python implementation**.

Each module follows a dual-format learning architecture:
1. **`notes.md`**: Medium-detail concept study notes explaining the *why*, *when*, mathematical/statistical definitions, and key trade-offs.
2. **`*.ipynb`**: A clean, fully annotated Jupyter Notebook following the **Theory → Implementation → Observation → Practical Interpretation** pattern.

---

## 🗂️ Repository Structure

```text
ML/
├── README.md
├── requirements.txt
├── data/
│   ├── titanic.csv                     # Primary dataset used across Modules 1 & 2
│   └── wine.csv                        # Chemical recognition dataset used for Feature Scaling
├── 01_Data_Handling/
│   ├── 01_Working_with_Files/
│   │   ├── notes.md                    # File formats, I/O parameters, head/tail/sample
│   │   └── working_with_files.ipynb
│   └── 02_Understanding_Data/
│       ├── notes.md                    # Structural inspection, stats, missing/duplicates, correlation
│       └── understanding_data.ipynb
├── 02_EDA/
│   ├── 01_Univariate_Analysis/
│   │   ├── notes.md                    # Categorical & numerical distributions, outliers, dispersion
│   │   └── univariate_analysis.ipynb
│   ├── 02_Bivariate_Analysis/
│   │   ├── notes.md                    # Num-Num, Num-Cat, Cat-Cat relationships, crosstabs
│   │   └── bivariate_analysis.ipynb
│   ├── 03_Multivariate_Analysis/
│   │   ├── notes.md                    # 3+ variables, encodings (hue, style, size), Simpson's Paradox
│   │   └── multivariate_analysis.ipynb
│   └── 04_Pandas_Profiling/
│       ├── notes.md                    # YData Profiling, automated EDA, metrics & warnings
│       └── pandas_profiling.ipynb
├── 03_Feature_Engineering/
│   └── 01_Standardization/
│       ├── notes.md                    # Z-score theory, algorithms affected, leakage prevention
│       └── standardization.ipynb       # Practical experiment: KNN Without vs With Scaling
└── outputs/
    ├── plots/                          # Saved figures and visualizations
    └── reports/                        # Automated profiling HTML reports
```

---

## 📚 Syllabus & Modules Covered

### Module 1: Data Handling
* **01_Working_with_Files**:
  * CSV file architecture & delimiter mechanics
  * Reading CSVs (`pd.read_csv`) with key parameters (`usecols`, `nrows`, `na_values`, `index_col`)
  * Writing CSVs cleanly (`df.to_csv(index=False)`)
  * Inspecting data subsets: `head()`, `tail()`, and unbiased inspection with `sample(random_state=42)`
* **02_Understanding_Data**:
  * Shape, column types, and memory breakdown (`shape`, `columns`, `info()`, `dtypes`)
  * Five-number numerical summaries & categorical summaries (`describe(include='all')`)
  * Missing value detection and nullity percentages (`isnull().sum()`)
  * Duplicate detection (`duplicated().sum()`)
  * Cardinality analysis: `unique()`, `nunique()`, and normalized `value_counts()`
  * Linear correlation coefficients ($r$), correlation matrices (`df.corr()`), and annotated heatmaps
  * **Critical thinking**: Correlation vs. Causation (spurious correlations, confounding variables)

### Module 2: Exploratory Data Analysis (EDA)
* **01_Univariate_Analysis**:
  * **Categorical variables**: Frequency tables, Count plots, Bar charts, Pie & Donut charts, Pareto Analysis (80/20 rule)
  * **Numerical variables**: Discrete vs. Continuous, Histograms & binning bias, KDE smoothing, Violin plots, ECDF curves, Strip plots
  * **Central Tendency & Dispersion**: Mean, Median, Mode (skew sensitivity); Range, Variance, Standard Deviation, Interquartile Range (IQR)
  * **Outlier Detection**: The $1.5 \times IQR$ Tukey rule and Five-Number Summary
  * **Distribution Shape**: Skewness (right vs. left), Kurtosis (peakedness & tail heaviness)
* **02_Bivariate_Analysis**:
  * **Numerical + Numerical**: Scatter plots, linear/non-linear relationships, line plots, pair plots
  * **Numerical + Categorical**: Bar plots (mean comparison), Box plots (distribution comparison across groups), Grouped KDEs
  * **Categorical + Categorical**: Two-way contingency tables (`pd.crosstab`), raw counts vs. conditional row/column percentages, heatmap representations, and hierarchical clustering (`clustermap`)
* **03_Multivariate_Analysis**:
  * Analyzing 3 or more variables simultaneously to reveal hidden confounders
  * Multi-channel visual encodings: `hue` (color), `style` (marker), `size` (magnitude)
  * Multi-dimensional scatter plots and subset pair plots
  * Faceted category grids (`sns.catplot`, `FacetGrid`)
  * Pivoted heatmaps for 3-way interactions
  * Case study: Demystifying survival probabilities across Sex AND Class on the Titanic
* **04_Pandas_Profiling (YData Profiling)**:
  * Purpose, benefits, and execution of automated profiling reports (`ProfileReport`)
  * Generating interactive standalone HTML reports (`profile.to_file(...)`)
  * Interpreting warnings: High cardinality, collinearity/high correlation, zeros, missingness
  * The boundaries of automated EDA: Why automation accelerates discovery but cannot replace human domain intuition or problem formulation

### Module 3: Feature Engineering
* **01_Standardization (`StandardScaler`)**:
  * Mathematical foundations of Z-score normalization: $z = \frac{x - \mu}{\sigma}$
  * Distinction between **Centering** ($\mu \rightarrow 0$) and **Scaling** ($\sigma \rightarrow 1$)
  * Why distribution shape is strictly preserved (standardization does NOT convert skewed data into normal)
  * Outlier persistence (outliers remain outliers; only their scale shifts into standard deviation units)
  * Identifying scale-sensitive algorithms (KNN, SVM, K-Means, Neural Networks) vs. scale-invariant tree models (Decision Trees, Random Forests, XGBoost)
  * Rigorous prevention of **Train/Test Data Leakage** (`fit_transform` on `X_train`, `transform` on `X_test`)
  * **Head-to-Head Practical Experiment**: KNN on Wine recognition dataset demonstrating a jump from **72.22%** (unscaled) to **94.44%** (standardized)

---

## 🚢 Datasets Used

### 1. Titanic Dataset (`data/titanic.csv`)
Used across Modules 1 & 2 for Data Handling and Exploratory Data Analysis (891 passenger records, 12 features).

### 2. Wine Recognition Dataset (`data/wine.csv`)
Used in Module 3 for Feature Scaling (178 samples, 13 continuous chemical features with disparate magnitudes ranging from 0.1 to 1,680+).

| Column | Type | Description |
| :--- | :--- | :--- |
| `PassengerId` | Discrete Integer | Unique passenger identifier |
| `Survived` | Binary Categorical | Survival target (0 = No, 1 = Yes) |
| `Pclass` | Ordinal Categorical | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Text / String | Passenger name (including title) |
| `Sex` | Nominal Categorical | Passenger gender (male, female) |
| `Age` | Continuous Numerical | Passenger age in years |
| `SibSp` | Discrete Numerical | Number of siblings / spouses aboard |
| `Parch` | Discrete Numerical | Number of parents / children aboard |
| `Ticket` | Categorical / String | Ticket number |
| `Fare` | Continuous Numerical | Passenger fare paid (£) |
| `Cabin` | Categorical / String | Cabin number (high missingness) |
| `Embarked` | Nominal Categorical | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🛠️ Environment Setup & Getting Started

1. **Activate the Python virtual environment**:
   ```bash
   source ~/ml-env/bin/activate
   ```

2. **Verify or install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn ydata-profiling jupyter nbformat nbclient
   ```

3. **Launch Jupyter Lab / Notebook**:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

4. **Recommended Study Routine**:
   * Open `notes.md` in the target topic directory to absorb the conceptual and statistical foundations.
   * Open the companion `.ipynb` notebook.
   * Read the markdown rationale, run each cell, observe the output/plot, and read the practical interpretation before proceeding to the next cell.
