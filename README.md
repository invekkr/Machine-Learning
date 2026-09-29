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
│   ├── 01_Standardization/
│   │   ├── notes.md                    # Z-score theory, algorithms affected, leakage prevention
│   │   └── standardization.ipynb       # Practical experiment: KNN Without vs With Scaling
│   ├── 02_Normalization/
│   │   ├── notes.md                    # Min-Max, Mean Norm, MaxAbs (Sparse data), Robust (IQR)
│   │   └── normalization.ipynb         # Step-by-step mental math + Scikit-Learn implementations
│   ├── 03_Encoding_Categorical_Data/
│   │   ├── notes.md                    # Nominal vs Ordinal, OrdinalEncoder, LabelEncoder vs OHE
│   │   └── encoding_categorical_data.ipynb # Decision trees, artificial distance traps, Titanic case study
│   ├── 04_One_Hot_Encoding/
│   │   ├── notes.md                    # Deep dive: OHE mechanics, Dummy Variable Trap, high cardinality
│   │   └── one_hot_encoding.ipynb      # Step-by-step Pandas vs Sklearn, leakage, rare grouping
│   └── 05_Column_Transformer/
│       ├── notes.md                    # Traffic controller, syntax (name, transformer, cols), remainder
│       └── column_transformer.ipynb    # Multi-type preprocessing, SimpleImputer, Pipeline integration
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
* **02_Normalization (`MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`)**:
  * Step-by-step mental math before code for every technique
  * **Min-Max Scaling**: Mapping to $[0, 1]$ via $\frac{x - x_{min}}{x_{max} - x_{min}}$, image pixels $0\dots 255 \rightarrow 0\dots 1$, and outlier squashing risk
  * **Mean Normalization**: Centering at 0 and dividing by range $\frac{x - \mu}{x_{max} - x_{min}}$ (distinguished from standardization)
  * **Max Absolute Scaling**: Scaling to $[-1, 1]$ via $\frac{x}{\max(|x|)}$ and its critical superpower on **Sparse Data** (preserves zero structure without memory explosions)
  * **Robust Scaling**: Leveraging Median and Interquartile Range ($IQR = Q_3 - Q_1$) to scale data without being corrupted by extreme outliers
  * Visual side-by-side box plots comparing all scalers on outlier-contaminated data
  * Comprehensive **Normalization vs. Standardization** comparison table
  * **Real-World Experiment**: Benchmarking KNN on Wine recognition dataset comparing Unscaled (72.22%) vs. Min-Max (94.44%) vs. Robust Scaling (94.44%)
* **03_Encoding_Categorical_Data (`OrdinalEncoder`, `OneHotEncoder`, `LabelEncoder`)**:
  * Qualitative vs. Quantitative features and why ML math requires numbers
  * **Nominal vs. Ordinal**: Dissecting unordered labels (City, Gender) vs. hierarchical ranks (Education, Feedback)
  * **Ordinal Encoding**: Preserving true real-world order via `OrdinalEncoder(categories=[...])` and why arbitrary alphabetical sorting fails
  * **Label Encoding**: Strictly for 1D classification target labels ($y$), NOT nominal input features ($X$)
  * **The Artificial Distance Trap**: Why assigning integers to nominal features (Chennai: 0, Delhi: 1, Mumbai: 2) teaches algorithms false linear hierarchies
  * **One-Hot Encoding**: Representing nominal features as equidistant binary vectors without artificial rankings (`OneHotEncoder(sparse_output=False)`)
  * Decision framework: Input ($X$) vs. Target ($y$), Ordinal vs. Nominal
  * Data leakage prevention across train/test splits with `handle_unknown='ignore'`
  * **Titanic Case Study**: Encoding nominal `Sex` & `Embarked` with OHE, ordinal `Pclass` with OrdinalEncoder, and verifying clean model ingestion with Logistic Regression
* **04_One_Hot_Encoding (Deep Dive)**:
  * Foundational concept: mapping each unique nominal category to an independent binary indicator column ($1 = \text{present}, 0 = \text{absent}$)
  * Mathematical distance preservation: proving categories are equidistant ($\sqrt{2} \approx 1.414$)
  * **The Dummy Variable Trap**: Why $N$ categories cause perfect multicollinearity ($r = -1.0$) and how dropping one column ($N - 1$) preserves full information while avoiding singular matrices
  * **Pandas Implementation**: `pd.get_dummies(df, drop_first=True)` for rapid exploratory analysis
  * **Scikit-Learn Implementation**: `OneHotEncoder(sparse_output=False, drop='first')` for production pipelines
  * **Leakage & Execution**: Strict split-first protocol, difference between `fit()`, `transform()`, and `fit_transform()`
  * **Unseen Categories**: Production handling via `handle_unknown="ignore"` (zero-vector fallback without crashing)
  * **High Cardinality & Dimensionality Explosion**: Memory and computational risks of wide matrices
  * **Rare Category Grouping**: Pruning rare labels into `"Other"` using frequency/percentage thresholds without arbitrary magic rules
* **05_Column_Transformer (`sklearn.compose.ColumnTransformer`)**:
  * Foundational concept: The central traffic controller for datasets with mixed numerical, ordinal, and nominal data types
  * The manual preprocessing problem: Why manual column slicing and stacking is error-prone and brittle
  * Syntax architecture: The `(name, transformer, columns)` tuple structure
  * Execution mechanics: How `fit_transform()` splits, fits, transforms, and horizontally stacks arrays
  * Handling unmentioned columns: `remainder='drop'` vs. `remainder='passthrough'`
  * Multi-type preprocessing: Unifying `StandardScaler`, `OrdinalEncoder`, and `OneHotEncoder`
  * Missing value handling: Integrating `SimpleImputer` inside `ColumnTransformer`
  * Leakage-proof train/test workflow: `fit_transform(X_train)` and `transform(X_test)`
  * Output feature tracking with `get_feature_names_out()` and understanding output column expansion
  * Production ML pipelines: Connecting `ColumnTransformer` directly to estimators using `Pipeline`

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
