# Practical Machine Learning Study Repository

Welcome to your practical, hands-on Machine Learning study repository. This project is structured as a self-study course designed to bridge the gap between **mathematical/statistical intuition** and **practical Python implementation**.

Each module follows a dual-format learning architecture:
1. **`notes.md`**: Medium-detail concept study notes explaining the *why*, *when*, mathematical/statistical definitions, and key trade-offs.
2. **`*.ipynb`**: A clean, fully annotated Jupyter Notebook following the **Theory → Implementation → Observation → Practical Interpretation** pattern.

---

## 📚 Syllabus & Modules Covered

### Foundational Prerequisites ([`Prerequisites/`](file:///Users/shamvi/stuff/ML/Prerequisites/))
* **01_Python_Prerequisites**:
  * **01_Python_Basics**: Variables, basic data types (`int`, `float`, `str`, `bool`, `None`), lists, tuples, sets, dicts, indexing, loops, functions, comprehensions
  * **02_NumPy**: `ndarray` memory layout, 1D/2D shapes, slicing, vectorized arithmetic, aggregations (`axis=0`/`axis=1`), dot products
  * **03_Pandas**: Series vs DataFrames, CSV loading, row/column slicing, boolean filtering, missing value imputation, `describe()`, `groupby()`
  * **04_Data_Handling**: Features ($X$) vs Target ($y$), Continuous vs Discrete, Train/Val/Test splits, Data Leakage prevention
  * **05_Matplotlib_Seaborn**: Scatter plots, line charts, histograms & KDE, bar plots, box plots, correlation heatmaps
  * **06_Scikit_Learn**: The Estimator API, Transformers (`fit`, `transform`), Predictors (`fit`, `predict`), `Pipeline`
  * **07_Jupyter_Notebook**: Cells, background kernel memory, execution order, reproducibility
  * **08_ML_Programming_Concepts**: Dimensionality `(N, D)`, parameters vs hyperparameters, `random_state` reproducibility
* **02_Mathematics_Prerequisites**:
  * **01_Basic_Mathematics**: Exponents, square roots, logarithms ($\log_{10}$, $\log_2$, $\ln$), Euler's constant ($e$), scientific notation
  * **02_Algebra**: Variables, constants, linear equations, rearranging formulas, systems of equations, quadratic loss curves ($y = w^2$)
  * **03_Functions_and_Graphs**: Inputs $\to$ rule $\to$ outputs, $f(x) = 2x + 1$, slope ($m = \text{Rise}/\text{Run}$), intercept ($b$), linear vs non-linear curves
  * **04_Statistics**: Central tendency (Mean, Median, Mode), Dispersion (Range, Variance, Standard Deviation, IQR), Skewness, Outliers, Z-scores
  * **05_Probability**: Sample space, probabilities $[0, 1]$, independent vs dependent events, conditional probability, **Bayes' Theorem**
  * **06_Linear_Algebra**: Scalars, vectors, dot products, matrices, matrix multiplication, transpose, identity, inverse, covariance matrix, **Eigenvalues & Eigenvectors**
  * **07_Calculus**: Limits, derivatives as speed/slope, polynomial power rules, **Partial Derivatives** ($\frac{\partial f}{\partial x}$), **Gradient Vector** ($\nabla$)
  * **08_Optimization**: Minima, maxima, loss vs cost functions, **Gradient Descent** ($\theta = \theta - \alpha \nabla J$), learning rates, SGD
  * **09_Common_ML_Mathematical_Concepts**: Euclidean vs Manhattan distance, MAE/MSE/RMSE, Sigmoid function, Entropy & Information Gain, $L_1$/$L_2$ Regularization

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
* **06_Machine_Learning_Pipeline (`sklearn.pipeline.Pipeline`)**:
  * Foundational concept: The automated assembly line connecting data cleaning, preprocessing, and modeling
  * Why Pipelines are necessary: Code simplicity, train/test consistency, and absolute prevention of data leakage
  * Pipeline mechanics: What happens during `fit()` (fit + transform on transformers, fit on model) vs. `predict()` (transform only)
  * The Dream Team: How `ColumnTransformer` (WHERE) and `Pipeline` (WHEN / ORDER) nest together seamlessly
  * Modular Sub-Pipelines: Building dedicated imputation + scaling/encoding lanes for numerical and categorical groups
  * Pipeline inspection: Exploring components via `named_steps` and parameters via `get_params()` with double underscores (`__`)
  * Leakage-proof Cross-Validation: Ensuring preprocessing is fit strictly inside training folds during `cross_val_score`
  * Simultaneous hyperparameter tuning using `GridSearchCV` on models and preprocessing steps
  * Safe model deployment: Saving and loading full pipelines with `pickle` (and security considerations)
  * End-to-end practical experiment: Training and evaluating a customer purchase classification pipeline
* **07_Mathematical_Transformations (`FunctionTransformer`, Log, Sqrt, Reciprocal, QQ Plots)**:
  * Difference between **Feature Scaling** (changes range/scale, preserves shape) and **Mathematical Transformation** (changes distribution shape and skewness)
  * Identifying **Right Skewness** ($\text{Mean} > \text{Median}$) vs. **Left Skewness** ($\text{Mean} < \text{Median}$)
  * **Log Transformation**: Compressing extreme right tails via $\ln(x)$ and zero-handling via $\ln(1 + x)$ (`np.log1p`)
  * **Square Root Transformation**: Mild compression for moderate positive skewness via $\sqrt{x}$
  * **Reciprocal Transformation**: Inverting features via $\frac{1}{x}$ and handling non-zero requirements
  * **Square Transformation**: Expanding variance for left-skewed distributions via $x^2$
  * Visual normality diagnostics: **Quantile-Quantile (QQ) plots** using `scipy.stats.probplot`
  * Scikit-Learn integration: Wrapping NumPy mathematical functions into reusable pipeline objects with `FunctionTransformer(..., feature_names_out='one-to-one')`
  * Multi-column preprocessing: Combining `FunctionTransformer` with `StandardScaler` and `OneHotEncoder` inside `ColumnTransformer`
  * Model sensitivity reality check: Why tree-based models (Random Forests, XGBoost) are invariant to monotonic transformations, while linear/distance-based models benefit
  * **Head-to-Head Practical Experiment**: Evaluating Linear Regression on skewed feature data, demonstrating a test error reduction ($\text{MAE}: \$1,036.87 \to \$883.27$, $R^2: 0.5402 \to 0.6275$)
* **08_Discretization_and_Binarization (`KBinsDiscretizer`, `Binarizer`, `pd.cut`)**:
  * Foundational concept: Converting continuous numerical features into discrete intervals or binary decisions
  * Why bin: Simplifying noisy signals, handling outliers, modeling non-linear piecewise ranges, and business policy interpretation
  * **Equal Width Binning (`uniform`)**: Partitioning by constant range width $\frac{x_{\max} - x_{\min}}{k}$ and handling outlier skew risks
  * **Equal Frequency Binning (`quantile`)**: Partitioning by sample quantiles to guarantee balanced observation counts per bin
  * **K-Means Binning (`kmeans`)**: Grouping multimodal clustered data via 1D K-Means cluster centroid midpoints
  * **Custom Domain Binning**: Segmenting via business rules and statutes (e.g. Minor, Adult, Senior) using `pd.cut()`
  * `KBinsDiscretizer` parameters: `n_bins`, `strategy` (`uniform`, `quantile`, `kmeans`), and `encode` (`ordinal`, `onehot-dense`)
  * **Binarization (`Binarizer`)**: Threshold-based conversion to 0/1; precise mathematical behavior ($x > \text{threshold} \implies 1, \ x \le \text{threshold} \implies 0$)
  * **The Information Loss Trade-Off**: Why collapsing values into bins permanently discards within-bin precision
  * Outlier capping: How binning bounds extreme tail values without row deletion
  * Leakage-proof Train/Test workflow and integration into `ColumnTransformer` and `Pipeline`
  * **Real-World Titanic Experiment**: Benchmarking Logistic Regression comparing continuous scaled `Age` (78.77% acc, 72.86% F1) vs. quantile binned `Age` (77.09% acc, 71.33% F1), validating the impact of information loss

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
