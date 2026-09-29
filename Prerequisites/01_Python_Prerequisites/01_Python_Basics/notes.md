# Python Prerequisites: 01. Python Basics for Machine Learning

Python is the primary language used in data science and machine learning. You do not need to build complex web applications or game engines in Python—you only need to master how Python stores numbers, manages collections of data, loops through samples, and encapsulates logic in functions.

---

## 1. What is Python?

Python is a high-level, human-readable programming language. Unlike lower-level languages (like C or C++) where you have to manually manage computer memory, Python allows you to express data manipulation in clean, clear syntax that reads almost like plain English.

---

## 2. Variables and Basic Data Types

A **variable** is simply a labeled container or storage box in your computer's memory that holds a value.

```python
# A simple variable
age = 25
```
- `age`: The **variable name** (the label on the box).
- `=`: The **assignment operator** (stores the value into the box).
- `25`: The **value** stored inside.

### The 5 Fundamental Data Types in ML:

| Data Type | What It Represents | Example | ML Context |
| :--- | :--- | :--- | :--- |
| **`int`** (Integer) | Whole numbers (positive, negative, zero) | `num_bedrooms = 3` | Counts, discrete quantities |
| **`float`** (Floating-point) | Numbers with decimal points | `price = 250.75` | Continuous features, probabilities, weights |
| **`str`** (String) | Text enclosed in quotes (`" "` or `' '`) | `city = "Delhi"` | Categorical labels, text data |
| **`bool`** (Boolean) | True or False binary states | `is_churned = True` | Binary classification targets (1 / 0) |
| **`None`** (`NoneType`) | Absence of a value / Empty | `missing_value = None` | Representing missing data (`NaN`) |

```python
# Demonstrating types in Python
score = 95.5
label = "Approved"
is_valid = True

print(type(score))  # <class 'float'>
print(type(label))  # <class 'str'>
print(type(is_valid))  # <class 'bool'>
```

---

## 3. Data Structures: Collections of Data

Machine learning models rarely process one number at a time. They process thousands of rows and columns. Python provides four fundamental built-in collection structures:

### 1. Lists (`[]`)
An **ordered, mutable (changeable)** sequence of elements.
```python
# A list of feature values
salaries = [50000, 60000, 75000, 90000]

# Modifying and adding elements
salaries.append(120000)
print(salaries)  # [50000, 60000, 75000, 90000, 120000]
```

### 2. Tuples (`()`)
An **ordered, immutable (unchangeable)** sequence. Once created, its values cannot be altered.
```python
# Typically used for fixed configurations or data shapes
image_shape = (28, 28)
```

### 3. Sets (`{}`)
An **unordered collection of UNIQUE elements**. Duplicates are automatically removed.
```python
# Extracting unique categories from a column
cities = ["Delhi", "Mumbai", "Delhi", "Bangalore", "Mumbai"]
unique_cities = set(cities)
print(unique_cities)  # {'Bangalore', 'Delhi', 'Mumbai'}
```

### 4. Dictionaries (`{key: value}`)
A collection of **key-value pairs**, like a real-world dictionary where you look up a word (key) to get its definition (value).
```python
# Storing a single customer record
customer = {"age": 28, "city": "Delhi", "purchased": True}

print(customer["city"])  # 'Delhi'
```

---

## 4. Indexing and Slicing

In Python, counting starts at **0** (zero-based indexing).

```text
Values:  [ 10,   20,   30,   40,   50 ]
Index:      0     1     2     3     4
Negative:  -5    -4    -3    -2    -1
```

- **`items[0]`**: First item (`10`).
- **`items[-1]`**: Last item (`50`).
- **`items[1:4]`**: Slicing from index 1 up to (but not including) index 4 $\longrightarrow$ `[20, 30, 40]`.

---

## 5. Control Flow: `if`, `elif`, `else`

Control flow allows your program to make decisions based on conditions:

```python
score = 0.85  # Probability of customer purchasing

if score >= 0.80:
  prediction = "High Likelihood"
elif score >= 0.50:
  prediction = "Moderate Likelihood"
else:
  prediction = "Low Likelihood"

print(prediction)  # 'High Likelihood'
```

---

## 6. Loops: Iterating Over Data

### `for` Loop
Used when you know what collection you are iterating over:
```python
# Calculating sum manually
errors = [2.0, -1.0, 3.5, 0.5]
total_error = 0.0

for err in errors:
  total_error += err

print("Total Error:", total_error)  # 5.0
```

### `while` Loop
Repeats as long as a condition remains `True` (frequently used in optimization until a model converges):
```python
iteration = 0
max_iterations = 3

while iteration < max_iterations:
  print(f"Optimization step {iteration + 1}")
  iteration += 1
```

---

## 7. Functions: Reusable Code Blocks

A function is a reusable block of code that takes **inputs (arguments)**, performs a calculation, and **returns an output**.

```python
def calculate_mean(numbers):
  """Calculates the average of a list of numbers."""
  total = sum(numbers)
  count = len(numbers)
  return total / count


# Test with simple numbers
data = [10, 20, 30]
average = calculate_mean(data)
print("Mean:", average)  # 20.0
```

---

## 8. List Comprehensions: Clean, Fast Data Transformations

A list comprehension is Python's concise way to transform an existing list into a new list in a single readable line.

```python
# Convert raw errors into squared errors
errors = [2, -3, 4]

# Traditional way:
squared_manual = []
for e in errors:
  squared_manual.append(e**2)

# List Comprehension way (preferred in ML):
squared = [e**2 for e in errors]
print(squared)  # [4, 9, 16]
```

---

## 9. Basic Exception Handling (`try` / `except`)

When loading messy datasets, errors (like missing files or corrupted values) can occur. Exception handling allows your program to handle errors gracefully without crashing.

```python
raw_value = "invalid_number"

try:
  clean_number = float(raw_value)
except ValueError:
  clean_number = 0.0  # Safe default fallback

print("Processed value:", clean_number)  # 0.0
```

---

## 10. Objects, Attributes, and Methods in Machine Learning

In modern Python libraries like Scikit-Learn:
- An **Object** is a software instance representing a tool or model (e.g. `scaler = StandardScaler()`).
- An **Attribute** is data or learned statistics stored inside the object (e.g. `scaler.mean_`).
- A **Method** is an action or function that the object can perform (e.g. `scaler.fit(X)` or `scaler.transform(X)`).

```python
# Mental model:
object_name.method_name(data)  # Performs an action
object_name.attribute_name  # Accesses a stored property
```

---

## 11. Connecting Python Basics to Machine Learning

| Python Concept | Real Machine Learning Application |
| :--- | :--- |
| **`float` & `int`** | Representing features (Age = 25, Weight = 72.5 kg). |
| **`Lists` & `Dicts`** | Holding raw records, feature names, hyperparameter configurations. |
| **`List Comprehension`** | Computing residual errors, transforming labels quickly. |
| **`Functions`** | Custom loss functions, metric evaluation (Accuracy, Precision). |
| **`Objects & Methods`** | Interacting with the Scikit-Learn Estimator API (`model.fit()`, `model.predict()`). |
