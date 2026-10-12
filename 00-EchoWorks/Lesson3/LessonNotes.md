# ML Zoomcamp – Module 3: Machine Learning for Classification

Project: **churn prediction** for a telecom company (Telco Customer Churn dataset, ~7,000 customers, ~27% churned).

Goal: give every customer a score between 0 and 1 (the probability they will leave), then target high-risk customers with retention offers.

---

## 1. The problem: binary classification

- **Churn** = a customer stops using a service.
- The target `y` has two values:
    - `1` = positive class (churned)
    - `0` = negative class (stayed)
- The model `g` learns from historical data so that `g(xi) ≈ yi`, where `xi` is the feature vector for customer *i*.
- The output of the model is a **probability**, not just a label.

The same setup works for spam detection, default prediction, lead scoring, and any other yes/no problem.

---

## 2. Data preparation

Typical cleaning steps:

```python
df = pd.read_csv('data.csv')

# Uniform column names and values
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

# Fix a numeric column that was read as text
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# Encode the target as 0/1
df.churn = (df.churn == 'yes').astype(int)
```

Key points:

- `df.head().T` transposes the preview so you can see all columns at once.
- `errors='coerce'` turns values that can't be parsed into `NaN` instead of raising an error.
- Filling with 0 is only reasonable when 0 has a real meaning (here: new customers who haven't paid yet).

In a notebook, `!command` runs a shell command and `$var` inserts a Python variable into it:

```python
!wget $data -O data.csv
```

---

## 3. Validation framework (train / val / test)

A **60 / 20 / 20** split in two steps:

```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
```

- The second split uses `0.25` because 25% of the remaining 80% is 20% of the original data.
- `random_state` makes the split reproducible.

Then reset the indexes and separate the target:

```python
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']
```

Save the `y_` arrays **before** deleting the column. Removing the target from the feature dataframes prevents it from leaking into the model.

---

## 4. EDA

- **Missing values:** `df.isnull().sum()`
- **Target distribution:** `df.churn.value_counts(normalize=True)`
- **Churn rate shortcut:** the mean of a 0/1 column is the fraction of 1s, so `df.churn.mean()` is the churn rate.
- **Unique values per categorical column:** `df[categorical].nunique()`. This tells you how many columns one-hot encoding will create.

Split the columns into lists:

- `numerical`: real numbers (e.g. `tenure`, `monthlycharges`, `totalcharges`)
- `categorical`: everything else that's a feature (an ID column is **not** a feature)

A 0/1 column like `seniorcitizen` can still be categorical.

---

## 5. Feature importance

### 5a. Churn rate difference and risk ratio (categorical)

Compare each group's churn rate with the global churn rate:

- **Difference** = group rate − global rate
    - `> 0` → group churns more than average
    - `< 0` → group churns less
- **Risk ratio** = group rate / global rate
    - `> 1` → more likely to churn
    - `< 1` → less likely to churn
    - `= 1` → same as average

Doing it for every variable at once:

```python
global_churn = df_full_train.churn.mean()

for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
```

Use `display()` inside loops, since tables aren't shown automatically there.

Findings in the churn data: month-to-month contracts, fiber optic internet and electronic check payments are high-risk. Gender barely matters.

### 5b. Mutual information (categorical)

Answers the question: *how much do we learn about the target if we know this feature?* It gives one number per variable, so whole variables can be ranked against each other.

```python
from sklearn.metrics import mutual_info_score

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)
```

- Always ≥ 0. Higher means more informative.
- Symmetric: the order of the arguments doesn't matter.
- In the churn data, `contract` scored highest (~0.098) and `gender` was close to 0.

### 5c. Correlation coefficient (numerical)

The **Pearson correlation** `r` is a value between −1 and 1:

| abs(r) | Strength |
|---|---|
| 0 – 0.2 | Low |
| 0.2 – 0.5 | Medium |
| 0.5 – 1.0 | Strong |

- The sign gives the direction:
    - positive → the feature goes up, churn goes up
    - negative → the feature goes up, churn goes down
- To rank importance, take the absolute value.

```python
df_full_train[numerical].corrwith(df_full_train.churn).abs()
```

Double-check by binning: compute the churn rate for groups such as `tenure <= 2`, `2 < tenure <= 12` and `tenure > 12`. In the churn data, `tenure` was the most important numerical feature (r ≈ −0.35).

---

## 6. One-hot encoding

Models need numbers. One-hot encoding turns one categorical column into several 0/1 columns, one per category. Exactly one is "hot" per row.

```python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
```

- `to_dict(orient='records')` creates one dictionary per row.
- **`fit` only on train**, then just `transform` on val and test. Otherwise the columns can change order or new columns can appear.
- Numerical columns pass through unchanged.
- `dv.get_feature_names_out()` lists the created columns.
- `sparse=False` returns a normal NumPy array instead of a sparse matrix.
- Alternative: Scikit-Learn's `OneHotEncoder`.

---

## 7. Logistic regression

It is the same weighted sum as linear regression, passed through a **sigmoid**:

```
g(xi) = sigmoid(w0 + w1·x1 + w2·x2 + ... + wn·xn)

sigmoid(z) = 1 / (1 + exp(-z))
```

- The sigmoid squashes any number into the range (0, 1), which turns the score into a probability.
- `sigmoid(0) = 0.5`
    - A positive score means more likely than not.
    - A negative score means less likely.
- It's still a **linear model**: only the final transformation differs from linear regression.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

---

## 8. Training with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

model.intercept_[0]   # bias w0
model.coef_[0]        # weights, one per column
```

Predictions:

- `model.predict(X)` gives **hard** predictions (0/1).
- `model.predict_proba(X)` gives **soft** predictions: two columns, P(0) and P(1).

```python
y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
accuracy = (y_val == churn_decision).mean()
```

**Accuracy** is the fraction of correct predictions (~80% on validation in the churn project). The 0.5 threshold can be tuned; that's covered in the next module.

---

## 9. Interpreting the model

Pair the weights with the feature names:

```python
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))
```

- A **positive weight** pushes the probability up; a **negative weight** pushes it down.
- With one-hot encoding, only one category's weight per feature is "used" for a given row. The others are multiplied by 0.
- A smaller model (e.g. `contract`, `tenure`, `monthlycharges`) is easier to read.

Scoring by hand: bias + matching category weight + (value × weight) for each numeric feature, then apply the sigmoid.

---

## 10. Using the model

1. Retrain on `df_full_train` (train + validation) with a **new** `DictVectorizer`.
2. Evaluate once on the **test** set. Validation and test accuracy should be close; a big gap suggests overfitting.
3. Score one customer:

```python
customer = dicts_test[-1]
X_small = dv.transform([customer])          # wrap the dict in a list
model.predict_proba(X_small)[0, 1]
```

In production, a service sends the customer's data as a dict. The model returns a probability, and if it's above the threshold, the business acts (e.g. sends a discount).

---

## 11. Main takeaways

- Feature importance: **risk ratio**, **mutual information**, **correlation**
- One-hot encoding with **`DictVectorizer`**
- Logistic regression is a linear model with a sigmoid on top
- Its output is a **probability**
- Interpreting the weights works like linear regression

---

## 12. Homework – Lead scoring dataset (2026)

Target: `converted` (did the lead sign up?)

- **Categorical:** `lead_source`, `industry`, `employment_status`, `location`
- **Numerical:** `annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score`
- **Missing values:** categorical filled with `'NA'`, numerical filled with `0.0`

| Question | Method | My answer |
|---|---|---|
| Q1 | Mode of `industry` (after filling NAs) | |
| Q2 | Biggest correlation among the given pairs | |
| Q3 | Highest mutual information with `converted` (train set) | |
| Q4 | Validation accuracy, `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)` | 0.65 |
| Q5 | Least useful feature (smallest accuracy difference) | `number_of_courses_viewed` |
| Q6 | Best `C` from `[0.000001, 0.00001, 0.0001, 0.001]` | 0.001 |

Observations:

- Dropping `annual_income` *raised* accuracy by about 0.08. Its large scale (tens of thousands) throws off logistic regression, so scaling it (`StandardScaler` or `np.log1p`) would likely help.
- Smaller `C` means stronger regularization. Accuracy rose as `C` increased: 0.598 → 0.598 → 0.613 → 0.645.

---

## Cheat sheet

| Function / method | What it does |
|---|---|
| `!wget url -O file` | Download a file from a notebook |
| `df.head().T` | Transposed preview of the dataframe |
| `pd.to_numeric(x, errors='coerce')` | Convert to numbers; bad values become NaN |
| `df.fillna(v)` | Replace missing values |
| `(df.x == 'yes').astype(int)` | Yes/no column to 1/0 |
| `train_test_split(df, test_size, random_state)` | Split the data |
| `df.reset_index(drop=True)` | Reset the index, discard the old one |
| `df.isnull().sum()` | Count missing values per column |
| `df.x.value_counts(normalize=True)` | Share of each category |
| `df[cols].nunique()` | Number of unique values per column |
| `df.groupby(c).y.agg(['mean', 'count'])` | Group statistics |
| `mutual_info_score(x, y)` | Mutual information |
| `df[num].corrwith(df.y)` | Correlation with the target |
| `df.to_dict(orient='records')` | Rows to a list of dicts |
| `DictVectorizer(sparse=False)` | One-hot encoder for dicts |
| `dv.fit_transform` / `dv.transform` | Fit on train, transform the rest |
| `dv.get_feature_names_out()` | Encoded column names |
| `LogisticRegression().fit(X, y)` | Train the model |
| `model.predict_proba(X)[:, 1]` | Probability of class 1 |
| `model.coef_[0]`, `model.intercept_[0]` | Weights and bias |
| `zip(a, b)` | Pair two sequences element by element |
| `display(df)` | Show a table inside a loop |
