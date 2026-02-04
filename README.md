# 🚢 Titanic Survival Prediction (Machine Learning)

This project predicts passenger survival on the Titanic dataset using a **Random Forest Classifier**.  
It demonstrates **data cleaning, feature engineering, categorical encoding, and model evaluation**.

---

## Dataset

- Source: Kaggle Titanic Dataset
- Files:
  - `train.csv`
  - `test.csv`

---

## Data Preprocessing

- Dropped non-informative columns:
  - `PassengerId`, `Name`, `Ticket`
- Handled missing values:
  - `Embarked` → filled with `"U"` (Unknown)
  - `Cabin` → reduced to first letter and mapped ordinally
- Feature engineering:
  - Extracted and mapped cabin codes
- Encoding:
  - `Sex` → label encoded
  - `Embarked` → one-hot encoded
- Ensured **train/test feature alignment** using `reindex`

---

## Model

- **RandomForestClassifier**
- Hyperparameters:
  - `n_estimators = 10`
  - `max_depth = 15`
  - `min_samples_leaf = 10`
  - `min_samples_split = 8`

---

## Evaluation

- 80/20 train–validation split
- Metrics:
  - Accuracy
  - Confusion Matrix

Example:

```text
Validation Accuracy ≈ 0.815
```
