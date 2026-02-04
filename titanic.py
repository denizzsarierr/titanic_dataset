import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv("train.csv")

def clean_data(data):

    data = data.copy()

    # Drop columns that are not characteristic
    data.drop("PassengerId",axis = 1,inplace = True)
    data.drop("Name",axis = 1,inplace = True)
    data.drop("Ticket",axis = 1,inplace = True)


    # Deal nan - U = "Unknown"
    data["Embarked"] = data["Embarked"].fillna("U")

    data["CabinCode"] = data["Cabin"].str[0].fillna("N")
    data.drop("Cabin",axis=1,inplace=True)

    # Mapping Cabin Codes

    cabin_map = {"N" : 0,
                 "A" : 1,
                 "B" : 2,
                 "C" : 3,
                 "D" : 4,
                 "E" : 5,
                 "F" : 6,
                 "G" : 7,
                 "T" : 8}
    
    data["CabinCode"] = data["CabinCode"].map(cabin_map)

    # One-Hot for Embarked
    data = pd.get_dummies(data, columns=['Embarked'],dtype=int, drop_first=True)

    label_encoder = LabelEncoder()

    data["Sex"] = label_encoder.fit_transform(data["Sex"])
    
    return data

datas = clean_data(data)

print(datas.head())

y = datas.Survived

X = datas.drop("Survived",axis = 1)


X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.8,random_state=0)

model = RandomForestClassifier(
    max_features=7,
    min_samples_leaf=10,
    min_samples_split=8,
    n_estimators=10,
    max_depth=15,
    verbose=1,random_state=0)

model.fit(X_train,y_train)

pred = model.predict(X_val)

acc = accuracy_score(y_val, pred)
print("Validation Accuracy:", acc)

cm = confusion_matrix(y_val, pred)
print("Confusion Matrix:\n", cm)

acc_random_forest = round(model.score(X_train, y_train) * 100, 3) 
random_forest = round(model.score(X_val, y_val) * 100, 3)
print(acc_random_forest)
print("----------------")
print(random_forest)

test_data = pd.read_csv("test.csv")

cleaned_test = clean_data(test_data)
cleaned_test = cleaned_test.reindex(columns=X.columns, fill_value=0)

pred_test = model.predict(cleaned_test)
pred_test = pred_test.astype(int)
#print(pred_test)

submission = pd.DataFrame({
    "PassengerId": range(1, len(pred_test) + 1),
    "Survived": pred_test
})

submission.to_csv("titanic_predictions.csv", index=False)