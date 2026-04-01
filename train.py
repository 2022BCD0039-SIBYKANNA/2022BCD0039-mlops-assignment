import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import argparse
import joblib
import os

mlflow.set_tracking_uri("sqlite:///./mlflow.db")
mlflow.set_experiment("2022BCD0039_experiment")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

def load_data(path):
df = pd.read_csv(path)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

```
df['Age'] = df['Age'].fillna(df['Age'].mean())

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

return df
```

def train(args):
df = load_data(args.data_path)

```
X = df.drop('Survived', axis=1)
y = df['Survived']

if args.use_subset:
    X = X[['Pclass', 'Sex']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42
)

with mlflow.start_run():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("use_subset", args.use_subset)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)

    os.makedirs("./models", exist_ok=True)
    joblib.dump(model, "./models/model.pkl")

    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)
    print("Precision:", prec)
```

if **name** == "**main**":
parser = argparse.ArgumentParser()

```
parser.add_argument("--data_path", type=str, default="data/titanic.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--use_subset", action="store_true")

args = parser.parse_args()

train(args)
```
