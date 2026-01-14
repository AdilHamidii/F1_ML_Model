import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("training_data.csv") # Reading the data from the training_data.csv file

df = df.dropna(subset=["RacePosition"]) # Drop rows where RacePosition is NaN



X = df[["Q1", "Q2", "Q3", "QualiPosition", "TeamName", "Year"]]
y = df["RacePosition"] 

train = df[df["Year"] < 2024]
test  = df[df["Year"] == 2024]

X_train = train[["Q1", "Q2", "Q3", "QualiPosition", "TeamName", "Year"]]
y_train = train["RacePosition"]

X_test = test[["Q1", "Q2", "Q3", "QualiPosition", "TeamName", "Year"]]
y_test = test["RacePosition"]



year_weights = {2022: 1.0, 2023: 1.3}
sample_weight = train["Year"].map(year_weights)


sample_weight = sample_weight * (1 / train["QualiPosition"])


preprocess = ColumnTransformer([
    ("team", OneHotEncoder(handle_unknown="ignore"), ["TeamName"]),
    ("num", "passthrough", ["Q1", "Q2", "Q3", "QualiPosition", "Year"])
])


model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])


model.fit(
    X_train,
    y_train,
    rf__sample_weight=sample_weight
)


pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
within_3 = (np.abs(pred - y_test) <= 3).mean()

print("MAE:", mae)
print("+-3 positions accuracy:", within_3*100, "%")
