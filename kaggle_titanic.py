import numpy as np
import pandas as pd
import csv

import math

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#load files
df_train_orig = pd.read_csv("./train.csv")
df_test_orig = pd.read_csv("./test.csv")

df_train = df_train_orig.copy(deep=True)
df_train.name = "Training set"
df_test = df_test_orig.copy(deep=True)
df_test.name = "Test set"

#fixing null values
for df in [df_train, df_test]:
    #df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

t_index = df_train[df_train.loc[:, "Cabin"] == "T"].index
df_train.drop(t_index, inplace=True)

df_train.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)
df_test.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True)

#feature engineering
for df in [df_train, df_test]:
    df["Family_Members"] = df["SibSp"] + df["Parch"] + 1

    df["Is_Alone"] = df["Family_Members"].map(lambda x: 1 if x == 1 else 0)
    df["SmallFamily"] = df["Family_Members"].map(lambda x: 1 if 2 <= x <= 4 else 0)
    df["LargeFamily"] = df["Family_Members"].map(lambda x: 1 if 5 <= x else 0)

    df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

train_title_names = (df_train["Title"].value_counts() < 10)
df_train["Title"] = df_train["Title"].apply(lambda x: "Other" if train_title_names.loc[x] == True else x)
test_title_names = (df_test["Title"].value_counts() < 10)
df_test["Title"] = df_test["Title"].apply(lambda x: "Other" if test_title_names.loc[x] == True else x)

grouped_train = df.groupby(["Sex", "Pclass", "Title"])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[["Sex", "Pclass", "Title", "Age"]]

def fill_age(row):
    condition = ((grouped_median_train["Sex"] == row["Sex"]) & (grouped_median_train["Title"] == row["Title"]) & (grouped_median_train["Pclass"] == row["Pclass"]))
    if np.isnan(grouped_median_train[condition]["Age"].values[0]):
        print("True")
        condition = ((grouped_median_train["Sex"] == row["Sex"]) & (grouped_median_train["Pclass"]))

    return grouped_median_train[condition]["Age"].values[0]

for df in [df_train, df_test]:
    df["Age"] = df.apply(lambda row: fill_age(row) if np.isnan(row["Age"]) else row["Age"], axis=1)

#categorical to dummy
#categorical data are transformed to numerical data with th LabelEncoder()
le = LabelEncoder()
for df in [df_train, df_test]:
    df["Pclass"] = le.fit_transform(df["Pclass"])
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    df["Title"] = le.fit_transform(df["Title"])

#the categorical columns are converted to one-hot encoding with get_dummies()
df_train_dummy = pd.concat([df_train, pd.get_dummies(df_train["Pclass"])], axis=1)
df_train_dummy = pd.concat([df_train_dummy, pd.get_dummies(df_train["Sex"])], axis=1)
df_train_dummy = pd.concat([df_train_dummy, pd.get_dummies(df_train["Embarked"])], axis=1)
df_train_dummy = pd.concat([df_train_dummy, pd.get_dummies(df_train["Title"])], axis=1)

df_train_dummy.drop(columns=["Pclass", "Sex", "Embarked", "Title", "Name"], inplace=True)
df_train_dummy.columns = ("Survived", "Age", "SibSp", "Parch", "Fare", "Family_Members", "Is_Alone", "SmallFamily", "LargeFamily", "Pclass_1", "Pclass_2", "Pclass_3", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S", "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Other",)

df_test_dummy = pd.concat([df_test, pd.get_dummies(df_test["Pclass"])], axis=1)
df_test_dummy = pd.concat([df_test_dummy, pd.get_dummies(df_test["Sex"])], axis=1)
df_test_dummy = pd.concat([df_test_dummy, pd.get_dummies(df_test["Embarked"])], axis=1)
df_test_dummy = pd.concat([df_test_dummy, pd.get_dummies(df_test["Title"])], axis=1)

df_test_dummy.drop(columns=["Pclass", "Sex", "Embarked", "Title", "Name"], inplace=True)
df_test_dummy.columns = ("Age", "SibSp", "Parch", "Fare", "Family_Members", "Is_Alone", "SmallFamily", "LargeFamily", "Pclass_1", "Pclass_2", "Pclass_3", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S", "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Other",)

#normalizing the continous data
#the range of continous data is too wide, so we have to normalize them. std normalization.
for df in [df_train_dummy, df_test_dummy]:
    df["SibSp"] = (df["SibSp"] - df["SibSp"].mean()) / df["SibSp"].std()
    df["Parch"] = (df["Parch"] - df["Parch"].mean()) / df["Parch"].std()
    df["Family_Members"] = (df["Family_Members"] - df["Family_Members"].mean()) / df["Family_Members"].std()
    df["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()
    #df["Fare"] = (df["Fare"] - df["Fare"].mean()) / df["Fare"].std()

    #log10 Fare
    df["Fare"] = df["Fare"].map(lambda x: 0 if x == 0 else math.log10(x))

print(df_train_dummy)

#separatin x and y
#the data is finally ready for training. the input x and output y are separeted here
X_train = df_train_dummy.drop(["Survived"], axis=1)
Y_train = df_train_dummy["Survived"]

#creating forest
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(X_train, Y_train)

output = forest.predict(df_test_dummy)

df_test_orig = df_test_orig.values

with open("./result.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(df_test_orig[:, 0].astype(int), output.astype(int)):
        writer.writerow([pid, survived])
