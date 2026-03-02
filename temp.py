import pandas as pd

# to check if the given dataset requires data cleaning
a = pd.read_csv("data/TEST.csv")

print("Shape:", a.shape)
print("\nColumns:\n", a.columns)
print("\nMissing values:\n", a.isnull().sum())
print("\nClass distribution:\n", a["ID"].value_counts())
print("\nFeature summary:\n", a.describe())