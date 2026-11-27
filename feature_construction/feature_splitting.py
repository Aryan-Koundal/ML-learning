import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 

df = pd.read_csv("csv_files/titanic.csv")

# print(df.head())

df["Title"] = df["Name"].str.split(",",expand=True)[1].str.split(".",expand=True)[0]
print(df.groupby('Title')['Survived'].mean().sort_values(ascending=False))