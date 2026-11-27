import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 


df = pd.read_csv("csv_files/titanic.csv",usecols=["Age","Pclass","SibSp","Parch","Survived"])

df.dropna(inplace=True)
x = df.iloc[:,1:5]
y = df.iloc[:,0]

# print(np.mean(cross_val_score(LogisticRegression(),x,y,scoring="accuracy",cv=20)))
# print(x.head())
x["Family_size"] = x["SibSp"] + x["Parch"]+1
# print(x.head())
def myfunc(num):
    if num == 1:
        #alone
        return 0
    elif num >1 and num <=4:
        # small family
        return 1
    else:
        # large family
        return 2

x["Family_type"] = x["Family_size"].apply(myfunc)

x.drop(columns=["SibSp","Parch","Family_size"],axis=1,inplace=True)
# print(x.head())
# print(np.mean(cross_val_score(LogisticRegression(),x,y,scoring="accuracy",cv=20)))