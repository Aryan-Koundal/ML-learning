import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("people_1000_ml_practice.csv")
# df.head()
# df.info()
# df.describe()
# print(df)
# ===========extracting employed===========
# x = df.iloc[:,:2]
# y = df.iloc[:,3:]
# z = df.iloc[:,2]
# a = pd.concat([x,y],axis=1)
# print(a)
# print(z)
x = df.drop(["Person_ID","Employed"],axis=1)
y = df["Employed"]
# print(x)
# print(y)

# === training ===
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1)
# print(y_train)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)

# print(x_train)

model = LogisticRegression()

model.fit(x_train,y_train)

prediction = model.predict(x_test)
# print(prediction)
# print("=====")
# print(y_test)
accuracy = accuracy_score(y_test,prediction)
# print("Accuracy = ",accuracy)
# print(y.value_counts())
# print(y.value_counts(normalize=True) * 100)