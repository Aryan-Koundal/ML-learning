import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mysql.connector
import requests
import seaborn as sns
from data_profiling import ProfileReport
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler





# # =================================== Function Transformer ==================================

from scipy import stats
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


df = pd.read_csv("titanic.csv",usecols=["Age","Fare","Survived"])

df["Age"].fillna(df["Age"].mean(),inplace=True)

x_train,x_test,y_train,y_test = train_test_split(df.drop("Survived",axis=1),df["Survived"],random_state=0)

# sns.displot(x_train["Age"])
# plt.show()

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(x_train,y_train)
clf2.fit(x_train,y_train)

y_predicition_1 = clf.predict(x_test)
y_predicition_2 = clf2.predict(x_test)

print("Accuracy = ",accuracy_score(y_test,y_predicition_1))
print("Accuracy = ",accuracy_score(y_test,y_predicition_2))

trf = FunctionTransformer(func=np.log1p)

x_train_transform = trf.fit_transform(x_train)
x_test_transform = trf.transform(x_test)

clf.fit(x_train_transform,y_train)
clf2.fit(x_train_transform,y_train)

y_predicition_trf = clf.predict(x_test_transform)
y_predicition_trf2 = clf2.predict(x_test_transform)

print("Accuracy = ",accuracy_score(y_test,y_predicition_trf))
print("Accuracy = ",accuracy_score(y_test,y_predicition_trf2))

x_transformed = trf.fit_transform(x_train)

print("LR",np.mean(cross_val_score(clf,x_transformed,y_train,scoring="accuracy",cv=10)))
print("DR",np.mean(cross_val_score(clf2,x_transformed,y_train,scoring="accuracy",cv=10)))