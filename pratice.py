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


api = requests.get("https://science.nasa.gov/wp-json/wp/v2/apod-basic/?api_key=DEMO_KEY")
print(api)
a = pd.DataFrame(api.json())
print(a)
conn = mysql.connector.connect(host="localhost",user="root",password="@#aryan_3039",database="student_management")
x = pd.read_sql_query("SELECT * FROM students",conn)
print(x)
df = pd.read_csv("people_1000_ml_practice.csv")
df.head()
df.info()
df.describe()
print(df)
# ===========extracting employed===========
x = df.iloc[:,:2]
y = df.iloc[:,3:]
z = df.iloc[:,2]
a = pd.concat([x,y],axis=1)
print(a)
print(z)
x = df.drop(["Person_ID","Employed"],axis=1)
y = df["Employed"]
print(x)
print(y)

# === training ===
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.1)
print(y_train)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.transform(y_test)

print(x_train)

model = LogisticRegression()

model.fit(x_train,y_train)

prediction = model.predict(x_test)
print(prediction)
print("=====")
print(y_test)
accuracy = accuracy_score(y_test,prediction)
print("Accuracy = ",accuracy)
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)

df = pd.DataFrame({
    "Age": ["20", "25", "30"],
    "Salary": [20000, 25000, 30000]
})

df["Age"] = df["Age"].astype(int)
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.duplicated().sum())
print(df.corr()["Age"])

# ===working with titanic dataset===
# =====univariate analysis=====
df = pd.read_csv("titanic.csv")
print(df["Survived"].value_counts())
plt.figure()
df["Survived"].value_counts().plot(kind="pie",autopct="%.2f")# sns.countplot(df["Survived"])
plt.show()
plt.figure()
y = sns.countplot(x=df["Survived"])
plt.show()
plt.hist(df["Age"])
sns.displot(df["Age"])
sns.boxplot(x=df["Fare"])
plt.show()
print(df["Age"].min())
print(df["Age"].max())
print(df["Age"].skew())

# ======================================
titanic = pd.read_csv("titanic.csv")
iris = pd.read_csv("iris_dataset.csv")
passengers = pd.read_csv("air_passengers.csv")

sns.scatterplot(x=passengers["Passengers"],y=passengers["Month"],hue=passengers["Year"])
plt.show()
sns.barplot(x=titanic["Pclass"],y=titanic["Age"],hue=titanic["Sex"])
plt.show()
sns.boxplot(x=titanic["Age"],y=titanic["Sex"],hue=titanic["Survived"])
plt.show()
sns.histplot(x=titanic[titanic["Survived"]==0]["Age"],kde=True)
sns.histplot(x=titanic[titanic["Survived"]==1]["Age"],kde=True)
plt.show()
sns.heatmap(pd.crosstab(titanic["Pclass"],titanic["Survived"]))
plt.show()
(titanic.groupby("Pclass")["Survived"].mean()*100).plot(kind="bar")
plt.show()
sns.clustermap(pd.crosstab(titanic["Parch"],titanic["Survived"]))
plt.show()
sns.pairplot(iris)
plt.show()
sns.pairplot(iris,hue="species")
plt.show()
sns.lineplot(x=passengers["Year"],y=passengers["Passengers"])
plt.show()
sns.heatmap(passengers.pivot_table(values="Passengers",index="Month",columns="Year"))
plt.show()

sns.clustermap(passengers.pivot_table(values="Passengers",index="Month",columns="Year"))
plt.show()