import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import mysql.connector
import requests
import seaborn as sns
from data_profiling import ProfileReport
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy.stats import stats 
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Binarizer


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

# # ===working with titanic dataset===
# # =====univariate analysis=====
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

# # ======================================
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
# ================================================================

titanic = pd.read_csv("titanic.csv")
y = ProfileReport(titanic,title="Titanic EDA Report")
y.to_file("titanic_report.html")


# =================================== feature scaling standardization ==================================


df = pd.read_csv("customer_purchase_10000.csv")

x_train,x_test,y_train,y_test = train_test_split(df.drop("Purchased",axis=1),df["Purchased"],test_size=0.2,random_state=0)
print(x_train.shape,x_test.shape)\
print(y_test)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)

print(x_train_scaled.describe())

sns.kdeplot(x_train_scaled["Age"])
sns.kdeplot(x_train_scaled["EstimatedSalary"])
plt.show()

lr_scaled = LogisticRegression()
lr_scaled.fit(x_train_scaled,y_train)

y_test_prediction = lr_scaled.predict(x_test_scaled)
print(y_test)
print("Accuracy = ",accuracy_score(y_test,y_test_prediction))
print(confusion_matrix(y_test,y_test_prediction))
print(classification_report(y_test,y_test_prediction))

# # =================================== feature scaling Normalization ==================================

df = pd.read_csv("wine_data.csv",header=None,usecols=[0,1,2])
df.columns = ["Class label","Alcohol","Malic acid"]
# print(df)

x_train,x_test,y_train,y_test = train_test_split(df.drop("Class label",axis=1),df["Class label"],test_size=0.1,random_state=0,stratify=df["Class label"])

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled,columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled,columns=x_test.columns)

lr_scaled = LogisticRegression()
lr_scaled.fit(x_train_scaled,y_train)

y_test_prediction = lr_scaled.predict(x_test_scaled)
# print(y_test)
print("Accuracy = ",accuracy_score(y_test,y_test_prediction))
print(confusion_matrix(y_test,y_test_prediction))
print(classification_report(y_test,y_test_prediction))

# # =================================== Encoding Categorical Data ==================================
from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv("customer.csv")
df = df.iloc[:,2:]
print(df.head())
print(df.columns)
x_train,x_test,y_train,y_test = train_test_split(df.drop("purchased",axis=1),df["purchased"],test_size=0.2,stratify=df["purchased"])
print(x_train)

oe = OrdinalEncoder(categories=[["Poor","Average","Good"],["School","UG","PG"]])
oe.fit(x_train)
x_train_ = oe.transform(x_train)
x_test_ = oe.transform(x_test)

x_train_ = pd.DataFrame(x_train_,columns=x_train.columns)
x_test_ = pd.DataFrame(x_test_,columns=x_test.columns)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(y_train)
y_train_ = le.transform(y_train)
y_test_ = le.transform(y_test)

lr = LogisticRegression()
lr.fit(x_train_,y_train_)

prediction = lr.predict(x_test_)

print("Accuracy",accuracy_score(y_test_,prediction))
print(confusion_matrix(y_test_,prediction))
print(classification_report(y_test_,prediction))

# =================================== without Column Transformer ==================================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("covid_toy.csv")
# print(df.isnull().sum())

x_train,x_test,y_train,y_test = train_test_split(df.drop("has_covid",axis=1),df["has_covid"],test_size=0.1,random_state=0,stratify=df["has_covid"])
# print(x_train.head())

si = SimpleImputer()

x_fever_train = si.fit_transform(x_train[["fever"]])
x_fever_test = si.transform(x_test[["fever"]])
# print(x_fever_train.shape)

oe = OrdinalEncoder(categories=[["Mild","Strong"]])

x_cough_train = oe.fit_transform(x_train[["cough"]])
x_cough_test = oe.transform(x_test[["cough"]])

ohe = OneHotEncoder(drop="first",sparse_output=False)

x_gender_city_train = ohe.fit_transform(x_train[["gender","city"]])
x_gender_city_test = ohe.transform(x_test[["gender","city"]])

x_age_train = x_train[["age"]]
x_age_test = x_test[["age"]]

x_train_transformed = np.concatenate((x_age_train,x_gender_city_train,x_fever_train,x_cough_train),axis=1)
x_test_transformed = np.concatenate((x_age_test,x_gender_city_test,x_fever_test,x_cough_test),axis=1)

x_train_df = pd.DataFrame(x_train_transformed)
# print(x_train_df)

# # =================================== with Column Transformer ==================================
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(transformers=[
    ("tf1",SimpleImputer(),["fever"]),
    ("tf2",OrdinalEncoder(categories=[["Mild","Strong"]]),["cough"]),
    ("tf3",OneHotEncoder(drop="first",sparse_output=False),["gender","city"])
    ],remainder="passthrough")

x_train_transformed_ = transformer.fit_transform(x_train)
x_test_transformed_ = transformer.transform(x_test)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train_ = le.fit_transform(y_train)
y_test_ = le.transform(y_test)


lr = LogisticRegression()
lr.fit(x_train_transformed_,y_train_)

prediction = lr.predict(x_test_transformed_)

print("Accuracy",accuracy_score(y_test_,prediction))
print(confusion_matrix(y_test_,prediction))
print(classification_report(y_test_,prediction))
print(y_train.value_counts())
print(y_test.value_counts())

# # =================================== without using Pipelines ==================================

df = pd.read_csv("titanic.csv")

df.drop(columns=["PassengerId","Name","Cabin","Ticket"],inplace=True)

print(df.head())
print(df.isnull().sum())
x_train,x_test,y_train,y_test = train_test_split(df.drop("Survived",axis=1),df["Survived"],test_size=0.1,random_state=0,stratify=df["Survived"])
print(x_train.head())

from sklearn.impute import SimpleImputer

si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy="most_frequent")

x_age_train = si_age.fit_transform(x_train[["Age"]])
x_age_test = si_age.transform(x_test[["Age"]])

x_emarked_train = si_embarked.fit_transform(x_train[["Embarked"]])
x_emarked_test = si_embarked.transform(x_test[["Embarked"]])

from sklearn.preprocessing import OneHotEncoder

ohe_sex = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
ohe_embarked = OneHotEncoder(sparse_output=False,handle_unknown="ignore")

x_sex_train = ohe_sex.fit_transform(x_train[["Sex"]])
x_sex_test = ohe_sex.transform(x_test[["Sex"]])

x_emarked_train = ohe_embarked.fit_transform(x_emarked_train)
x_emarked_test = ohe_embarked.transform(x_emarked_test)

x_train_rem = x_train.drop(["Age","Sex","Embarked"],axis=1)
x_test_rem = x_test.drop(["Age","Sex","Embarked"],axis=1)

x_train_transformed = np.concatenate((x_train_rem,x_age_train,x_sex_train,x_emarked_train),axis=1)
x_test_transformed = np.concatenate((x_test_rem,x_age_test,x_sex_test,x_emarked_test),axis=1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(x_train_transformed,y_train)
prediction = dtc.predict(x_test_transformed)

print("Accuracy = ",accuracy_score(y_test,prediction))


=================================== with using Pipelines ==================================

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("titanic.csv")

df.drop(columns=["PassengerId","Name","Cabin","Ticket"],inplace=True)
x_train,x_test,y_train,y_test = train_test_split(df.drop("Survived",axis=1),df["Survived"],test_size=0.1,random_state=0,stratify=df["Survived"])

trf1 = ColumnTransformer([
    ("age_impute",SimpleImputer(),[2]),
    ("embraced_impute",SimpleImputer(strategy="most_frequent"),[6])
    ],remainder="passthrough")

trf2 = ColumnTransformer([
    ("impute_age_ebraced",OneHotEncoder(sparse_output=False,handle_unknown="ignore"),[1,6])
    ],remainder="passthrough")

trf3 = ColumnTransformer([
    ("scale",MinMaxScaler(),slice(0,10))
    ])

trf4 = SelectKBest(score_func=chi2,k=8)

trf5 = DecisionTreeClassifier()

# pipe = Pipeline([
#     ("trf1",trf1),
#     ("trf2",trf2),
#     ("trf3",trf3),
#     ("trf4",trf4),
#     ("trf5",trf5)
# ])

pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5)

pipe.fit(x_train,y_train)

from sklearn import set_config
set_config(display="diagram")

prediction = pipe.predict(x_test)

print("Accuracy = ",accuracy_score(y_test,prediction))

from sklearn.model_selection import cross_val_score

x = cross_val_score(pipe,x_train,y_train,cv=5,scoring="accuracy").mean()

print("Accuracy = ",x)


# # =================================== Box - Cox Transform ==================================

df = pd.read_csv("concrete_data.csv")

# print(df.head())
x_train,x_test,y_train,y_test = train_test_split(df.drop("Strength",axis=1),df["Strength"],test_size=0.1,random_state=0)

# lr = LinearRegression()
# lr.fit(x_train,y_train)
# y_pridiction = lr.predict(x_test)
# print(r2_score(y_test,y_pridiction))
# print(np.mean(cross_val_score(lr,x_test,y_test,scoring="r2")))

pt = PowerTransformer(method="box-cox")

x_train_transformed = pt.fit_transform(x_train+0.00001)
x_test_transformed = pt.transform(x_test+0.00001)
# print(pd.DataFrame({"cols":x_train.columns,"lambda":pt.lambdas_}))

lr = LinearRegression()
lr.fit(x_train_transformed,y_train)
prediction = lr.predict(x_test_transformed)
print(r2_score(y_test,prediction))
print(np.mean(cross_val_score(lr,x_test_transformed,y_test,scoring="r2")))

# # =================================== Yeo - Johnson Transform ==================================

df = pd.read_csv("concrete_data.csv")

# print(df.head())
x_train,x_test,y_train,y_test = train_test_split(df.drop("Strength",axis=1),df["Strength"],test_size=0.1,random_state=0)

# lr = LinearRegression()
# lr.fit(x_train,y_train)
# y_pridiction = lr.predict(x_test)
# print(r2_score(y_test,y_pridiction))
# print(np.mean(cross_val_score(lr,x_test,y_test,scoring="r2")))

pt = PowerTransformer()

x_train_transformed = pt.fit_transform(x_train+0.00001)
x_test_transformed = pt.transform(x_test+0.00001)
# print(pd.DataFrame({"cols":x_train.columns,"lambda":pt.lambdas_}))

lr = LinearRegression()
lr.fit(x_train_transformed,y_train)
prediction = lr.predict(x_test_transformed)
print(r2_score(y_test,prediction))
print(np.mean(cross_val_score(lr,x_test_transformed,y_test,scoring="r2")))


# # =================================== Discretization ==================================

df = pd.read_csv("titanic.csv",usecols=["Age","Fare","Survived"])
df.dropna(inplace=True)
x_train,x_test,y_train,y_test = train_test_split(df.drop("Survived",axis=1),df["Survived"],test_size=0.1,random_state=0)


# without Discretization

# dtc = DecisionTreeClassifier()
# dtc.fit(x_train,y_train)
# prediction = dtc.predict(x_test)
# print("Accuracy = ",accuracy_score(y_test,prediction))
# print(np.mean(cross_val_score(dtc,x_test,y_test,scoring="accuracy",cv=10)))

# with Discretization

kbins_age = KBinsDiscretizer(n_bins=10,encode="ordinal",strategy="quantile")
kbins_fare = KBinsDiscretizer(n_bins=10,encode="ordinal",strategy="quantile")

trf = ColumnTransformer([
    ("first",kbins_age,["Age"]),
    ("second",kbins_fare,["Fare"])
    ])

x_train_transformed = trf.fit_transform(x_train)
x_test_transformed = trf.transform(x_test)

dtc = DecisionTreeClassifier()

dtc.fit(x_train_transformed,y_train)
prediction = dtc.predict(x_test_transformed)
print("Accuracy = ",accuracy_score(y_test,prediction))
print(np.mean(cross_val_score(dtc,x_test_transformed,y_test,scoring="accuracy",cv=10)))


# # =================================== Binarization ==================================

df = pd.read_csv("titanic.csv",usecols=["Age","Survived"])
df.dropna(inplace=True)

x_train,x_test,y_train,y_test = train_test_split(df.drop("Survived",axis=1),df["Survived"],test_size=0.1,random_state=0)

# without Binarization

# dtc = DecisionTreeClassifier()
# dtc.fit(x_train,y_train)
# prediction = dtc.predict(x_test)
# print("Accuracy = ",accuracy_score(y_test,prediction))
# print(np.mean(cross_val_score(dtc,x_test,y_test,scoring="accuracy",cv=10)))

# with Binarization

bin = Binarizer(threshold=18)
x_train_transformed = bin.fit_transform(x_train)
x_test_transformed = bin.transform(x_test)

dtc = DecisionTreeClassifier()
dtc.fit(x_train_transformed,y_train)
prediction = dtc.predict(x_test_transformed)
print("Accuracy = ",accuracy_score(y_test,prediction))
print(np.mean(cross_val_score(dtc,x_test_transformed,y_test,scoring="accuracy",cv=10)))

# # =================================== Handling Mixed Variables ==================================

df = pd.read_csv("mixed_variables.csv")
# print(df.head())

df['number_numerical'] = pd.to_numeric(df["number"],errors='coerce',downcast='integer')

df['number_categorical'] = np.where(df['number_numerical'].isnull(),df['number'],np.nan)

print(df.head())
print("//////////////////////////////////////////")
df['cabin_num'] = df['Cabin'].str.extract(r'(\d+)') # captures numerical part
# df['cabin_cat'] = df['Cabin'].str[0] # captures the first letter
df['cabin_cat'] = df['Cabin'].str.extract(r'([A-Za-z]+)')

print(df.head())

# extract the last bit of ticket as number
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'],errors='coerce',downcast='integer')

# extract the first part of ticket as category
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(),np.nan,df['ticket_cat'])
print("//////////////////////////////////////////")
print(df.head(20))

# # =================================== Handling Date and Time Variables ==================================

df = pd.read_csv("messages.csv")
print(df.head())
df["date"] = pd.to_datetime(df["date"])
print(df.info())

df["date_year"] = df["date"].dt.year
df["date_month_no"] = df["date"].dt.month
df["date_month"] = df["date"].dt.month_name()
df["date_day_no"] = df["date"].dt.day
df["date_day"] = df["date"].dt.day_name()
df["date_day_no_of_week"] = df["date"].dt.day_of_week
df["date_weekend"] = np.where(df["date_day"].isin(["Saturday","Sunday"]),"yes","no")
df["date_quarter"] = df["date"].dt.quarter
print(df.sample(10))


import datetime

today = datetime.datetime.today()1
print(today - df['date'])
print((today - df['date']).dt.days)
print(np.round((today -df['date']) / np.timedelta64(1, 'M'),0))
df["time"] = df["date"].dt.time
df["hour"] = df["date"].dt.hour
df["second"] = df["date"].dt.second
df["minute"] = df["date"].dt.minute
df["microsecond"] = df["date"].dt.microsecond
print(df.sample(5))
print((today - df['date'])/np.timedelta64(1,'s'))
print((today - df['date'])/np.timedelta64(1,'m'))
print((today - df['date'])/np.timedelta64(1,'h'))