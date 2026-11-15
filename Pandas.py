import pandas as pd

data = {
    "Name" :["Aryan","Anku","Mohit","Isha"],
    "Age":[18,20,25,23],
    "Employed":["NO","Yes","Yes","No"],
    "Weight":[52,45,85,42]
}
df = pd.DataFrame(data)
# print(df)
# df.info()
# print(df.iloc[1])
df2 = pd.DataFrame(data)
df2 = df2.iloc[:,:3]
print(df2)