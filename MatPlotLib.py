import matplotlib.pyplot as plt

# age = [18,20,25,23]
# weight = [52,45,85,42]

data = {
    "name" :["Aryan","Anku","Mohit","Isha"],
    "age":[18,20,25,23],
    "employed":["NO","Yes","Yes","No"],
    "weight":[52,45,85,42],
    "x":[1,0,1,0]
} 
colors = ["green" if i == "Yes" else "red" for i in data["employed"]]
# plt.plot(data["age"],data["weight"])
# plt.show()

x = plt.scatter(data["age"],data["weight"],c=colors)
plt.xlabel("Age")
plt.ylabel("Weight")
plt.title("Age vs Weight")
plt.show()