import numpy as np

a = np.array([10,20,30,40,50])
print(a + 10)
print(a*2)


# np.array()	      #Create an array from a list
# np.zeros(5)	      #Create 5 zeros
# np.ones(5)	      #Create 5 ones
# np.arange(1, 6)	  #Create numbers from 1 to 5
# np.full(5, 7)	      #Create 5 elements containing 7

print(np.zeros(5))
print(np.ones(5))
print(np.arange(1, 6))
print(np.full(5, 7))

marks = np.array([45, 67, 89, 32, 95, 76, 54, 88])

print("Average",np.mean(marks))
print("Heighest marks",np.max(marks))
print("lowest marks",np.min(marks))
print("marks greater than 70",marks[marks>70])
b = marks.reshape(2,4)
print(b)
print(b.shape)

a = np.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])

print(np.sum(a, axis=0))
print(np.sum(a, axis=1))