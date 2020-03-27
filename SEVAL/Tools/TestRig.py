import numpy as np

true_k = 250

a = np.array([[0 for x in range(true_k)], [4, 5, 6]])

b = np.array([[400], [800]])

newArray = np.append(a, b)

print(newArray)