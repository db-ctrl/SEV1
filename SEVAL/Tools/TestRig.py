import numpy as np

# Initialise 2D Array
a = np.array([[0 for x in range(10)] for y in range(5)])


a[1, 8] = 1


print(a)
