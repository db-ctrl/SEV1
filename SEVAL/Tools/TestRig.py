import numpy as np




# Initialise 2D Array
a = np.array([[" ".join(" ") for x in range(10)] for y in range(5)])


a[1, 8] = "muuuuuuuuuuuuuuu"


print(a)
