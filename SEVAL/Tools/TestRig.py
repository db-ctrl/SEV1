
import numpy as np

# Initialise 2D Array
c = np.array([[0 for x in range(10)] for y in range(5)])
p = 0

for i in (c[0, ...]):
    c[..., i] = p
    p += 1

print (c)
