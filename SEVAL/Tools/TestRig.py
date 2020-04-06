import numpy as np

array = [111.4, 21, 60, 30.908823293, 11.45]

normal_array = array / (np.linalg.norm(array))

print(normal_array)
