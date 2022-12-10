import numpy as np

a = np.array([0, 1, 0, 0, 0, 1, 1, 0, 1, 0])
b = np.arange(10, dtype=object)

b[a == 0] = np.array([0, 0, 0], np.uint8)

print(b)
