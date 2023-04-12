import numpy as np


datasets = [1,2,3]

for i in datasets:
    d = np.genfromtxt(f"Al{i}.csv", delimiter=',')
    new_d = d[:,:3]
    np.save(f"Al{i}", new_d)
