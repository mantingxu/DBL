import numpy as np

file = np.load('../numpy/myResTest0601.npy', allow_pickle=True)
for i in file:
    print(i[1])
    print(type(i[1]))
