import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

train_data = np.genfromtxt('dataset/dtA/train_data.csv', delimiter=',', skip_header = True)
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

choices = list(combinations([i for i in range(8)], 2))

ROW = 4
COL = 7

for i in range(ROW * COL):
    plt.subplot(ROW, COL, i+1)
    plt.scatter(x_train[:, choices[i][0]], x_train[:, choices[i][1]], s = 2, c = y_train)

plt.show()