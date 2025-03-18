import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

DATASET = "dtA"

train_data = np.genfromtxt(f'dataset/{DATASET}/train_data.csv', delimiter=',', skip_header=True)
x_train = train_data[:, :-1]
y_train = train_data[:, -1]

test_data = np.genfromtxt(f'dataset/{DATASET}/test_data.csv', delimiter=',', skip_header=True)
x_test = test_data[:, :-1]
y_test = test_data[:, -1]

sscaler = StandardScaler()
x_train = sscaler.fit_transform(x_train)
x_test = sscaler.transform(x_test)

score_list = []

for k in range(1, 50):

    knn = KNeighborsClassifier(n_neighbors = k, weights = 'distance')

    knn.fit(x_train, y_train)

    y_predict = knn.predict(x_test)

    score = knn.score(x_test, y_test)
    score_list.append(score)

print("Max Score", max(score_list))

plt.plot(score_list)
plt.xlabel('neighbors')
plt.ylabel('score')
plt.show()