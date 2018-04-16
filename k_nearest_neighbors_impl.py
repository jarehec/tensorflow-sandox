import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
from collections import Counter

# eucildean_distance = ((p1[0] - q1[0])**2 + (p1[1] - q1[1])**2)**0.5
# euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    for group in data:
        for feature in data[group]:
            euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
complete_data = df.astype(float).values.tolist()
random.shuffle(complete_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = complete_data[:-int(test_size*len(complete_data))]
test_data = complete_data[-int(test_size*len(complete_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data: # add 20% data, omit class (last column)
    test_set[i[-1]].append(i[:-1])

total, correct = 0, 0
for group in test_set:
    for data in test_set[group]:
        prediction = k_nearest_neighbors(train_set, data, k=5)
        if group == prediction:
            correct += 1
        total += 1
print('Accuracy: ', correct/total)

# Simple dataset test
# dataset = {'k': [[1,2],[2,3],[3,1]], 'r':[[6, 5],[7,7],[8,6]]}
# new_features = [0,1]
# prediction = k_nearest_neighbors(dataset, new_features, k=1)
# [[plt.scatter(j[0], j[1], c=i) for j in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], c=prediction)
# plt.show()