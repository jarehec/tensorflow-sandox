import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

dataframe = pd.read_csv('breast-cancer-wisconsin.data')
dataframe.replace('?', -99999, inplace=True)
dataframe.drop(['id'], 1, inplace=True)

X = np.array(dataframe.drop(['class'], axis=1))
y = np.array(dataframe['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

knn_classifier = neighbors.KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

accuracy = knn_classifier.score(X_test, y_test)
print(accuracy)
example_measures = np.array([4,2,1,1,1,2,3,2,1])
prediction = knn_classifier.predict(example_measures.reshape(1, -1))
print(prediction)