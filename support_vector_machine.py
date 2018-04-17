import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm

dataframe = pd.read_csv('breast-cancer-wisconsin.data')
dataframe.replace('?', -99999, inplace=True)
dataframe.drop(['id'], 1, inplace=True)

X = np.array(dataframe.drop(['class'], axis=1))
y = np.array(dataframe['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

svm_classifier = svm.SVC()
svm_classifier.fit(X_train, y_train)

confidence = svm_classifier.score(X_test, y_test)
print(confidence)
example_measures = np.array([4,2,1,1,1,2,3,2,1])
prediction = svm_classifier.predict(example_measures.reshape(1, -1))
print(prediction)