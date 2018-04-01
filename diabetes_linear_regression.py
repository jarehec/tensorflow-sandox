from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

diabetes = load_diabetes()
diabetes_X = diabetes.data[:, None, 2]
LinReg = LinearRegression()
X_trainset, X_testset, y_trainset, y_testset = train_test_split(
                                                            diabetes_X,
                                                            diabetes.target,
                                                            test_size=0.3,
                                                            random_state=7)
LinReg.fit(X_trainset, y_trainset)
plt.scatter(X_testset, y_testset, c='black')
plt.plot(X_testset, LinReg.predict(X_testset), linewidth=3)
plt.show()
