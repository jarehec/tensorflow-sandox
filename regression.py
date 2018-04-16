import numpy as np
import random
from numpy import mean
from matplotlib import pyplot as plt


# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs) * mean(xs)) - mean(xs * xs)))

    b = mean(ys) - (m * mean(xs))
    return m, b


def squared_error(ys_og, ys_line):
    return sum((ys_line - ys_og)**2)


def coefficient_of_determination(ys_og, ys_line):
    ''' r^2 = 1 - SquaredError of ys_og / SquaredError of ys_mean_line '''
    y_mean_line =  [mean(ys_og) for y in ys_og]
    squared_error_regr = squared_error(ys_og, ys_line)
    squared_error_y_mean = squared_error(ys_og, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


xs, ys = create_dataset(40, 15, correlation='neg')
m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m * x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)
plt.scatter(xs, ys, label='data')
plt.plot(xs, regression_line, c='red', label='regression_line')
plt.legend(loc=4)
plt.show()
