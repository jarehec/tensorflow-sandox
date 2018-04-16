import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')

class support_vector_machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    def fit(self, data):
        ''' train '''
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}
        transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
        all_data = []
        for yi in self.data:
            for feature_set in self.data[yi]:
                for feature in feature_set:
                    all_data.append(feature)
        self.max_feature_val = max(all_data)
        self.min_feature_val = min(all_data)
        del all_data

        step_sizes = [self.max_feature_val * 0.1,
                      self.max_feature_val * 0.01,
                      # expensive computation
                      self.max_feature_val * 0.001]
        # extremely expensive
        b_range_multiple = 2
        b_multiple = 5
        latest_optimum = self.max_feature_val * 10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1 *(self.max_feature_val * b_range_multiple),
                                    self.max_feature_val * b_range_multiple,
                                    step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # yi(xi.w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                    print('optimized a step!')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            # get the smallest magnitude
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2


    def predict(self, features):
        ''' sign (x.w + b) '''
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        # if the classification isn't zero, and we have visualization on, we graph
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200,
                            marker='*', c=self.colors[classification])
        else:
            print('feature set', features, ' is on the decision boundary')
        return classification

    def visualize(self):
        for i in data_dict:
            [self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in data_dict[i]]
        # hyperplane = x.w + b
        # psv = 1
        # nsv = -1
        # db (decision boundary)= 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]
        data_range = (self.min_feature_val * 0.9, self.max_feature_val * 1.1)
        hypr_x_min = data_range[0]
        hypr_x_max = data_range[1]

        # positive support vector hyperplane
        # w.x + b = 1
        psv1 = hyperplane(hypr_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hypr_x_max, self.w, self.b, 1)
        self.ax.plot([hypr_x_min, hypr_x_max], [psv1,psv2])

        # negative support vector hyperplane
        # w.x + b = -1
        nsv1 = hyperplane(hypr_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hypr_x_max, self.w, self.b, -1)
        self.ax.plot([hypr_x_min, hypr_x_max], [nsv1,nsv2])

        # positive support vector hyperplane
        # w.x + b = 0
        db1 = hyperplane(hypr_x_min, self.w, self.b, 0)
        db2 = hyperplane(hypr_x_max, self.w, self.b, 0)
        self.ax.plot([hypr_x_min, hypr_x_max], [db1,db2])

        plt.show()


data_dict = {-1: np.array([[1,7], [2, 8], [3,8]]),
              1: np.array([[5,1], [6,-1], [7,3]])}

predict_data = [[0,10],[4,5],[-1,-1]]

svm = support_vector_machine()
svm.fit(data=data_dict)
[svm.predict(p) for p in predict_data]
svm.visualize()