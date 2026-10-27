from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
U = iris.data
Y = np.zeros((len(iris.target), 3))
Y[np.arange(iris.target.size), iris.target] = 1
