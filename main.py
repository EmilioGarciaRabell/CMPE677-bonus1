from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
U = iris.data
Y = np.zeros((len(iris.target), 3))
Y[np.arange(iris.target.size), iris.target] = 1

# A) Write code to randomly divide the data into two sets: 
# U_train, Y_train and U_test, Y_test. 80%
# of the data should be in U_train and 20% should be in U_test
rng = np.random.default_rng(seed=42)
indices = np.arange(len(U))
rng.shuffle(indices)

split_idx80 = int(len(U) * 0.8)

train_indices = indices[:split_idx80]
test_indices = indices[split_idx80:]

U_train = U[train_indices]
Y_train = Y[train_indices]

U_test = U[test_indices]
Y_test = Y[test_indices]

# B) Perform classification using k nearest neighbors with k = 5, and Euclidean distance as the distance
# metric. Report training and test accuracy

