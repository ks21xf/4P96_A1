import numpy
from sklearn.datasets import fetch_openml

# Fetch the dataset
fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')

# The data is stored in the .data and .target attributes
X, y = fashion_mnist.data, fashion_mnist.target
# X is a NumPy array of shape (70000, 784) (flattened images)
# y is a NumPy array of shape (70000,) (labels as strings)

print(X.shape)