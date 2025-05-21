# In this file, we will learn about vectors and matrix operations
# in Python using NumPy.

import numpy as np

# Demonstrating a 0D array (scalar)
scalar = np.array(42)
print("Dimension of 0D array (scalar):", scalar.ndim)
print("Shape of 0D array:", scalar.shape)

# Demonstrating a 1D array (vector)
vector = np.array([1, 2, 3])
print("Dimension of 1D array (vector):", vector.ndim)
print("Shape of 1D array:", vector.shape)

# Demonstrating a 2D array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Dimension of 2D array (matrix):", matrix.ndim)
print("Shape of 2D array:", matrix.shape)

# Demonstrating a 3D array (tensor)
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("Dimension of 3D array (tensor):", tensor.ndim)
print("Shape of 3D array:", tensor.shape)

# Matrix operations
A = np.array([8, 4, 5])
B = np.array([1, 2, 3])
print("Sum of A and B:", A + B)  # Element-wise addition
print("Difference of A and B:", A - B)  # Element-wise subtraction

A = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
B = np.array([[5, 6, 7, 8],[1, 2, 3, 4]])
print('두 행렬의 합 :')
print(A + B)
print('두 행렬의 차 :')
print(A - B)

# Dot product of two vectors
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print("Dot product of A and B:", np.dot(A, B))  # 1*4 + 2*5 + 3*6 = 32

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Dot product of A and B:")
print(np.matmul(A, B))  # [[19 22]
#                          [43 50]]

# Samples and Features
# In machine learning, we often refer to the data as samples and features.
# A sample is a single data point, and features are the individual attributes or properties of that sample.

