import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0, dtype = np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x = np.array([1.0, 0.5])
w1 = np.array([[0.1, 0.3, 0.5],[0.2,0.4,0.6]])
b1 = np.array([0.1,0.2,0.3])
A1 = np.dot(x,w1) + b1
Z1 = sigmoid(A1)

w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1,0.2])
A2 = np.dot(Z1, w2) + b2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

w3 = np.array([[0.1, 0.3],[0.2,0.4]])
b3 = np.array([0.1,0.2])
A3 = np.dot(Z2, w3) + b3
Y = identity_function(A3)

print(Y)