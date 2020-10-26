import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(normalize = True, one_hot_label= True)

def mean_squared_error(y,label):
    return 0.5 * np.sum((y-label)**2)

def cross_entropy_error(y, label):
    #delta는 혹시나 log에 입력되는 값이 0이나 음수일 상황을 대비해서 더해주는 값이다.(아주 작은 값으로 설정해줌)
    delta = 1e-7
    return -np.sum(label * np.log(y+delta))

batch_size = 10
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]

def batch_cee_onehot(y, label):
    if y.ndim == 1:
        label = label.reshape(1, label.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(label * np.log(y+delta))/batch_size

def batch_cee(y, label):
    if y.ndim == 1:
        label = label.reshape(1, label.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), label] + delta))/batch_size

def numerical_gradient(f,x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        now = x[i]
        fx1 = f(now - h)
        fx2 = f(now + h)
        gradient[i] = (fx2-fx1)/2*h
    
    return gradient
def func1(x):
    return x**2 + x**5

print(numerical_gradient(func1, np.array([3.0,4.0])))

def gradient_descent(f, init_x, learning_rate = 0.001, epoch = 100):
    x = init_x

    for i in range(epoch):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
    return x

