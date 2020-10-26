import numpy as np
from lossFunction import batch_cee
from common.functions import softmax
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist

(x_train,y_train), (x_test, y_test) = load_mnist(one_hot_label = False)

class SimpleNet():
    def __init__(self):
        self.w = np.random.randn(784,10)
    
    def predict(self, x):
        return np.dot(x, self.w)
    
    def loss(self, x):
        y = self.predict(x)
        y = softmax(y)
        return batch_cee(y, y_train)

net = SimpleNet()
print(net.loss(x_train))