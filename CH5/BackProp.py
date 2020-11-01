import numpy as np
import matplotlib.pyplot as plt

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, grad):
        dx = grad * self.y
        dy = grad * self.x
        return dx, dy

#덧셈 연산을 수행하는 노드의 경우에는 별다른 여과없이 역전파에서 그대로 입력을 하면 된다.
class AddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y
    def backward(self, grad):
        dx = grad * 1
        dy = grad * 1
        return dx, dy
    
class Relu:
    def __init__(self):
        #self.mask는 True/False로 구성이 된 Numpy 배열이다.
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
    def backward(self, grad):
        grad[self.mask] = 0
        dx = grad
        return grad

x = np.array([[1.0, -0.5], [-2.0,3.0]])
layer = Relu()
layer.forward(x)
grad = layer.backward(x)
#layer.mask를 구해보면 [[False True][True False]]로 출력이 됨

#Sigmoid계층은 y의 값으로, 즉 순전파의 출력만으로 역전파의 계산이 가능하다.
#따라서 계층을 구현을 해 줄 떄에 (1-y) * y * (이전 역전파에서의 입력된 값)
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        self.out = 1/(1 + np.exp(-x))
        return self.out
    def backward(self, grad):
        dx = (1-self.out) * grad * (self.out)
        return dx

sigmoid = Sigmoid()
sigmoid.forward(x)
