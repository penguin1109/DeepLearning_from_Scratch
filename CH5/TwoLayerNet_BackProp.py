from collections import OrderedDict
import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, grad):
        #W, 즉 가중치 배열의 전치(transpose)와의 행렬곱 계산
        dx = np.dot(grad, self.W.T)
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis = 0)
        return dx

class Relu:
    def __init__(self):
        #self.mask는 True/False로 구성이 된 Numpy 배열이다.
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, grad):
        grad[self.mask] = 0
        dx = grad
        return grad

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        #y는 softmax의 출력, t는 정답 label의 값
        self.y = None
        self.t = None

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
        x = x - np.max(x) # 오버플로 대책
        return np.exp(x) / np.sum(np.exp(x))

    def cross_entropy_error(self,y, label):
        delta = 1e-7
        #label값은 one-hot-encoding된 상태일 것이기 떄문에 정답 label에만 1의 가중치가 부여되고
        #나머지는 0으로 처리될 것이다.
        return -np.sum(label * np.log(y+delta))

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss
    
    #softmax 함수와 cross_entropy_error, 항등 함수와 Mean_Squared_Error 끼리 연결이 되어야
    #역전파를 수행할 당시에 제대로 학습이 가능하다.

    #다시 강조하지만 신경망 학습의 목적은 신경망의 출력이 정답 레이블과 가따워지도록 가중치 매개변수의 값을 조정하는 것이다.
    def backward(self, grad = 1):
        batch_size = self.t.shape[0]
        #전파하는 batch의 개수로 전파하는 값을 나누어서
        #데이터 1개당 오차를 앞의 계층으로 전파한다.
        dx = (self.y - self.t) / batch_size
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['b2'] = np.zeros(output_size)

        #OrderedDict()는 순서가 있는 dictionary 자료구조로, dictionary에 추가한 순서를 기억하게 된다.
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        #one-hot-encoding된 정답 레이블을 이용한다면
        if t.ndim != 1:t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def backprop(self, x, t):
        self.loss(x, t)
        grad = 1
        grad = self.lastLayer.backward(grad)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            grad = layer.backward(grad)
        
        #행렬곱 연산에서 계산하는 가중치와 편향값의 미분값을 저장해 준다.
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads



#그렇다면 수치 미분을 왜 사용하는지 간단하게 알아본다면
#우선 구현이 쉬워서 버그가 잘 발생하지 않는다는 장점이 있고, 때문에 상대적으로 오류가 발생하기 쉬운 오차역전파의 기울기 결과를 확인하는데에 사용된다.
        
    
    




