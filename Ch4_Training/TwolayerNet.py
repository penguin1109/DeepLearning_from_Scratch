from lossFunction import cross_entropy_error
import pickle
from dataset.mnist import load_mnist
from common.functions  import *
from common.gradient import numerical_gradient
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #활성화 함수로 sigmoid 함수 사용
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

    #마지막 출력층에서 사용할 softmax 함수
    def softmax(self, x):
        c = np.max(x)
        pred = np.exp(x-c)
        return pred/np.sum(pred)

    #출력값을 return하는 predict함수 
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        A1 = np.dot(x, W1) + b1
        Z1 = self.sigmoid(A1)
        A2 = np.dot(Z1, W2) + b2
        Z2 = self.softmax(A2)

        return Z2

    #손실 계산 함수
    #cross_entropy_error을 이용한다.
    def loss(self, x, label):
        y = self.predict(x)
        return cross_entropy_error(y, label)

    #mini batch를 적용하여서 학습을 시키고 있기 때문에, 그리고 y가 softmax함수를 통해 확룰값으로 나오기 떄문에
    #np.argmax()를 적용해서 클래스를 구하고 해당 값과 label값이 동일한, True를 갖는 boolean값에 대해서
    #np.sum()으로 정확하게 예측한 개수를 세어준다.
    def accuracy(self, x, label):
        y = self.predict(x)
        p = np.argmax(y, axis = 1)
        answer = np.argmax(label, axis = 1)
        accuracy = np.sum(p == answer) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, label):
        #이는 일반적으로 딥러닝에서 많이 사용되는 오차역전파를 사용하여 구현한 것은 아니다.
        #오차 역전파를 사용한 것을 아니지만 그래도 '최적의 parameter'을 찾기위해서 손실함수를 parameter로 편미분하여주었다.
        #오차 역전파와 달리 순방향으로 미분을 해준다는 차이가 존재한다.
        loss_w = lambda w: self.loss(x, label)
        grad = {}
        grad['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grad