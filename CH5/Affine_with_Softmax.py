import numpy as np

#Affine 계층
#실제로 신경파의 연산을 하게 되면 np.dot을 이용해서 행렬곱의 연산을 수행한다.
#따라서 다차원 배열의 계산을 하게 되는데, 이때 중요한 것은 차원의 원소의 수를 일치시키는 것이다.
#이렇게 신경망의 순전파 과정에서 수행하는 행렬의 곱은 Affine Transformation이라고 한다.

class Affine:
    def __init__(self, W, b):
        #W는 weight b는 bias
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + b
        return out
    def backward(self, grad):
        #W, 즉 가중치 배열의 전치(transpose)와의 행렬곱 계산
        dx = np.dot(grad, self.W.T)
        self.dW = np.dot(self.x.T, grad)
        self.db = np.sum(grad, axis = 0)
        return dx


    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        #y는 softmax의 출력, t는 정답 label의 값
        self.y = None
        self.t = None

    def softmax(self, x):
        c = np.max(x)
        pred = np.exp(x + c)
        return pred/np.sum(pred)

    def cross_entropy_error(y, label):
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





        



