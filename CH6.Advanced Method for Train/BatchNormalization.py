#1. 학습 속도의 개선
#2. Parameter의 초깃값에 크게 의존하지 않는다.
#3. Overfitting을 억제해 준다.(Dropout의 필요성 감소)

#Batch Normalization의 기본 아이디어는 각 층에서의 활성화 값이 적당히 분포되도록 조정하는 것이다.
#학습시에 mini batch를 단위로 정규화를 한다. (데이터 분포가 평균이 0, 분산이 1이 되도록)

import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from Optimization import SGD, Adam

(x_train, y_train), (x_test, y_test) = load_mnist(normalization  =True)

x_train = x_train[:1000]
y_train = y_train[:1000]

epoch = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#batch normalization 계층은 활성화 함수의 앞이나 뒤에 삽입하여 데이터의 분포가 덜 치우치도록 한다.
#이렇게 각 batch normalization 계층마다 정규화된 데이터에 고유헌 확대와 이동의 변환을 수행한다.
#따라서 확대를 담당하는 계수 r과 이동을 담당하는 계수 b를 이용하여 y = r*x+b를 하여 r의 초깃값은 1, b의 초깃값은 0으로 설정한다.

#즉, 결과적으로 가중치의 초기값을 제대로 선정해야 하는 필요성과 같은 이유이다.
#결국 목적은 각 층의 활성화 값의 분포를 적절하게 퍼지게 하기 위해서 데이터 값들을 결국 데이터의 값에 변형을 줌으로서
#데이터 자체의 분초의 형태를 일정하게 유지시켜 주는 것이다.
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

        # 시험할 때 사용할 평균과 분산
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward(역전파) 시에 사용할 중간 데이터
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_valid=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_valid)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_valid):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_valid:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    #여기서부터 batch normalization의 역전파 진행
    #dout는 출력값을 미분한 값을 의미(마지막 출력층에서)
    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        #이동을 담당하는 계수의 미분값
        dbeta = dout.sum(axis=0)
        #확장을 담당하는 계수의 미분값
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

#거의 모든 경우에 batch normalization(배치 정규화)를 사용하게 되면 학습의 속도가 빠르고 
#가중치의 초깃값이 잘 분포되어 있지 않은 상항에서도 학습이 진행이 잘 된다.
#따라서 신경망 학습 시에 activation function의 앞이나 뒤에 넣어주는 것이 당연히 훨씬 성능을 높여줄 것이다.