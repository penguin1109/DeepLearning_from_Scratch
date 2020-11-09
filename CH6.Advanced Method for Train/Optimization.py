import numpy as np
import matplotlib.pyplot as plt

#SGD(확률적 경사 하강법)을 보면 매개변수로 미분한 손실 함수의 기울기를 이용해서 학습률에 곱하여 매개변수를 갱신해 준다.
class SGD:
    def __init__(self, learning_rate = 0.01):
        self.lr = learning_rate
    
    def update(self, parameter, gradient):
        for key in parameter.keys():
            parameter[key] -= self.lr * gradient[key]

#그러나 SGD에 존재하는 문제점이 있는데, 바로 비등방성 함수(방향에 따라 기울기가 달라지는 성질을 가진 함수)에서는 탐색 경로가 비효율적이라는 것이다.
#그리고 SGD가 비효율적인 지그재그 모양으로 진행이 되는 원인이 기울어진 방향이 본래의 최솟값과 다른 방향을 가리키기 때문이다.
#따라서 이러한  단점 개선을 위해 Momentum, Adagrad, Adam등을 이용할 수 있다.

#원래 SGD는 그냥 가중치에 학습률과 손실함수의 미분을 빼 주었다면 Momentum기법을 이용할떄는
#갱신 속도 v를 개선해서 해당 값을 개선하고자 하는 파라미터에 적용해준다.
class Momentum:
    def __init__(self, learning_rate = 0.01, momentum = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        #v는 물체의 속도를 의미한다. 따라서 초기화 할 때에는 아무런 값도 담지 않는다.
        self.v = None
    
    def update(self, parameter, gradient):
        if self.v is None:
            self.v = {}
            for key, value in parameter.items():
                self.v[key] = np.zeros_like(value)
        for key in parameter.keys():
            parameter[key] += self.v[key]
#Momentum을 사용하면 SGD를 사용할 때와 달리 갱신 경로의 지그재그 정도가 덜하다.

#2. 학습률 감소(Learning Rate Decay)
#Adagrad는 각각의 매개변수에 맞춤형 값을 만들어 준다.
#h라는 새로운 변수를 이용해서 손실함수의 매개변수에 대한 기울기의 제곱을 계속 더해주고 매개변수를 갱신할 떄는 h의 루트값을 곱해서 학습률을 조정한다.
#이렇게 되면 매개변수의 원소 중에서 많이 움직인 원소의 학습률이 낮아지게 된다.
#물론 이렇게 먼 과거의 기울기 정보까지 이용한다면 어느 순간 갱신량이 0이 되어서 학습률이 더는 안낮아질 수 있기 때문에 새로운 기울기의 정보를 더 크게 반영하는 RMSProp이라는 방법을 사용하기도 한다.

class AdaGrad:
    def __init__(self, learning_rate = 0.01):
        self.lr = learning_rate
        self.h = None
    
    def update(self, gradient, parameter):
        if self.h is None:
            self.h = {}
            for key, value in parameter.items():
                self.h[key] = np.zeros_like(value)
        for key in parameter.keys():
            self.h[key] += gradient[key] * gradient[key]
            parameter[key] -= self.lr * gradient[key] / (np.sqrt(self.h[key]) + 1e-7)


#Adam(Adagrad + Momentum)
#하이퍼파라미터의 편향 보정이 가능하다는 특징을 갖고 있다.
#Adam은 하이퍼파라미터를 3개를 설정하는데, 하나는 학습률이고 나머지 두개는 1차 모멘텀용과 2차 모멘텀용이다.
#일반적으로 1차 모멘텀은 0.9, 2차 모멘텀은 0.999로 설정하면 좋은 결과를 얻을 수 있다고 한다.

class Adam:
    def __init__(self, learning_rate = 0.001):
        self.lr = learning_rate
        self.b1 = 0.9
        self.b2 = 0.999
        self.m = None
        self.t = None
        self.v = None
    
    def update(self, gradient, parameter):
        if self.m is None:
            self.m = {}
            for key, value in parameter.items():
                self.m[key] = np.zeros_like(value)
        if self.t is None:
            self.t = {}
            for key, value in parameter.items():
                self.t[key] = np.zeros_like(value)
        if self.v is None:
            self.v = {}
            for key, value in parameter.items():
                self.v[key] = np.zeros_like(value)

        for key in parameter.keys():
            self.m[key] = self.b1 * self.m[key] + (1-self.b1) * gradient[key]
            self.v[key] = self.b2 * self.v[key] + (1-self.b2) * gradient[key]**2
            self.m[key] = self.m[key] / (1-self.b1)
            self.v[key] = self.v[key] / (1-self.b2)
            parameter[key] -= self.lr * self.m[key] / (np.sqrt(self.v) + 10**-8)
            #1e-8 = 10**-8이었다.

#일반적으로 학습을 하고자 랗 때에 Adam을 많이 사용하고, 그에 못지 않게 SGD만으로도 충분히 학습이 가능하다.





