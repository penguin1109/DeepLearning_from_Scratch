from typing import OrderedDict
import numpy as np
from Convolution import Convolution
from Pooling import Pooling

#Convolution-ReLU-Pooling-Affine(Dense)-ReLU-Affine(Dense)-Softmax

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Dense:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

class SimpleConvNet:
    #class의 __init__ 함수내에서는 모든 가중치밒 매개변수를 초기값을 설정해주기 위해서 초기화를 한다.
    def __init__(self, input_dim = (1,28,28),
            conv_parameter = {'filter_num' : 32, 'filter_size' : 4, 
            'stride' : 1, 'padding' : 0},
            hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        filter_num = conv_parameter['filter_num']
        filter_size = conv_parameter['filter_size']
        filter_stride = conv_parameter['stride']
        filter_pad = conv_parameter['padding']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad)/filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2)*(conv_output_size/2))

        self.params = {}
        #np.random.randn을 하면 정규분포의 형태로 임의의 수를 지정해 준다.
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        #편향 값을 1개이기 떄문에 계층의 개수만큼 있으면 된다
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                self.params['b1'],
                                conv_parameter['stride'],
                                conv_parameter['padding'])
        self.layers['ReLU'] = Relu()                            
        self.layers['MaxPool1'] = Pooling(pool_h=2, pool_w=2)
        self.layers['Dense'] = Dense(self.params['W2'],
                                self.params['b2'])
        self.layers['ReLU2'] = Relu()
        self.layers['Dense2'] = Dense(self.params['W3'],
                                self.params['b3'])
        self.last_layer = SoftmaxWithLoss()

    #이미지의 특징 추출한 값을 출력하는 함수 predict
    def predict(self, x):
        for layers in self.layers.values():
            x = layers.forward(x)
        return x

    #predict의 출력값을 바탕으로 손실을 계산해서 (softmax with loss)출력한다.
    def loss(self, x, t):
        output = self.predict(x)
        return self.last_layer.forward(output)



