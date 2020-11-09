import numpy as np
from Optimization import SGD

#OverFitting이 발생할때에 train과 validataion data의 accuracy의 크기의 차이가 많이 난다.
#즉, validation과 test data의 정확도가 train보다 훨씬 크게, 100%에 가깝게 된다.
#이는 신경망이 훈련 데이터에만 지나치게 적응이 되어 그 외의 데이터는 제대로 대응하지 못하는 상태를 의미한다.

#Overfitting을 억제하는 하나의 방법으로 모든 가중치의 각각의 손실 함수에 0.5 * 람다 * 가중치를 더하는 방법이 존재한다.
#따라서 역전파의 진행 과정에서는 오차역전파법에 따른 결과에 정규화항을 미분한 람다 * 가중치의 값을 더해준다.

#즉, overfitting이 발생하는 원인의 하나인 큰 가중치 매개변수를 막기 위해서 손실함수의 값을 키워서 가중치가 커지는 것을 억제하는 것이다.

#L2 Norm의 경우에는 어떤 계층으로 설정하기보다는 일반적으로 훈련을 시키는 과정에서 loss를 계산할때에 더해주는 값이기 때문에 class L2Norm으로 구현할 필요가 없다.
#만약 훈련하는 네트워크에 L2Normalization을 넣고자 한다면

class LayerNetWithL2:
    def __init__(self, input_size, hidden_size_list, output_size, activation = 'relu', weight_init_std = 'relu', weight_decay_lambda):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.output_size = output_size
        self.activation = activation
        self.weight_init_std = weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.parameter = {}

    def loss(self, x, y, train_val):
        predict = self.predict(x, train_val)

        weight_decay = 0
        for idx in range(1, len(self.hidden_size_list) + 2):
            W = self.parameter['W' + str(idx)]
            weight_decay = 0.5 * self.weight_decay_lambda * np.sum(W**2)
        #각각의 층에서 각각의 뉴런마다 가중치가 존재하기 때문에 각각의 가중치의 값이 커지는 것을 막기 위새허
        #모든 마지막 계층의 출력값에 대한 손실에 weight_decay, 즉 0.5 * weight**2 * lambda를 더해준다.
        return self.last_layer.forward(predict, y) + weight_decay

    def gradient(self, x,y):
        #위에서 정의한 네트워크에서 손실을 구하는 loss함수를 이용한다.
        loss_w = lambda w: self.loss(x, y, train_val=True)
        gradient = {}
        for idx in range(1, len(self.hidden_size_list) + 2):
            W = self.parameter['W' + str(idx)]
            #가중치의 기울기를 구하는 계산에서 오차역전파법에 따른 결과인 gradient(loss_w, parameter)의 값에 
            #정규화 항인 0.5 * lambda * weight**2를 미분한 lambda * weight를 더해준다.
            gradient['W' + str(idx)] = gradient(loss_w, self.parameter['W' + str(idx)]) + self.weight_decay_lambda * np.sum(W**2)


#OverFitting을 억제하는 방법으로 손실 함수에 가중치의 L2 Norm을 더한 가중치 감소 방법이 존재한다는 것을 알 수 있었다.
#그러나 신경망 모델이 점점 복잡해 지면 가중치 감소만으로는 overfitting을 감당하기 어려울 수 있다.
#따라서 이럴 때에 Dropout이라는 기법을 사용한다.

#Dropout은 임의로 뉴런을 삭제하면서 진행이 된다.
#따라서 하나의 계층에 존재하는 뉴런들을 무작위로 삭제함으로서 신호 전달이 없도록 하는 것이다.
#Train과정에서는 삭제할 뉴런을 무작위로 선택하고, Test 과정에서는 모든 뉴런에 신호를 전달한다.(눌론 Test의 과정에서는 각 뉴런의 출력에 훈련 때에 삭제 안한 비율을 곱하여 출력한다.)

class Dropout:
    def __init__(self, rate = 0.3):
        self.dropout_rate = rate
        self.mask = None
    
    def forward(self, x, train_valid = True):
        if train_valid:
            #self.mask는 self.dropout_rate를 넘는 값에 대해서는 True, 아니면 False의 boolean값으로 저장한다.
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)
    
    def backward(self, dout):
        return dout * self.mask

#dropout를 진행하게 되면 훈련데이터와 시험데이터에 대한 정확도의 차이가 감소한다.
#또한, 훈련 데이터에 대한 정확도 또한 100%에 도달하지 않는다.(훈련 데이터, 즉 tensorflow에서 validation data에 대한 validation accuracy만이 현저하게 높은 상태가 overfitting이 시험데이터에 된 상태이다.)        
#이렇게 dropout을 적용하게 되면 표현력을 높이면서도 overfitting의 억제가 충분히 가능하다.


#사실 tensorflow나 keras와 같은 framework를 사용할 때에는 Ensenble Learning을 이용한다.
#Ensenble Learning(앙상블 학습)은 결국에는 다양한 모델을 이용해서 학습을 시킨 뒤에(비슷한 구조의)
#따로따로 학습이 되면 test data를 예측할 때에는 그 출력들을 평균내는 기법이다.
#따라서, dropout는 ensenble learning과 같은 효과를 하나의 네트워크로 구현했다고 생각해도 무방하다.
