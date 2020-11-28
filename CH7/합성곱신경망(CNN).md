### CH7. 합성곱 신경망(CNN)
- CNN은 Convolutional Neural Network의 약자로, 이미지 인식과 음성 인식 등 다양한 곳에 사용이 된다.

#### 7.1. 전체 구조
- Convolution Layer 과 Pooling Layer이라는 새로운 종류의 계층을 연결하는 방법으로 사용하게 된다.
- 그러나 다른 신경망 구조와의 차이라고 한다면 Affinr, 즉 완전 연결 계층을 사용하지 않는다는 점이다.
  - 원래의 신경망은 (Affine-ReLU)의 반복 이후에 Softmax 계층에서 최종 결과(확률)을 출력한다.
  - CNN은 (Convolution-ReLU-Pooling(생략 가능))의 반복 이후에 Softmax출력 계층을 이용하여 작동한다.

#### 7.2. 합성곱 계층
7.2.1. Affine의 문제점
- 데이터의 형상이 무시된다는 문제점이 존재한다. 
  - 이미지나 음성 등은 3차원이상의 데이터인데 이를 1차원 데이터로 바꿔 주어야만 완전 연결 계층에 입력이 가능하기 때문이다.

- 합성곱 계층의 입출력 데이터는 **특징맵(feature map)**이라고 한다.

7.2.2. 합성곱 연산
- 이미지 처리에서의 필터(간홍 filter 대신 kernell이라는 용어 사용) 연산에 해당한다.
- 데이터와 필터의 shape는 **(높이, 너비)**로 표현해 준다.

1. 합성곱 연산은 **filter의 window를 slide**하면서 진행이 된다.
2. 그리고 입력과 필터에 대응하는 원소끼리 곱한 뒤에 그 총합을 구해준다.(Fused Multiply Add)
3. 이 과정을 모든 장소에서 수행하면 합성곱 연산의 출력이 생긴다.
   - 따라서 예를 들면 input shape가 (4,4), filter shape가 (3,3)이면 output shape는 (2,2)가 된다.
  
- Affine 신경망에 존재하는 가중치 매개변수와 편향의 역할을 CNN에서는 필터의 매개변수가 대신한다.
  - 편향도 나중에 고정값을 더하면 되는데, 항상 (1,1) 크기의 하나의 값만을 모든 원소에 적용한다.

7.2.3. Padding
- 합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정 값(대부분 0)으로 채우는 것을 의미한다.
  - 따라서 원래 input shape가 (4,4)였다면 padding size를 2로 설정한다고 할 때 input shape는 (8,8), 3으로 설정하면 (10,10)이 된다.
- 이는 주로 출력의 크기를 조절할 용도로 사용한다.
  - 계속 filter을 이용해 합성곱 연산을 수행하는 과정에서 크기가 줄어 (1,1)이 되어 학습을 멈추게 되는 상황을 피하기 위해서이다.

7.2.4. Stride
- 필터를 적용하는 위치의 간격을 의미한다.
  - 위에서 지금까지 본 stride의 크기는 1이었지만, 이를 2로 설정하면 filter가 움직이는 칸수가 2로 늘어나게 된다.
input shape = (H, W)  filter size = (FH, FW)   output size = (OH, OW)   Padding = P    Stride = S  
1. OH = ((H + 2*P - FH) / S) + 1
2. OW = ((W + 2*P - FW) / S) + 1

7.2.5. 3차원 데이터의 합성곱 연산
- (세로, 가로, **channel**)이라는 세번째 데이터가 추가가 된다.
- 이때 channel은 색깔의 형태를 정해주는데, 1이면 흑백의 이미지이고 2이면 RGB의 이미지인 것이다.
  - channel 쪽으로 feature map이 여러 개 있다면 입력 데이터와 필터의 합성곱 연산을 채널마다 수행하고 그 결과를 더해서 하나의 출력을 얻는다.
  - 주의할 점은 **input shape의 channel 크기와 filter의 channel이 같아야**한다는 것이다.

7.2.6. 블록으로 생각하기
- 3차원의 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각하면 된다.
- 3차원 데이터를 다차원 배열로 나타낼 떄는 (channel, height, weight)의 순서로 작성한다.
- 여기서 출력 데이터는 한장의 특징 맵이 될 것이다. 즉, channel이 1개인 특징 맵인 것이다. 그렇다면 합성곱 연산의 출력으로 **다수의 channel**을 내보내고 싶다면 **filter을 여러개 사용**하면 된다.
- 그래서 입력 데이터의 shape가 (Channel, Height, Width)이고 filter shape가 (Number of Filters, Channel, Height, Width)가 된다.
  - 이때 편향의 형상은 (Number of Filters, 1,1)이어야 한다.

7.2.7. 배치 처리
- 일반 Affine계층과 마찬가지로 입력 데이터를 한 덩어리로 묶어서 배치로 처리한다.
- 이렇게 하기 위해 각 계층을 흐르는 데이터의 차원을 하나 늘려 4차원 데이터로 저장을 하게 된ㄴ다.
  - 즉, (데이터 수, 채널수, 높이, 너비)이런 식으로 저장하게 되는 것이다.

#### 7.3 Pooling 계층
- 세로, 가로 방향의 공간을 줄이는 연산
- 특정 크기의 영역을 잡아서 Max Pooling을 한다면 영역에서 제일 큰 수를 선택하고, Average Pooling을 한다면 영역의 평균을 구한다.
  - 일반적으로 pooling의 window의 크기와 stride를 같은 값으로 설정한다.

7.3.1. Pooling 계층의 특징
1. 학습해야 할 매개변수가 없다.
   - 따라서 역전파를 진행할 때에 학습할 가중치와 편향이 없다.
2. 채널의 수가 변하지 않는다.
3. 입력의 변화에 영향을 적게 받는다.

#### 7.4. Convolution/Pooling Layer 구현하기
1. Convolution Layer
```py3
class Convolution:
    def __init__(self, W, b, stride = 1, padding = 0):
        self.W = W   #가중치
        self.b = b   #편향
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        #Filter개수, Channel, Height, Width
        FN, C, FH, FW = self.W.shape
        N,C,H,W = x.shape

        out_h = int(1 + (H+2*self.padding - FH) / self.stride)
        out_w = int(1 + (W + 2*self.padding - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.padding)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W)

        #numpy의 transpose 함수를 이용해서 축의 순서를 변경한다.
        #data_format = channels first로 저장할 떄 와 같은 순서로 데이터의 형상을 바꾼 것이다.
        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)

        return out
```        
2. Pooling Layer
```py3
class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, padding = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        #N은 이미지 개수, C는 channel수, H는 높이, W는 너비
        N, C, H, W = x.shape
        #출력값의 높이와 너비의 크기를 계산
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #image data를 column data인 2차원으로 펼쳐주어서 4중 for문을 사용할 필요가 없게끔 한다.
        col = im2col(x, out_h, out_w, self.stride, self.padding)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis = 1)

        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out
```

#### 7.5 CNN 구현하기
```py3
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
```

#### Tensorflow & Keras Code
1. dilation_rate는 Dilated COnvolution을 수행하는 과정에서 사용이 되는데, 필터 내부에 zero padding을 추가해서 강제로 reception field를 늘리는 방법이다.
2. receptive field란 필터가 한 번의 보는 영영으로 볼 수 있는데, 결국 필터를 통해 어떤 사진의 전체적인 특징을 잡아내기 위해서는 receptive field는 높으면 높을 수록 좋다.
3. 그래서 일반적인 CNN에서는 이를 conv-pooling의 결합으로 해결한다. 
4. pooling을 통해 dimension을 줄이고 다시 작은 크기의 filter로 conv를 하면, 전체적인 특징을 잡아낼 수 있다. 
5. 하지만 pooling을 수행하면 기존 정보의 손실이 일어난다. 
6. 이를 해결하기 위한것이 Dilated Convolution으로 Pooling을 수행하지 않고도 receptive field의 크기를 크게 가져갈 수 있기 때문에 spatial dimension의 손실이 적고, 대부분의 weight가 0이기 때문에 연산의 효율도 좋다. 
7. 공간적 특징을 유지하는 특성 때문에 Dilated Convolution은 특히 Segmentation에 많이 사용된다.
8. 여기서 말하는 image segmentation이란 이미지에서 픽셀 단위로 해당 픽셀이 어떤 class에 속하는지를 정해주는 것이다.
9. image classifcation은 그렇다면 이미지에서 어떤 물체의 존재 여부를 판단하는 것이라고 할 수 있다.

```py3
import tensorflow as tf
import numpy as np

CNN = tf.keras.models.Sequential()
#filter과 kernel_size는 각각 차원의 개수, (세로, 가로) 길이를 의미한다.
#만약에 CNN을 dataset없이 그냥 정의하고자 한다면 input_shape라는 변수를 추가해야 한다.
#이 때 input_shape = (,28,28,2)라고 했다면 앞의 숫자는 batch_size, 즉 한번에 학습하는 데이터의 한 묶음(batch)의 개수이고 마지막에 2는 channel size이다.
#batch_size의 자리, 즉 맨 앞자리는 모델이 알아서 None,즉 어떤 수던 가능하다고 판단하기 떄문에 우리가 저기서 input_shape을 그대로 입력하면 5차원으로 인식해 error가 발생한다.
#우리는 일반적으로 data_format이라는 변수를 설정해 주지는 않지만 기존에 지정된 값은 channels_last로, channel의 개수가 shape의 맨 뒤에 오게 한다.

CNN.add(tf.keras.layers.Conv2D(filters = 2, kernel_size = 3,
    strides = (1,1), padding = 'valid', dilation_rate = (1,1),
    activation = 'relu',
    use_bias = True, kernel_initializer = 'glorot_uniform',
    bias_initializer = 'zeros', input_shape = input_shape))

#input_shape를 설정해 주지 않으면 나중에 compile과 fit을 하기 전에 CNN.summary()가 불가능하다.

CNN.add(tf.keras.layers.Conv2D(filters = 2, kernel_size = 3, 
    strides = (1,1), padding = 'valid', activation  ='relu'))
CNN.add(tf.keras.layers.MaxPooling2D(pool_size = (3,3),
     strides = (1,1), padding = 'valid')))
CNN.add(tf.keras.layers.Conv2D(filters = 2, kernel_size = 3,
    strides = (1,1), padding = 'False', activation = 'relu))
CNN.add(tf.keras.layers.Flatten())
#Dense layer에 넣어야 하기 떄문에 3차원 이미지를 Flatten()을 이용해서 줄여야 한다.
CNN.add(tf.keras.layers.Dense(10, activation = 'softmax'))

#만약에 마지막 출력층에 output의 개수가 1개였다면 activation = 'sigmoid'로 설정해야 하며, loss = 'binary_crossentropy'여야 한다.

CNN.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
CNN.fit(x_data, y_data, epochs = 10)
```

