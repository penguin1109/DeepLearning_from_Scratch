import numpy as np

#실제로 나중에 모델을 정의할 때에도 class NeuralNetwork(keras.models) 로 정의하게 된다면
#def __init__()부분을 이용해서 가중치와 편향을 초기화 하게 될 것이다.

def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['w3'] = np.array([[0.1,0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def sigmoid(x):
    return 1/ (1+(np.exp(-x)))

#forward함수인 이유는 처음에 신호가 순방향으로 전달되는 것은 순전파라 하고
#학습 과정에서 손실을 이용하여 trainable parameter의 값을 수정하는 것은 backward과정인 역전파라고 한다.
def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    A1 = np.dot(x, w1) + b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, w2) + b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, w3) + b3
    return A3

network = init_network()
x = np.array([1.0, 0.5])
output_1 = forward(network, x)

#위와 같이 numpy의 다차원 배열을 이용하면 신경망을 효율적으로 구현할 수 있게 된다.

#위에서는 network의 구성에서 마지막 출력 데이터를 그냥 return했다면 이번에는 softmax함수를 구현해서 이를 output에 이용하여 network를 구성할 것이다.
def softmax(x):
    expect = np.exp(x)
    sum = np.sum(expect)
    y = expect/sum
    return y

#그러나 위와 같은 방법으로 코드를 작성하게 되면 지수의 연산이기 때문에 큰값/큰값이 되어 overflow가 발생할 수 있다.
#따라서 분자 분모에 각각 c라는 임의의 정수를 더해주는데, 이 값은 일반적으로 A의 행렬값들중 최댓값으로 설정한다.

def aug_softmax(x):
    c = np.max(x)
    expect = np.exp(x-c)
    y = expect/np.sum(expect)
    return y

#softmax함수의 출력값을 0과 1사이이고, 모든 원소들의 합은 1이다. 그렇기 때문에 확률로 생각하면 된다.
#그러나 문제는 이 함수가 단조 증가함수이기 때문에 결국 원래 입력해주는 원소들의 대소와 일치한다.
#따라서 결국에는 현업에서는 출력층의 sotfmax함수는 생략하는 경우도 많다.

