from dataset.mnist import load_mnist
import numpy as np
import pickle
import os
import sys

def get_data():
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    return x_test, y_test

def init_network():
    with open("H:\\DeepLearningFromScratch\\CH3\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def softmax(x):
    c = np.max(x)
    expect = np.exp(x-c)
    return expect / np.sum(expect)

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    A1 = np.dot(x, w1) + b1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, w2) + b2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, w3) + b3
    y = softmax(A3)
    return y

x,label = get_data()
network = init_network()

correct = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == label[i]:correct += 1

print('Accuracy:' + str(float(correct)/len(x)))

#이렇게 해서 데이터를 입력해 넣은 이후에 Accuracy를 출력해 보면 0.9352, 즉 93.52%가까이 나온다.

#여기에 batch처리를 해서 한번에 batch_size만큼 묶어서 연산을 해주어 보자.
batch_size = 100
correct = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y = predict(network, x_batch)
    #y의 크기가 (100,10)이기 때문에 1번째 차원을 축으로 최댓값의 인덱스를 찾도록 한다.
    p = np.argmax(y, axis = 1)
    #label의 원소들과 위에서 구한 각 행의 최댓값의 일칭 여부를 boolean값으로 반환하고
    #np.sum()을 이용해서 해당 배열에서의 True의 개수를 세어서 반환한다.
    correct += np.sum(p == label[i:i+batch_size])

print('Batch_Accuracy :' + str(float(correct)/len(x)))

#이렇게 데이터를 batch처리 함으로서 효율적이고 빠르게 처리가 가능하다.
#신경망은 각 층의 뉴런이 다음 층의 뉴런으로 신호를 전달한다는 점에서 퍼셉트론과 동일하지만
#신경망에서는 매끄럽게 변화하는 sigmoid함수를 활성화 함수로, 퍼셉트론에서는 갑자기 변화하는 계단함수를 사용한다는 차이를 보인다.
#이는 역전파 과정에서, 즉 손실을 학습하는 과정에서 유의미한 차이를 보인다.