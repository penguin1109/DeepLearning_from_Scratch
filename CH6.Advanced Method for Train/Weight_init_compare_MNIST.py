from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from Optimization import SGD

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = load_mnist(normalize = True)
train_size = x_train.shape[0]
batch_size = 128
epoch = 1000

weight_init_types = {'std=0.01':0.01, 'Xavier':'sigmoid', 'He':'relu'}
optimizer = SGD()

networks = {}
loss = {}

for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size = 784, hidden_size_list = [100,100,100,100],
    output_size = 10, weight_init_std = weight_type)
    loss[key] = []

for i in range(epoch):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    for key in weight_init_types.keys():
        gradient = networks[key].gradient(x_batch, y_batch)
        optimizer.update(networks[key].params, gradient)

        train_loss = networks[key].loss(x_batch, y_batch)
        loss[key].append(train_loss)

    if i %100 == 0:
        for key in weight_init_types.keys():
            print(key + " : " + str(networks[key].loss(x_batch, y_batch)))

#아래와 같이 1000번 반복한 이후에 결과가 나왔고, 결론적으로 손실의 값이 제일 많이 주어들은 것은 
#초기 가중치의 설정을 He 가중치를 이용해서 했을 떄이다.
#std=0.01 : 2.2936275156260946
#Xavier : 0.46033906835967076
#He : 0.30963044313567756           

