import numpy as np
from TwolayerNet import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(normalize = True, one_hot_label = True)

train_loss = []

epoch, train_size, batch_size, learning_rate = 7, x_train.shape[0], 100, 0.001
network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
train_acc_list, test_acc_list = [],[]

for iter in range(epoch):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    gradient = network.numerical_gradient(x_batch, y_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * gradient[key]
    
    loss = network.loss(x_batch, y_batch)
    train_loss.append(loss)

    train_acc = network.accuracy(x_train, y_train)
    test_acc = network.accuracy(x_test, y_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print('Loss : {} Train Accuracy : {} Test Accuracy : {}'.format(loss, train_acc, test_acc))


