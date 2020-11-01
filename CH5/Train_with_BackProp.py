from numpy.lib.function_base import gradient
from TwoLayerNet_BackProp import TwoLayerNet
from dataset.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, one_hot_label = True)
net = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)
epoch = 10000
#x_train.shape = (60000, 784)
#t_train.shape = (60000, 10)

train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size/batch_size,1)

train_loss, train_accuracy, test_accuracy = [],[],[]

for i in range(epoch):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.backprop(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = net.accuracy(x_train, t_train)
        test_acc = net.accuracy(x_test, t_test)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        print('Epoch = {}  Train Accuracy = {}   Test Accuarcy = {}'.format(i, train_acc, test_acc))
    

    

    
