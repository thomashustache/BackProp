
import numpy as np
import matplotlib.pyplot as plt
import copy

# define the sigmoid function
def sigmoid(x, derivative=False):

    if (derivative == True):
        return sigmoid(x,derivative=False) * (1 - sigmoid(x,derivative=False))
    else:
        return 1 / (1 + np.exp(-x))
    
# define the relu function
def relu(x, derivative=False):

    if (derivative == True):
        o = np.zeros_like(x)
        o[x > 0] = 1
        return o
    else:
        return np.maximum(0, x)

# choose a random seed for reproducible results
np.random.seed(1)

# learning rate
alpha = .003

# number of nodes in the hidden layer
num_hidden = 20

# Construct Train and Test sest of the Sinus function
X = np.linspace(-np.pi, np.pi, num=5000)
X_test = np.linspace(-np.pi / 2, np.pi/2, num=5000)
xt = copy.deepcopy(X_test)
X = np.expand_dims(X, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# outputs
# x.T is the transpose of x, making this a column vector
y = np.sin(X)
y_test = np.sin(X_test)

# initialize weights randomly with mean 0 and range [-1, 1]
# the +1 in the 1st dimension of the weight matrices is for the bias weight (Bias trick)
hidden_weights = 2 * np.random.random((X.shape[1] + 1, num_hidden)) - 1
output_weights = 2 * np.random.random((num_hidden + 1, y.shape[1])) - 1

# number of iterations of gradient descent
num_epochs = 5000
loss_per_epoch = []
loss_test = []

# for each iteration of simple gradient descent step on whole dataset
for i in range(num_epochs):

    # forward phase
    # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
    input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
    hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), relu(np.dot(input_layer_outputs, hidden_weights))))
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)
    
    
    mse = abs(output_layer_outputs - y) ** 2
    loss_per_epoch += [np.mean(mse)]

    if i % 100 == 0:
        print('MSE: {:.5f}'.format(np.mean(mse)))

    # backward phase
    # output layer error term
    output_error = output_layer_outputs - y
    # hidden layer error term
    # [:, 1:] removes the bias term from the backpropagation
    hidden_error = relu(x=hidden_layer_outputs[:, 1:], derivative=True) * np.dot(output_error, output_weights.T[:, 1:])

    # partial derivatives of hidden and output layers
    hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
    output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

    # average for total gradients
    total_hidden_gradient = np.average(hidden_pd, axis=0)
    total_output_gradient = np.average(output_pd, axis=0)

    # update weights
    hidden_weights += - alpha * total_hidden_gradient
    output_weights += - alpha * total_output_gradient
    
    # evaluation :
    input_layer_outputs = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    hidden_layer_outputs = np.hstack((np.ones((X_test.shape[0], 1)), relu(np.dot(input_layer_outputs, hidden_weights))))
    output_layer_outputs = np.dot(hidden_layer_outputs, output_weights)
    
    sinus_test = output_layer_outputs
    mse_test = abs(output_layer_outputs - y_test) ** 2
    loss_test += [np.mean(mse_test)]


plt.figure()
plt.plot(np.arange(len(loss_per_epoch)), loss_per_epoch, label = 'train')
plt.plot(np.arange(len(loss_test)), loss_test, label = 'test')
plt.grid(True)
plt.legend()
plt.title('Train and test MSE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('losses.png')

plt.figure()
plt.plot(xt, y_test, label = 'Real sinus')
plt.plot(xt, sinus_test, label = 'Estimated')
plt.grid(True)
plt.legend()
plt.title('Estimation of Sinus')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig('estimations.png')
plt.show()