import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
X = np.array(iris.data)
y = np.array(iris.target)
y = y.reshape((-1, 1))

X = X/np.amax(X, axis=0)
y = y/2

class Perceptron(object):
    def __init__(self, eta=0.001, input_size=4, hidden_size=8, output_size=1, epochs=10000):
        self.eta = eta
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.epochs = epochs
        self.cost = []

        self.b0 = np.random.randn(1, self.hidden_size)
        self.b1 = np.random.randn(1, self.output_size)
        self.W0 = np.random.randn(self.input_size, self.hidden_size)
        self.W1 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.first = np.dot(X, self.W0)+self.b0
        self.hidden = self.sigmoid(self.first)
        self.third = np.dot(self.hidden, self.W1)+self.b1
        self.output = self.sigmoid(self.third)
        return self.output

    def backward_propagation(self, X, y, output):
        self.output_error = self.eta*(y - output)
        self.output_delta = self.output_error*self.sigmoid_prime(output)

        self.hid_error = self.output_delta*self.sigmoid_prime(self.W1.T)
        self.hid_delta = self.hid_error*self.sigmoid_prime(self.hidden)

        self.W0 += X.T.dot(self.hid_delta)
        self.W1 += self.hidden.T.dot(self.output_delta)

        self.cost.append(self.eta*(y - output))


    def train(self, X, y):
        for i in range(self.epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, output)
        self.last = output


    def decide(self):
        for i in range(X.shape[0]):
            if self.last[i] <= 0.35:
                self.last[i] = 0
            elif self.last[i] > 0.7:
                self.last[i] = 1
            else:
                self.last[i] = 0.5

    def count_efficiency(self):
        sum = 0
        for i in range(X.shape[0]):
            if self.last[i] == y[i]:
                sum += 1
        return sum/X.shape[0]


Neural = Perceptron(eta=0.005,
                    input_size=4,
                    hidden_size=30,
                    output_size=1,
                    epochs=10000)
Neural.train(X, y)
Neural.decide()
print('\nAccuracy: %.2f%%' % (100 * Neural.count_efficiency()))


