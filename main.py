import random
import numpy as np
from scipy.special import expit as sigmoid
import gzip

class Network(object):
	def __init__(self, s):
		self.layers = len(s)
		self.sizes = s
		self.biases = [np.random.randn(y, 1) for y in s[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(s[:-1], s[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return a

	def train(self, training_data, iters, mini_batch_size, eta, test_data):
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in xrange(iters):
			random.shuffle(training_data)
			mini_batches = [
				training_data[k:k+mini_batch_size]
				for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.batch_update(mini_batch, eta)
			print "Iteration {0}: {1} / {2}".format(
				j, self.evaluate(test_data), n_test)

	def batch_update(self, mini_batch, eta):
		der_b = [np.zeros(b.shape) for b in self.biases]
		der_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_der_b, delta_der_w = self.backprop(x, y)
			der_b = [nb+dnb for nb, dnb in zip(der_b, delta_der_b)]
			der_w = [nw+dnw for nw, dnw in zip(der_w, delta_der_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, der_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
					   for b, nb in zip(self.biases, der_b)]

	def backprop(self, x, y):
		der_b = [np.zeros(b.shape) for b in self.biases]
		der_w = [np.zeros(w.shape) for w in self.weights]
		# forward direction
		activation = x
		activations = [x]
		outputs = []
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			outputs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		# find derivatives and update params
		delta = self.cost_derivative(activations[-1], y) * \
			der_sigmoid(outputs[-1])
		der_b[-1] = delta
		der_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.layers):
			z = outputs[-l]
			sp = der_sigmoid(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			der_b[-l] = delta
			der_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (der_b, der_w)

	def evaluate(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y)
						for (x, y) in test_data]
		return sum([int(x == y) for (x, y) in test_results])

	def cost_derivative(self, output_activations, y):
		return (output_activations-y)


def der_sigmoid(z):
	return sigmoid(z)*(1-sigmoid(z))

def vectorized_result(j, i=10):
	e = np.zeros((i,1))
	e[j] = 1.0
	return e

def one_hot(x,m):
	ox = np.zeros((m))
	if x < m:
		ox[int(x)]=1
	return ox

def one_hot_2(nums):
	a = map(lambda l : one_hot(l-1,m=4), nums)
	b = map(lambda l : one_hot(l-1,m=13), nums)
	c = a[0].tolist()+b[1].tolist()+a[2].tolist()+b[3].tolist()+a[4].tolist()+b[5].tolist()+a[6].tolist()+b[7].tolist()+a[8].tolist()+b[9].tolist()
	return c

train_data = np.array(np.genfromtxt('train_trunc.csv', delimiter=',', skip_header=1), dtype='int32')

training_inputs = np.array([np.reshape(one_hot_2(x), (85, 1)) for x in train_data[:,:-1]])
training_results = [vectorized_result(y) for y in train_data[:,-1]]
training = zip(training_inputs, training_results)

testing = zip(training_inputs, train_data[:,-1])

nn = Network([85, 100, 10])
nn.train(training_data=training, iters=10, mini_batch_size=50, eta=1.0, test_data=testing)

valid_data = np.array(np.genfromtxt('test.csv', delimiter=',', skip_header=1), dtype='int32')
valid_data = np.array([np.reshape(one_hot_2(x), (85, 1)) for x in valid_data[:,1:]])
test_results = [np.argmax(nn.feedforward(x)) for x in valid_data]