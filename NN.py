import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# generate sample data
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)


# generate nn output target
t = np.zeros((X.shape[0], 2))
t[np.where(y==0), 0] = 1
t[np.where(y==1), 1] = 1

n, m = X.shape

# plot data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.show()
# print(X, y)

def g(z):
	return 1.0/(1.0+np.exp(-z))

def softmax(z):
	pass	


class NN_model:
	learning_rate = 0.01
	n_epoch = 1000

nn = NN_model()

nn.input_dim = 2
nn.hidden_dim = 4
nn.output_dim = 2

nn.a0 = np.ones(nn.input_dim+1) #输入层，第一项为bias,取成1
nn.w1 = np.random.randn(nn.hidden_dim, nn.input_dim+1) / np.sqrt(nn.input_dim) #!!!!!!!w1的初始化不能是相同的值，+1表示的是bias项的系数!!!!!!!!!!
nn.diff1 = np.ones(nn.hidden_dim) #每一层的误差
nn.a1 = np.ones(nn.hidden_dim+1)
nn.w2 = np.random.randn(nn.output_dim, nn.hidden_dim+1) / np.sqrt(nn.hidden_dim)
nn.diff2 = np.ones(nn.output_dim)
nn.a2 = np.ones(nn.output_dim)


def forward_caculate(nn, X):
	for i in range(m):
		nn.a0[i+1] = X[i]

	nn.a1[1:] = g(np.dot(nn.w1, nn.a0.T))
	nn.a2 = g(np.dot(nn.w2, nn.a1.T))

	nn.a2[np.where(nn.a2!=np.max(nn.a2))] = 0
	nn.a2[np.where(nn.a2==np.max(nn.a2))] = 1

	return nn


def back_propagation(nn, X, t):
	for i in range(n):
		forward_caculate(nn, X[i])

		for j in range(nn.output_dim):
			nn.diff2[j] = nn.a2[j] - t[i][j]

		nn.diff1 = nn.a1*(1 - nn.a1)*np.dot(nn.w2.T, nn.diff2)
		nn.w1 = nn.w1 - nn.learning_rate * np.dot(nn.a1, nn.diff1.T)
		nn.w2 = nn.w2 - nn.learning_rate * np.dot(nn.a2, nn.diff2.T)

	return nn

def caculate_accuracy(nn, X, t):
	acc = 0.0
	for i in range(n):
		forward_caculate(nn, X[i])

		if nn.a2[0] == t[i][0] and nn.a2[1] == t[i][1]:
			acc += 1.0

	return acc/n


for i in range(nn.n_epoch):
	print(nn.w1, nn.w2)
	back_propagation(nn, X, t)
	print(caculate_accuracy(nn, X, t))

# a = [1, 1]
# b = [1, 1]
# if a == b:
# 	print(1)
# else:
# 	print(2)

# a = np.arange(9)
# print(a, np.max(a))
# # for i in arange(9):
# a[np.where(a==3)] = 7387383783
# print(a)