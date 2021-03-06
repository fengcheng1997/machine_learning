import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# generate sample data
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)


# generate nn output target
y_true = np.zeros((X.shape[0], 2))
y_true[np.where(y==0), 0] = 1
y_true[np.where(y==1), 1] = 1


n, m = X.shape
# plot data
plt.figure('original')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("original")
# plt.show()
# print(X, y)

def g(z):
	return 1.0/(1.0+np.exp(-z))

def softmax(z):
	pass	


class NN_model:
	learning_rate = 0.001
	n_epoch = 1000

nn = NN_model()

nn.input_dim = 2
nn.hidden_dim = 4
nn.output_dim = 2

nn.a0 = np.ones((1, nn.input_dim)) #输入层，第一项为bias,取成1
nn.w1 = np.random.randn(nn.hidden_dim, nn.input_dim) / np.sqrt(nn.input_dim) #!!!!!!!w1的初始化不能是相同的值，+1表示的是bias项的系数!!!!!!!!!!
nn.b1 = np.ones((1, nn.hidden_dim))
nn.diff1 = np.ones((1, nn.hidden_dim)) #每一层的误差
nn.a1 = np.ones((1, nn.hidden_dim))
nn.w2 = np.random.randn(nn.output_dim, nn.hidden_dim) / np.sqrt(nn.hidden_dim)
nn.b2 = np.ones((1, nn.output_dim))
nn.diff2 = np.ones((1, nn.output_dim))
nn.a2 = np.ones((1, nn.output_dim))
# print(nn.a2, nn.w2.T)

def forward_caculate(nn, X, y_pred):
	for i in range(m):
		nn.a0[0][i] = X[i]


	nn.a1 = g(np.dot(nn.a0, nn.w1.T))
	nn.a2 = g(np.dot(nn.a1, nn.w2.T))

	y_pred[np.where(nn.a2!=np.max(nn.a2))] = 0
	y_pred[np.where(nn.a2==np.max(nn.a2))] = 1

	return nn, y_pred


def back_propagation(nn, X, y_true):
	y_pred = np.zeros((1, 2))
	for m in range(nn.n_epoch):
		acc = 0.0
		L = 0.0
		for i in range(n):
			nn, y_pred = forward_caculate(nn, X[i], y_pred)
			

			for j in range(nn.output_dim):
				nn.diff2[0][j] = nn.a2[0][j] - y_true[i][j]

			nn.diff1 = nn.a1*(1 - nn.a1)*np.dot(nn.diff2, nn.w2)
			nn.w1 = nn.w1 - nn.learning_rate * np.dot(nn.diff1.T, nn.a0)
			nn.w2 = nn.w2 - nn.learning_rate * np.dot(nn.diff2.T, nn.a1)

			if y_true[i][0] == y_pred[0][0] and y_true[i][1] == y_pred[0][1]:
				acc += 1.0

			L += np.sum((y_true[i] - nn.a2)**2)			
		print('epoch %d: L = %f acc = %f' % (m+1, L, acc/n))

	return nn

	# print(nn.w1, nn.w2)

def drawPicture_of_my_prediction(nn, X):
	y_pred_label = []
	y_pred = np.zeros((1, 2))
	for i in range(n):
		nn, y_pred = forward_caculate(nn, X[i], y_pred)
		y_pred_label.append(int(y_pred.argmax(axis=1)))

	plt.figure('my_prediction')
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
	plt.title("predicted of my_prog")

	return y_pred_label


back_propagation(nn, X, y_true)
y_pred = np.zeros(y.shape)
y_pred = drawPicture_of_my_prediction(nn, X)



##################   sklearn    ####################
model = MLPClassifier().fit(X, y)
y_pred_MLP = model.predict(X)

print('accuracy of my_prog:', accuracy_score(y, y_pred))
print('accuracy of sklearn:', accuracy_score(y, y_pred_MLP))

# print(model.predict(X))
plt.figure('predicted of sklearn')
plt.scatter(X[:, 0], X[:, 1], c=y_pred_MLP, cmap=plt.cm.Spectral)
plt.title("predicted of sklearn")

plt.show()
