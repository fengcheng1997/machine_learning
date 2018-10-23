#Linear_Regression

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# load data
data, label = sklearn.datasets.make_moons(400, noise=0.30)

m = data.shape[0]


# draw original data
# plt.figure('Original')

# plt.scatter(data[:,0], data[:,1], c=label)
# plt.xlabel("X")
# plt.ylabel("Y")


################   my program    #################

# def sigmoid(z):
# 	return 1.0/(1.0+np.float128(np.exp(-z)))

def sigmoid(inX): 
	#return 1.0/(1+exp(-inX)) 
	#优化 
	if inX.any()>=0: 
		return 1.0/(1.0+np.exp(-inX)) 
	else: 
		return np.exp(inX)/(1+np.exp(inX))


# S_X = data[:,0].sum()
S_Y = data[:,1].sum()
S_XY = np.sum(data[:,0]*data[:,1])
# S_XX = np.sum(data[:,0]*data[:,1])

learning_rate = 0.01	
n = 5000
coef = np.array([0., 0.])
intercept = 0



for i in range(n):
	error = label - sigmoid(np.dot(data, coef.T) + intercept)
	errormat = np.mat(error)
	datamat = np.mat(data)
	# print(errormat.shape, datamat.shape)
	gradient = np.dot(errormat, datamat).sum(axis=0)
	
	coef = coef + learning_rate * gradient
	intercept = intercept + learning_rate * error.sum()

	# J = 0
	# J = -1/m * (label*np.log(sigmoid(np.dot(datamat, coef.T) + intercept)) + (1-label) * np.log(1 - sigmoid(np.dot(datamat, coef.T) + intercept))).sum()

	# print('step %d:  theta = %.6f, %.6f  b = %.6f' % (i+1, float(coef[0,0]), float(coef[0,1]) ,intercept))

predict_of_my_prog = []
for i in range(m):
	if sigmoid(np.dot(data[i, :], coef.T) + intercept) > 0.5:
		predict_of_my_prog.append(1)
	else:
		predict_of_my_prog.append(0)


print('accuracy_score_of_my_prog:', accuracy_score(predict_of_my_prog, label))
# print(predict_of_my_prog) 
# print(label)


########  sklearn methon     #################

clf = LogisticRegression(penalty='l1', C=1.0, random_state=0)
clf.fit(data, label)
y_pred = clf.predict(data)
# y_test = y_test.reshape(1, -1)
# print('正确率:', clf.score(data,label))

print(clf.coef_, clf.intercept_)



print('accuracy_score_of_sklearn:', accuracy_score(y_pred, label)) #真值和预测值的0和1的lable刚好相反，所以精度为0，实际上是100%

# print(label, y_pred, predict_of_my_prog)

# #################   Mr.bu     #######################
# def plot_decision_boundary(predict_func, data, label):
#     """画出结果图
#     Args:
#         pred_func (callable): 预测函数
#         data (numpy.ndarray): 训练数据集合
#         label (numpy.ndarray): 训练数据标签
#     """
#     x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
#     y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
#     h = 0.01

#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#     Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)
#     plt.show()

# def sigmoid(x):
#     return 1.0 / (1 + np.exp(-x))

# class Logistic(object):
#     """logistic回归模型"""
#     def __init__(self, data, label):
#         self.data = data
#         self.label = label

#         self.data_num, n = np.shape(data)
#         self.weights = np.ones(n)
#         self.b = 1

#     def train(self, num_iteration=150):
#         """随机梯度上升算法
#         Args:
#             data (numpy.ndarray): 训练数据集
#             labels (numpy.ndarray): 训练标签
#             num_iteration (int): 迭代次数
#         """
#         for j in range(num_iteration):
#             data_index = list(range(self.data_num))
#             for i in range(self.data_num):
#                 # 学习速率
#                 alpha = 0.01
#                 rand_index = int(np.random.uniform(0, len(data_index)))
#                 error = self.label[rand_index] - sigmoid(sum(self.data[rand_index] * self.weights + self.b))
#                 self.weights += alpha * error * self.data[rand_index]
#                 self.b += alpha * error
#                 del(data_index[rand_index])

#     def predict(self, predict_data):
#         """预测函数"""
#         result = list(map(lambda x: 1 if sum(self.weights * x + self.b) > 0 else 0,
#                      predict_data))
#         return np.array(result)
# logistic = Logistic(data, label)
# logistic.train(200)
# plot_decision_boundary(lambda x: logistic.predict(x), data, label)

