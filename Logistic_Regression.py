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
	error = np.array([label - sigmoid(np.dot(data, coef.T) + intercept)]) #转化成矩阵，要加[]
	# errormat = np.mat(error)
	# datamat = np.mat(data)
	# print(error.shape, errormat.shape)
	# print(data.shape, datamat.shape)
	# print(errormat.shape, datamat.shape)
	gradient = np.dot(error, data).sum(axis=0)
	
	coef = coef + learning_rate * gradient
	intercept = intercept + learning_rate * error.sum()

	# J = 0
	# J = -1/m * (label*np.log(sigmoid(np.dot(datamat, coef.T) + intercept)) + (1-label) * np.log(1 - sigmoid(np.dot(datamat, coef.T) + intercept))).sum()

predict_of_my_prog = []
for i in range(m):
	if sigmoid(np.dot(data[i, :], coef.T) + intercept) > 0.5:
		predict_of_my_prog.append(1)
	else:
		predict_of_my_prog.append(0)
print('parameters of my_prog:', coef,intercept)
print('accuracy_score_of_my_prog:', accuracy_score(predict_of_my_prog, label))
# print(predict_of_my_prog) 
# print(label)


########  sklearn methon     #################

clf = LogisticRegression(penalty='l1', C=1.0, random_state=0)
clf.fit(data, label)
y_pred = clf.predict(data)
# y_test = y_test.reshape(1, -1)
# print('正确率:', clf.score(data,label))

print('parameters of sklearn:', clf.coef_, clf.intercept_)



print('accuracy_score_of_sklearn:', accuracy_score(y_pred, label)) #真值和预测值的0和1的lable刚好相反，所以精度为0，实际上是100%

# print(label, y_pred, predict_of_my_prog)


