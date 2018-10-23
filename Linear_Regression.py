#Linear_Regression

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression


# load data
d = datasets.load_diabetes()

X = d.data[:, 2]
# X = (X - np.mean(X))/(X.max() - X.min())
Y = d.target
# Y = (Y - np.mean(Y))/(Y.max() - Y.min())

m = X.shape[0]
# print(X, Y)

# a = np.array([[1,2,3],[2,3,4]])
# print(a.sum())

# print(a.transpose())
# # draw original data
# plt.scatter(X, Y)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()


################   my program    #################


S_X = X.sum()
S_Y = Y.sum()
S_XY = np.sum(X*Y)
S_XX = np.sum(X*X)

learning_rate = 0.5	#when learning rate is diffent, it may not be able to convergent or be too slow to convergent
n = 3000
a, b = 0, 0

for i in range(n):
	a = a - learning_rate * 1/m*(a * S_XX + b * S_X - S_XY)
	b = b - learning_rate * 1/m*(a * S_X + b * m - S_Y)

	J = 0
	for j in range(m):
		J = J + 0.5/m * (a * X[j] + b - Y[j])**2

	print('step %d:  loss = %.6f  a = %.3f  b = %.3f' % (i+1, J, a ,b))


plt.figure('Compare')
plt.plot(X, a*X + b, label='my_prog')


########  sklearn methon     #################

reg = LinearRegression().fit(X.reshape(-1, 1), Y)
a_ = reg.coef_
b_ = reg.intercept_

plt.plot(X, a_*X + b_, label='skelearn method')
plt.legend()

plt.show()