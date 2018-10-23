# KMeans

import matplotlib.pyplot as plt
import numpy as np

X0 = np.array([7, 5, 7, 3, 4, 1, 0, 2, 8, 6, 5, 3])
X1 = np.array([5, 7, 7, 3, 6, 4, 0, 2, 7, 8, 5, 7])

n = X0.shape[0]


##########   my prog  ############

# step 1:initiation
c = np.array([[X0[0], X1[0]], [X0[1], X1[1]]])
c_ = np.array([[0, 0], [0, 0]])
# print(n, c.shape)

# step 2:caculate the minmum length
label = np.zeros(n)
sum0, sum1 = [0, 0, 0], [0, 0, 0]

while np.fabs((c - c_).sum()) > 0.0001:
	print((c - c_).sum())
	for i in range(n):
		l0 = (X0[i]-c[0][0])**2 + (X1[i]-c[0][1])**2
		l1 = (X0[i]-c[1][0])**2 + (X1[i]-c[1][1])**2

		if l0 < l1:
			label[i] = 0
			sum0[0] += X0[i]
			sum0[1] += X1[i]
			sum0[2] += 1
		else:
			label[i] = 1
			sum1[0] += X0[i]
			sum1[1] += X1[i]
			sum1[2] += 1
	c_ = c
	c = np.array([[sum0[0]/sum0[2], sum0[1]/sum0[2]], [sum1[1]/sum1[2], sum1[1]/sum1[2]]])


print(label) 
print(c) 

plt.figure()
plt.axis([-1, 9, -1, 9])
plt.grid(True)
plt.plot(c[0, 0],c[0, 1],'rx',ms=12.0)
plt.plot(c[1, 0],c[1, 1],'g.',ms=12.0);

for i in range(n):
	if label[i] == 0:
		plt.plot(X0[i], X1[i], 'rx')
	else:
		plt.plot(X0[i], X1[i], 'g.')

plt.show()
