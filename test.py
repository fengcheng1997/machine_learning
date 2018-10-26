import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.axis([-10,10,0,1])
plt.grid(True)
X=np.arange(-10,10,0.1)
y=1/(1+np.exp(-X))
plt.plot(X,y,'b-')
plt.title("Logistic function")
plt.show()