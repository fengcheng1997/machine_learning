# machine learning tips

## numpy 

import numpy as np
np.array([1,1]) and np.array([[1,1]])
 x = x.reshape((-1,)) # 拉平

from numpy import *
v = array([1,2,3,4])	
type()	
v.shape
v.dtype
v.ndim
x = arange(0, 10, 1)  #1-10
linspace(0, 10, 25)		#[0,10]  25
x, y = mgrid[0:5, 0:5]	#网格	

random.rand(5,5)
random.randn(5,5)
zeros((3,3))
ones((3,3))

M = array([[1, 2], [3, 4]])
M[1,:] = 0
M[:,2] = -1

v*v, v+2
dot(v, v.T)
Other tips is on the Mr.Bu's notebook

### numpy 中的int和浮点
如果弄错了int和float，可能是错的

### np.meshgrid用来产生网格坐标 

### np.c_   np.r_   相加  列相加

### 矩阵乘法中要注意的问题
[1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。

[[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。

[[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
！！！！矩阵相乘一定要用下面两种形式，要不然就会出问题

所以在初始化的时候也要注意：
nn.b2 = np.zeros(nn.n_output_dim)
nn.b2 = np.zeros((1, nn.n_output_dim))
上面第一个是一维的，第二个是二维的



## pytorch

import torch as t

函数						功能
Tensor(*sizes)			基础构造函数
ones(*sizes)			全1Tensor
zeros(*sizes)			全0Tensor
eye(*sizes)				对角线为1，其他为0
arange(s,e,step			从s到e，步长为step
linspace(s,e,steps)		从s到e，均匀切分成steps份
rand/randn(*sizes)		均匀/标准分布
normal(mean,std)/uniform(from,to)	正态分布/均匀分布
randperm(m)				随机排列

t.ones_like(n)

b.tolist()
c.shape
a = t.arange(0, 6)
a.view(2, 3)	#通过tensor.view方法可以调整tensor的形状
a[:2, 0:2]
a > 1
a[a>1]


t.Tensor(numpy)
x.squeeze().numpy()
x.squeeze(1)
x.unsqueeze(1)
.view_as(x_mean)

x.mm(w)
torch.matmul(x, w))


from torch.autograd import Variable
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
x_train = torch.from_numpy(x_train)
x_train = Variable(x_train)	#numpy to tensor to variable

x_train.data.numpy()	#variable to numpy

x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)	
k.backward(torch.FloatTensor([1, 0]), retain_graph=True)

∂n∂m0=w0∂n0∂m0+w1∂n1∂m0=2m0+0=2×2=4
∂n∂m1=w0∂n0∂m1+w1∂n1∂m1=0+3m21=3×32=27
read the notebook of Mr.Bu(autograd)
It's different when caculating the one numbers and the list of numpy

# how to use variable
y_ = linear_model(x_train)
loss = get_loss(y_, y_train)

w.grad.zero_() # 记得归零梯度
b.grad.zero_() # 记得归零梯度
loss.backward()

w.data = w.data - 1e-2 * w.grad.data # 更新 w
b.data = b.data - 1e-2 * b.grad.data # 更新 b 


# label of classification
(a > 0.5) * 1


## DeepNN
```
net = nn.Sequential(
    nn.Linear(64, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
```
out = net(n, m) #输入数据类型 (number of samples, number of neuron)
只有一个m， 图片需要展开

## convD2 
```
net = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    nn.ReLU(),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    nn.ReLU(),
  	nn.MaxPool2d(2, 2)
)
```
out = net(n, in_channels, x, y) #(numbers of samples, in_channesls, x of img, y of img)
需要输入图片的通道数和长宽

## rnn
RNN 的输入形式，(length of sequence, batch, numbers of feature)
