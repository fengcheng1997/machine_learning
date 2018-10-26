# machine learning tips

## numpy 中的int和浮点
如果弄错了int和float，可能是错的

## np.meshgrid用来产生网格坐标 

## np.c_   np.r_   相加  列相加

## 矩阵乘法中要注意的问题
[1,2]的shape值(2,)，意思是一维数组，数组中有2个元素。

[[1],[2]]的shape值是(2,1)，意思是一个二维数组，每行有1个元素。

[[1,2]]的shape值是（1，2），意思是一个二维数组，每行有2个元素。
！！！！矩阵相乘一定要用下面两种形式，要不然就会出问题

所以在初始化的时候也要注意：
nn.b2 = np.zeros(nn.n_output_dim)
nn.b2 = np.zeros((1, nn.n_output_dim))
上面第一个是一维的，第二个是二维的