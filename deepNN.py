import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# load data
digits = load_digits()

# # plot the digits
# fig = plt.figure('show', figsize=(6, 6))  # figure size in inches
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# # plot the digits: each image is 8x8 pixels
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary)
    
#     # label the image with the target value
#     ax.text(0, 7, str(digits.target[i]))

# # plt.show()

# print(type(digits))


train_set = digits.images[0:1400,:] #ndarray
train_set = ((train_set / 16) - 0.5) / 0.5	#x_train.max(): 16 	
train_set = np.array([i.reshape((-1,)) for i in train_set])
label_set = np.array(digits.target[0:1400]).reshape(1400,1)
train_set = np.concatenate([train_set, label_set], axis=1)
train_set = torch.FloatTensor(train_set)

test_set = digits.images[1400:,:]
test_set = ((test_set / 16) - 0.5) / 0.5
test_set = np.array([i.reshape((-1,)) for i in test_set])
label_set_test = np.array(digits.target[1400:]).reshape(1797-1400,1)
test_set = np.concatenate([test_set, label_set_test], axis=1)
test_set = torch.FloatTensor(test_set)


# 使用 pytorch 自带的 DataLoader 定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=50, shuffle=True, drop_last=True)
test_data = DataLoader(test_set, batch_size=50, shuffle=True, drop_last=True)

# 使用Sequential定义4层神经网络
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

# 定义 loss 函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(10):
    train_loss = 0
    train_acc = 0
    # net.train()

    for data in train_data:
        im = Variable(data[:,0:-1])
        label = Variable(data[:,-1].long())
        # print(im, label)

        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data[0]
        # 计算分类的准确率
        _, pred = out.max(1)
        print(pred, out.max(1))
        print(label)
        num_correct = float((pred == label).sum().data[0])
        acc = num_correct / im.shape[0]
        train_acc += acc
        
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # net.eval() # 将模型改为预测模式
    for data in test_data:
        im = Variable(data[:,0:-1])
        label = Variable(data[:,-1].long())
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.data[0]
        # 记录准确率
        _, pred = out.max(1)
        num_correct = float((pred == label).sum().data[0])
        acc = num_correct / im.shape[0]
        eval_acc += acc
        
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    if((e+1) % 50 == 0):
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(e+1, train_loss / len(train_data), train_acc / len(train_data), 
                         eval_loss / len(test_data), eval_acc / len(test_data)))


# 画出loss和acc曲线
plt.figure('train loss')
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)

plt.figure('train acc')
plt.title('train acc')
plt.plot(np.arange(len(acces)), acces)


# plt.show()

