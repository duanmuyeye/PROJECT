import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.pyplot import MultipleLocator
# 加载数据
data = np.loadtxt("magic04.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
# 通过切片获取横坐标x1
x1 = data[:, 0]
x1_ = data[:, 1]
mean=np.mean(x1)
std= np.std(x1)
mean_=np.mean(x1_)
std_= np.std(x1_)
# print(mean)
# print(std)
# 正态分布的概率密度函数。 x 是对于均值和标准差的函数
x = np.linspace(mean - 3*std, mean + 3*std, 50)
y = np.exp(-(x - mean) ** 2 /(2* std **2))/(math.sqrt(2*math.pi)*std)
x_ = np.linspace(mean_ - 3*std_, mean_ + 3*std_, 50)
y_ = np.exp(-(x - mean_) ** 2 /(2* std_ **2))/(math.sqrt(2*math.pi)*std_)
fig = plt.figure()
# 将画图窗口分成1行2列，选择第一块区域作子图
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 设置标题
ax1.set_title('μ=%s'%(mean)+' σ=%s'%(std))
ax2.set_title('μ=%s'%(mean_)+' σ=%s'%(std_))
# 设置横坐标名称
ax1.set_xlabel('frequency')
ax2.set_xlabel('frequency')
# 设置纵坐标名称
ax1.set_ylabel('trait')
ax2.set_ylabel('trait')
# 属性是从1开始时
ax1.plot(x, y)
# 属性是从0开始
ax2.plot(x_, y_)
plt.grid(True)
plt.show()