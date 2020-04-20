import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
data = np.loadtxt("magic04.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
# 通过切片获取x1
x1 = data[:, 0]
# 通过切片获取x2...
x2 = data[:, 1]
x1_ = data[:, 1]
# 通过切片获取x2...
x2_ = data[:, 2]
d1 = np.mean((x1 - x1.mean()) * (x2 - x2.mean()))/np.std(x1)/np.std(x2)
d2 = np.mean((x1_ - x1_.mean()) * (x2_ - x2_.mean()))/np.std(x1_)/np.std(x2_)
print("余弦相似度为", d1, "属性从0算起", d2)
arr_mean = np.mean(data[:, 0])
arr_mean2 = np.mean(data[:, 1])
arr_mean_ = np.mean(data[:, 1])
arr_mean2_ = np.mean(data[:, 2])
# 创建画图窗口
fig = plt.figure()
# 将画图窗口分成1行2列，选择第一块区域作子图
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 设置标题
ax1.set_title('magic04-start from 1')
ax2.set_title('magic04-start from 0')
# 设置横坐标名称
ax1.set_xlabel('x1')
ax2.set_xlabel('x1')
# 设置纵坐标名称
ax1.set_ylabel('x2')
ax2.set_ylabel('x2')
# 画散点图
ax1.scatter(x1, x2, s=20,  marker=".")
ax1.scatter(arr_mean , arr_mean2 , s=20, marker=",", c='r')
ax2.scatter(x1_, x2_, s=20,  marker=".")
ax2.scatter(arr_mean_ , arr_mean2_ , s=20, marker=",", c='r')
x_major_locator=MultipleLocator(5)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(5)
# 调整横坐标的上下界0
# plt.xlim(xmax=8, xmin=4)
# plt.ylim(ymax=5, ymin=1)
# 显示
plt.show()
# 如果属性编号从0开始 这里不确定属性编号是从1还是从0开始
