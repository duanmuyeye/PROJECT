import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
# 加载数据
data = np.loadtxt("magic04.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
X= []
for i in range(10):
    print("第", i+1 ,"个属性的方差为：", np.var(data[:, i]))
    X.append(np.var(data[:, i]))
print("最大方差为：", max(X))
print("最小方差为：", min(X))