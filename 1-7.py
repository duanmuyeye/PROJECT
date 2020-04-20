import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
# 加载数据
data = np.loadtxt("magic04.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
X= []
for i in range(10):
    for g in range(10):
        if i!= g:
            print(i+1,"和",g+1,"的协方差为：",np.mean((data[:, i] - data[:, i].mean()) * (data[:, g] - data[:, g].mean())))
            X.append([(np.mean((data[:, i] - data[:, i].mean()) * (data[:, g] - data[:, g].mean()))), i+1, g+1])

print("最大协方差为", max(X),"最小协方差为：",min(X))