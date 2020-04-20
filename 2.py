import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linalg as la
from matplotlib.pyplot import MultipleLocator
# 加载数据
# 测试
np.set_printoptions(suppress=True)
# data =np.array([[5.9,3],[6.9,3.1],[6.5,2.9],[4.6,3.2],[6,2.2]])
data = np.loadtxt("iris.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3))
# 初始化一个零矩阵
K= np.zeros((data.shape[0],data.shape[0]))
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        # 齐次二阶核
        K[i][j]=math.pow((np.dot(data[i].T,data[j])), 2)
print("齐次二次核：","\n",K)
print("========================================================================")
# 单位矩阵
I = np.identity(data.shape[0])
# print(I)
# 生成奇异矩阵 全1
M =np.ones((data.shape[0], data.shape[0]))
# print(M)
#
N=I - M*(1/data.shape[0])
# print(N)
# 计算居中化核矩阵
A=np.dot(np.dot(N,K),N)
print("居中化核矩阵：","\n",A)
print("========================================================================")
# 取对角矩阵  其余补0
W=K.copy()
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        if i!=j:
            W[i][j]=0
            # W为对角线元素提取的矩阵
print("取出对角线元素:","\n",W)
print("========================================================================")
# print("tset",K)
# 求对角矩阵 -1/2
v, Q = la.eig(W)
# print(v)
V = np.diag(v**(-0.5))
# print(V)
T = Q * V * la.inv(Q)
print("对角矩阵W^-1/2：","\n",T)
print("========================================================================")
print("规范化核矩阵：","\n",np.dot(np.dot(T,K),T))
print("========================================================================")
B=A.copy()
for i in range(data.shape[0]):
    for j in range(data.shape[0]):
        if i!=j:
            B[i][j]=0
            # W为对角线元素提取的矩阵
# print("tset",K)
# 求对角矩阵 -1/2
v, Q = la.eig(B)
# print(v)
V = np.diag(v**(-0.5))
# print(V)
T = Q * V * la.inv(Q)
print("对中心化核矩阵进行规范化后的核矩阵：","\n",np.dot(np.dot(T,A),T))
print("========================================================================")
#
#
# 第二问
m=np.zeros((1,4))
# 初始化特征空间
n=np.zeros((data.shape[0],10))
# print(data.shape[0])
for i in range(data.shape[0]):
    m=data[i]
    n[i][0] = (m[0]*m[0])
    n[i][1] = (m[1]*m[1])
    n[i][2] = (m[2] * m[2])
    n[i][3] = (m[3] * m[3])
    n[i][4] = (m[0] * m[1]) * math.sqrt(2)
    n[i][5] = (m[0] * m[2]) * math.sqrt(2)
    n[i][6] = (m[0] * m[3]) * math.sqrt(2)
    n[i][7] = (m[1] * m[2]) * math.sqrt(2)
    n[i][8] = (m[1] * m[3]) * math.sqrt(2)
    n[i][9] = (m[2] * m[3]) * math.sqrt(2)
np.set_printoptions(threshold=1000)
print("特征空间为：","\n",n)
print("========================================================================")
#  对特征空间中心化  每一个点都减去μ！！！
# u为均值 sum为均值存放数组 temp为临时量
#初始化 中心化矩阵为m
m=np.zeros((n.shape[0],n.shape[1]))
for i in range(n.shape[0]):
    for j in range(n.shape[1]):
        m[i][j]=n[i][j]-np.mean(n[:,j])
print("特征空间中心化后的结果：","\n",m)
print("========================================================================")
# 开始均一化
# 求出向量的模
sum2=np.zeros((1,150))
for i in range(m.shape[0]):
    m2=np.copy(m[i])
    # 每一个向量的模都存到数组里
    sum2[0][i]=math.sqrt(np.dot(m[i],np.transpose(m2)))
# print(sum2)
#  每一个向量除以模  初始化归一矩阵为g
g=np.zeros((150,10))
for i in range(sum2.shape[1]):
    g[i]=m[i]*(1/sum2[0][i])
print("中心化归一化后的特征空间为：","\n",g)
print("========================================================================")
# 验证特征空间产生的核矩阵是否与之前的相等  pair-wise dot
# 定义特征空间计算出的和矩阵为l 并初始化
l=np.zeros((g.shape[0],g.shape[0]))
h=np.copy(g)
for i in range(g.shape[0]):
    for j in range(g.shape[0]):
        l[i][j]=np.dot(g[i],np.transpose(h[j]))
print("特征空间生成的核矩阵为：","\n",l)




