import numpy as np
import math
data = np.loadtxt("iris.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3))
# 计算每个点的密度吸引子
# 密度大于最小阈值，将其加入吸引子集合
# 对应数据加入吸引点集合
# 找出吸引子最大子集 使任何一对都互相密度可达
# 最大子集构成簇的种子
# 被吸引到密度吸引子周围的点加入集合构成簇

# 距离函数
def distance(X1,Xi):
    long=0
    for i in range(len(X1)):
        long= long + pow((X1[i] - Xi[i]), 2)
    long=math.sqrt(long)
    return long


#核函数
def kernel(X, Xi, h, degree):
    dist=distance(X,Xi)
    kernel=(1/pow(2*np.pi,degree/2))*np.exp(-(dist*dist)/(2*h*h))
    return kernel

#均值漂移函数
def shiftPoint(center,points,h):
    a=np.zeros([1,4])
    sum=0
    newcenter=np.zeros((1,4))#新的中心点
    for temp in points:
        weight=kernel(center, temp, h, 4)
        for i in range(0,4):
            a[0,i]+=weight*temp[i]
        sum+=weight
    for j in range(0,4):
        a[0,j]=a[0,j]/sum
        newcenter[0,j]=a[0,j]
    return newcenter

#定义寻找密度吸引子函数
def FindAttractor(X,D,h,si):
    t=0
    n=len(D)
    Xt=np.zeros([n,4])
    pointlist=[]
    Xt[t,:]=X
    for i in D:
        if distance(i, Xt[t, :])<=h:
            pointlist.append(i)
    Xt[t+1,:]=shiftPoint(X,pointlist,h)
    pointlist.clear()
    t=t+1
    while distance(Xt[t,:],Xt[t-1,:])>=si:
        # print(distance(Xt[t,:],Xt[t-1,:]))
        for i in D:
            if distance(i, Xt[t, :])<=h:
                pointlist.append(i)
        Xt[t+1]=shiftPoint(X,pointlist,h)
        pointlist.clear()
        t=t+1
#返回密度吸引子和漂移经过的点

    return Xt[t],Xt[1:t+1,:]


def DensityThreshold(X,points,h,degree):
    threshold=0
    for item in points:
        threshold= threshold + (1/len(points)*math.pow(h,degree)) * kernel(X, item, h, degree)
    return threshold

#定义DENCLUE函数
def Denclue(D:[], c, si, h):
    # 密度吸引子
    A=[]
    # 密度吸引子吸引的点
    R={}
    # 密度可达
    C={}
    points=[]
    num_of_attractor=0
    need_shift=[True]*len(D)
    global degree
    w=0
    for i in range(0,len(D)):
        if not need_shift[i]:
            continue
        X_star,shiftpoints=FindAttractor(D[i,:],D,h,si)
        # print(DensityThreshold(X_star,D,h,degree))
        for x in range(0,len(D)):
            if distance(X_star,D[x,:])<=h:
                points.append(D[x,:])
        # 阈值
        # print(DensityThreshold(X_star,points,h,4))
        if DensityThreshold(X_star,points,h,4)>=c:
            A.append(X_star)
            R.setdefault(num_of_attractor,[])
            for item in shiftpoints:
                for j in range(0,len(D)):
                    if need_shift[j]:
                        if distance(item,D[j,:])<=h:
                            # 均值漂移经过的点
                            R.get(num_of_attractor).append(D[j,:])
                            need_shift[j]=False
            num_of_attractor+=1
        points.clear()
    #输出 密度吸引子
    for i in range(0,len(A)):
        print("密度吸引子"+str(A[i]))
        print("被密度吸引子吸引的点")
        print(R[i])
    t=0
    C_star=np.empty([len(A),1],dtype=int)
    for i in range(0,len(A)):
        C_star[i]=-1

    for k in range(0,len(A)):
        if C_star[k]==-1:
            C_star[k]=t
            if k!=len(A):
                for i in range(k,len(A)-1):
                    # 判断密度吸引子是否密度可达
                    if distance(A[k],A[i+1])<=h:
                    #如果这两个密度吸引子的距离小于带宽，记录
                        C_star[i+1]=C_star[k]
            t=t+1
    number=0
    for i in range(0,len(A)):
        if C_star[i,0] not in C:
            number+=1
            C.setdefault(C_star[i,0],[])
            C.get(C_star[i,0]).append(R[i])
        else:
            C.get(C_star[i,0]).append(R[i])

    #统计每个类的个数
    class_element_number=[0]*number
    for i in range(0,len(A)):
        for j in range(0,number):
            if C_star[i]==j:
                class_element_number[j]+=len(R[i])
    # print(C_star)
    print("类簇内个数：")
    for i in range(0, number):
        print("类簇", str(i + 1), ":  ", str(class_element_number[i]), "\n")
    #输出每个类簇的点
    for i in range(0,number):
        print("类簇"+str(i+1)+"的点有：")
        print(C[i])
    return

if __name__ == '__main__':
    Denclue(data,0.001,0.0001,1)