import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
# 加载数据
data = np.loadtxt("magic04.txt", skiprows=0, delimiter=",", usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
# 通过切片获取x1
x1 = data[:, 0]
# 通过切片获取x2...
x2 = data[:, 1]
x3 = data[:, 2]
x4 = data[:, 3]
x5 = data[:, 4]
x6 = data[:, 5]
x7 = data[:, 6]
x8 = data[:, 7]
x9 = data[:, 8]
x10 = data[:, 9]
arr_mean = np.mean(x1)
arr_mean2 = np.mean(x2)
arr_mean3 = np.mean(x3)
arr_mean4 = np.mean(x4)
arr_mean5 = np.mean(x5)
arr_mean6 = np.mean(x6)
arr_mean7 = np.mean(x7)
arr_mean8 = np.mean(x8)
arr_mean9 = np.mean(x9)
arr_mean10 = np.mean(x10)
print("多元均值向量为：(", arr_mean , ",", arr_mean2, ",",
      arr_mean3, ",", arr_mean4, ",", arr_mean5,", ", arr_mean6, ", ",
      arr_mean7,", ", arr_mean8,", ",arr_mean9,", ",arr_mean10, ")")


class CCovMat(object):

      def __init__(self, samples):
            self.samples = samples
            self.covmat1 = []  # 保存方法1求得的协方差矩阵
            self.covmat2 = []  # 保存方法1求得的协方差矩阵

            # 用方法1计算协方差矩阵
            self._calc_covmat1()
            # 用方法2计算协方差矩阵
            self._calc_covmat2()

      def _covariance(self, X, Y):
            n = np.shape(X)[0]
            X, Y = np.array(X), np.array(Y)
            meanX, meanY = np.mean(X), np.mean(Y)
            # 按照协方差公式计算协方差，Note:分母一定是n-1
            cov = sum(np.multiply(X - meanX, Y - meanY)) / (n - 1)
            return cov

      def _calc_covmat1(self):
            '''
            方法1：外积计算
            '''
            S = self.samples  # 样本集
            ns = np.shape(S)[0]  # 样例总数
            mean = np.array([np.mean(attr) for attr in S.T])  # 样本集的特征均值
            print('样本集的特征均值:\n', mean)
            centrS = S - mean  ##样本集的中心化
            self.covmat1 = np.zeros((centrS.shape[1], centrS.shape[1]))
            for i in range(centrS.shape[0] ):
                  self.covmat1 += np.kron(centrS[[i]].T, centrS[[i]]) * 1 / (centrS.shape[0] - 1)
            return self.covmat1

      def _calc_covmat2(self):
            '''
            方法2：先样本集中心化再求协方差矩阵
            '''
            S = self.samples  # 样本集
            ns = np.shape(S)[0]  # 样例总数
            mean = np.array([np.mean(attr) for attr in S.T])  # 样本集的特征均值
            print('样本集的特征均值:\n', mean)
            centrS = S - mean  ##样本集的中心化
            print('样本集的中心化(每个元素将去当前维度特征的均值):\n', centrS)
            # 求协方差矩阵
            self.covmat2 = np.dot(centrS.T, centrS) / (ns - 1)
            return self.covmat2

      def CovMat1(self):
            return self.covmat1

      def CovMat2(self):
            return self.covmat2


if __name__ == '__main__':
      samples = np.array(data)
      cm = CCovMat(samples)

      # print('样本集):\n', samples)
      print('按照外积:\n', cm.CovMat1())
      print('按照样本集的中心化内积求得的协方差矩阵:\n', cm.CovMat2())
      print('numpy.cov()计算的协方差矩阵:\n', np.cov(samples.T))
