from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class KPCA():
    def __init__(self, kernel='rbf'):
        self.kernel = kernel

    def fit_transform_plot(self, X, y):
        if self.kernel == 'None':
            C = np.cov(X.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            arg_max = eigvals.argsort()[-10:]
            eigvecs_max = eigvecs[:, arg_max]
            K = X
        else:
            if self.kernel == 'linear':
                K = np.dot(X, X.T)
            elif self.kernel == 'polynomial':
                K = np.dot(X, X.T) + 1
                K = K*K
            elif self.kernel == 'log':
                dists = pdist(X) ** 0.10
                mat = squareform(dists)
                K = -np.log(1 + mat)
            elif self.kernel == 'rbf':
                dists = pdist(X) ** 10
                mat = squareform(dists)
                beta = 10
                K = np.exp(-beta * mat)
            else:
                print('kernel error!')
                return None
            N = K.shape[0]
            one_n = np.ones([N, N]) / N
            K_hat = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
            # 求解出特征向量alpha
            eigvals, eigvecs = np.linalg.eigh(K_hat)
            # 取出特征值最大的10个对应特征向量
            arg_max = eigvals.argsort()[-10:]
            eigvecs_max = eigvecs[:, arg_max]
        # 求解提升维度后的座标点在各个特征向量上的对应位置，本例中提升到三维，即抽取三个特征向量后处理的值(500个500维的矩阵，点乘500维的三个特征向量)
        X_new = np.dot(K, eigvecs_max)
        for i in range(2):
            # 布尔索引 https://www.jianshu.com/p/743b3bb340f6
            tmp = y == i
            # 将矩阵拉成向量
            tmp = tmp.ravel()
            Xi = X_new[tmp]
            a = Xi[:, -2]
            # l1 = plt.scatter(Xi[0, -1], Xi[0, -2], color="r", label="a points", marker="+")
            # l2 = plt.scatter(Xi[0, -1], Xi[0, -2], color="b", label="b points", marker="x")
            if i == 0:
                l1 = ax.scatter(Xi[:, -1], Xi[:, -2], Xi[:, -3], color="r", label="a points", marker="+")
                # l1 = plt.scatter(Xi[:, -1], Xi[:, -2], color="r", label="a points", marker="+") # 显示二维
            if i == 1:
                l2 = ax.scatter(Xi[:, -1], Xi[:, -2], Xi[:, -3], color="b", label="b points", marker="x")
                # l2 = plt.scatter(Xi[:, -1], Xi[:, -2], color="b", label="b points", marker="x") # 显示二维
            # plt.legend(handles=[l1, l2], labels=["a points", "b points"])
            # plt.scatter(Xi[:, 0], Xi[:, 1])
        plt.show()


if __name__ == '__main__':
    data = pd.read_csv("./ellipse_dataset1.csv")
    X = data.loc[:, ["x1", "y1"]]
    X = X.to_numpy()
    y = data.loc[:, ["category"]]
    y = y.to_numpy().T

    # 创建画布
    # fig = plt.figure(figsize=(8, 8))
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    # ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    # fig.add_axes(ax)

    # ax.axis[:].set_visible(False)  # 通过set_visible方法设置绘图区所有坐标轴隐藏
    # ax.axis["x"] = ax.new_floating_axis(0, 0)  # ax.new_floating_axis代表添加新的坐标轴
    # ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加上箭头
    # # 添加y坐标轴，且加上箭头
    # ax.axis["y"] = ax.new_floating_axis(1, 0)
    # ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # # 设置x、y轴上刻度显示方向
    # ax.axis["x"].set_axis_direction("top")
    # ax.axis["y"].set_axis_direction("right")
    # plt.xlim(-50, 50)
    # plt.ylim(-50, 50)

    # 创建3d视图
    ax = plt.subplot(111, projection='3d')

    kpca = KPCA('polynomial')
    kpca.fit_transform_plot(X, y)
