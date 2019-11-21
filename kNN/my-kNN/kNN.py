import numpy as np
from math import sqrt
from collections import Counter


class kNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be positive int"

        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集 X_train 和 y_train 训练(拟合)分类器"""
        assert X_train.shape[0] == y_train.shape[0], "X_train size must equals to y_train size"
        assert X_train.shape[0] >= self.k, "X_train size must be bigger than k"

        # 和其他机器学习的算法不同, 其实这里的训练和拟合什么也没做, 只是把参数保存在实例的参数
        self._X_train = X_train
        self._y_train = y_train

        # 可以不返回, 参考 scikit-learn 中的实现, 拟合后返回自己实例
        return self

    def predict(self, X_predict):
        """给定待预测的数据集, 返回结果向量"""
        assert self._X_train is not None and self._y_train is not None, "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1], "the feature number of X_predict must equals to X_train"

        # 针对每个待预测的特征向量, 分别调用内部方法计算
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个预测值, 返回预测结果"""
        assert x.shape[0] == self._X_train.shape[1], "the feature number of x must be equal to X_train"

        # 计算 x 到每个样本点的距离, 并形成一个数组
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]

        # 把距离排序，按照由近到远的顺序，返回的是点的索引
        nearest = np.argsort(distances)

        # 在训练集的结果中, 对应找到距离最近的几个结果
        topK_y = [self._y_train[i] for i in nearest[:self.k]]

        # 使用 Python 内置的 Counter 类来做分组统计
        votes = Counter(topK_y)

        # 使用 Counter 的内置方法，找到出现最多的第一组
        most = votes.most_common(1)

        # 就得到了结果
        return most[0][0]

    def __repr__(self):
        return "kNN(k=%d)" % self.k
