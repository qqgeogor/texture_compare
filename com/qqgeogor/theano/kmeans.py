#encoding=utf-8

import random
from numpy.linalg  import norm
import numpy.matlib as ml
from matplotlib import pyplot
import pickle
from numpy import zeros, array
from numpy.lib.shape_base import tile
def kmeans(n,k,observer=None,threshold=1e-15,maxiter=300):
    #获取样本点数组的长度
    N = len(n)
    #创建每个样本点的标签，初始化为0
    labels = zeros(N,dtype=int)
    #从n个样本中选择k个点作为初始化中心
    centers = array(random.sample(n, k))
    #迭代次数
    iter = 0
    
    #计算迭代计算sum，sum为所有聚类中点到对应聚类中心距离的和
    def calc_J():
        sum = 0
        for i in xrange(N):
            sum+=norm(n[i]-centers[labels[i]])
        return sum
    #计算点矩阵X与点Y之间的距离，返回一个数组
    def dismat(X,Y):
        n = len(X)
        m = len(Y)
        xx = ml.sum(X*X,axis=1)
        yy = ml.sum(Y*Y,axis=1)
        xy = ml.dot(X,Y.T)
        
        return tile(xx, (m, 1)).T+tile(yy, (n, 1)) - 2*xy
    
    Jprev = calc_J()
    
    print "first time is %d" % Jprev
    while True:
        if observer is not None:
            observer(iter,labels,centers)
        # calculate distance from n to each center
        # distance_matrix is only available in scipy newer than 0.7
        dist = dismat(n,centers)
        # assign n to the nearest center
        labels = dist.argmin(axis=1)
        # re-calculate each center
        for j in range(k):
            idx_j = (labels == j).nonzero()
            centers[j] = n[idx_j].mean(axis=0)
        J = calc_J()
        print "next time is %d" % Jprev
        iter += 1
 
        if Jprev-J < threshold:
            break
        Jprev = J
        if iter >= maxiter:
            break
 
        # final notification
        if observer is not None:
            observer(iter, labels, centers)

if __name__ == '__main__':
    # load previously generated points
    with open('cluster.pkl') as inf:
        samples = pickle.load(inf)
    N = 0
    for smp in samples:
        N += len(smp[0])
    X = zeros((N, 2))
    idxfrm = 0
    for i in range(len(samples)):
        idxto = idxfrm + len(samples[i][0])
        X[idxfrm:idxto, 0] = samples[i][0]
        X[idxfrm:idxto, 1] = samples[i][1]
        idxfrm = idxto
 
    def observer(iter, labels, centers):
        print "iter %d." % iter
        colors = array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        pyplot.plot(hold=False)  # clear previous plot
        pyplot.hold(True)
 
        # draw points
        data_colors=[colors[lbl] for lbl in labels]
        pyplot.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
        # draw centers
        pyplot.scatter(centers[:, 0], centers[:, 1], s=200, c=colors)
 
        pyplot.savefig('iter_%d.png' % iter, format='png')
 
    kmeans(X, 3, observer=observer)
        
        