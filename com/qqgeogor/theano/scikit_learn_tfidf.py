#encoding=utf-8
'''
Created on 2015-4-13

@author: qq
'''

import jieba
import jieba.posseg as pseg

import codecs

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from read_files import GetFileList

import random
from numpy.linalg  import norm
import numpy.matlib as ml
from matplotlib import pyplot
import pickle
from numpy import zeros, array
from numpy.lib.shape_base import tile
from com.qqgeogor.theano.kmeans import kmeans

path = "4.txt"
file = codecs.open(path,"r",'gbk')

content = file.read()

words1 = jieba.cut(content, cut_all=False)

array1= ""
for word in words1:
    array1+=(word+' ')



path = "4.txt"
file = codecs.open(path,"r",'gbk')

content = file.read()
array2 = ""
words2 = jieba.cut(content, cut_all=False)
for word in words2:
    array2+=(word+' ')

path = "8.txt"
file = codecs.open(path,"r",'gbk')

content = file.read()

words3 = jieba.cut(content, cut_all=False)
array3=""
for word in words3:
    array3+=(word+' ')

'''

corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
    "他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
    "小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
    "我 爱 北京 天安门"]#第四类文本的切词结果
'''
res=GetFileList("")
print res

def cutfile(path):
    #path = "8.txt"
    file = codecs.open(path,"r",'gbk')
    
    content = file.read()
    
    words = jieba.cut(content, cut_all=False)
    array=""
    for word in words:
        array+=(word+' ')
    return array
    

corpus = []

for f in res:
    str=cutfile(f)
    corpus.append(str)


vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
    for j in range(len(word)):
        print word[j],weight[i][j]
        


N = 0
for smp in weight:
    N += len(smp[0])
X = zeros((N, 2))
idxfrm = 0
for i in range(len(weight)):
    idxto = idxfrm + len(weight[i][0])
    X[idxfrm:idxto, 0] = weight[i][0]
    X[idxfrm:idxto, 1] = weight[i][1]
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

kmeans(X, 2)
