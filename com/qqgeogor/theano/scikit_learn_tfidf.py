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
from matplotlib import pyplot as plt
import numpy as np
import mlpy
import pickle
from numpy import zeros, array
from numpy.lib.shape_base import tile

from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.datasets import SequentialDataSet,SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer,SoftmaxLayer
from pybrain.structure import LSTMLayer,LinearLayer

def cutfile(path):
    #path = "8.txt"
    file = codecs.open(path,"r",)
    
    content = file.read()
    
    words = jieba.cut(content, cut_all=False)
    array=""
    for word in words:
        array+=(word+' ')
    return array

def create_weight():
    res1=GetFileList("datas/d1",'.txt')
    res2=GetFileList("datas/d2",'.txt')
    res3=GetFileList("datas/d3",'.txt')
    res4=GetFileList("datas/d4",'.txt')
   
    y = []
    corpus = []
    for f in res1:
        str=cutfile(f)
        corpus.append(str)
        y.append([0,0])
        
    for f in res2:
        str=cutfile(f)
        corpus.append(str)
        y.append((0,1))
        
    for f in res3:
        str=cutfile(f)
        corpus.append(str)
        y.append((1,0))
    for f in res4:
        str=cutfile(f)
        corpus.append(str)
        y.append((1,1))
    res =res1+res2+res3+res4
    print res
    print y
    
    
        
    
   
    
    
        
    
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    '''
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
        for j in range(len(word)):
            print word[j],weight[i][j]
    '''
    o1 = open('weight.pkl','wb')
    o2 = open('y.pkl','wb')        
    pickle.dump(weight,o1)
    pickle.dump(y,o2)
    o1.close()
    o2.close()


def plot_pca(weight,y):
    
   
    pca = mlpy.PCA() # new PCA instance
    pca.learn(weight) # learn from data
    z = pca.transform(weight, k=22) # embed x into the k=2 dimensional subspace
    oz = open('z.pkl','wb')        
    pickle.dump(z,oz)
    oz.close()
    fig1 = plt.figure(1)
    title = plt.title("PCA on novels dataset")
    plot = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    plt.show()




def mlp_network(z,y):
    input_size,output_size = len(z[0]),len(y)
    ds = SupervisedDataSet(input_size,output_size)
    print input_size
    for i in range(len(z)-3):
        ds.addSample(z[i],y[i])
    
    net = buildNetwork(input_size, 9,output_size, 
                   hiddenclass=SigmoidLayer, outputbias=False, recurrent=False)    

    trainer = RPropMinusTrainer(net, dataset=ds,)
    #trainer = BackpropTrainer( net, dataset=ds,verbose=True, momentum=0.9, learningrate=0.00001 )
    train_errors = [] # save errors for plotting later
    EPOCHS_PER_CYCLE = 5
    CYCLES = 100
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i+1) * EPOCHS_PER_CYCLE
        print "\r epoch {}/{}".format(epoch, EPOCHS)
    
    on = open('net.pkl','wb')
    pickle.dump(net,on)
    on.close()

    print("final error =", train_errors[-1])    
    plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.show()


def pred(net,z):
    yp = []
    
    for sample in z:
        yp.append(net.activate(sample)[0])
    print yp
    
    xmin, xmax = z[:,0].min()-0.1, z[:,0].max()+0.1
    ymin, ymax = z[:,1].min()-0.1, z[:,1].max()+0.1
    xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
    zgrid = np.c_[xx.ravel(), yy.ravel()]
    fig2 = plt.figure(2)
    title = plt.title("MLP on principal components")
    
    #plot1 = plt.pcolormesh(xx, yy, yp.reshape(xx.shape))
    plot2 = plt.scatter(z[:, 0], z[:, 1], c=yp)
    #plot1 = plt.scatter(z[:, 0], z[:, 1], c=y)
    labx = plt.xlabel("First component")
    laby = plt.ylabel("Second component")
    limx = plt.xlim(xmin, xmax)
    limy = plt.ylim(ymin, ymax)
    plt.show()

if __name__ == '__main__':
   
    #create_weight()
    o1 = open('weight.pkl','r')
    o2 = open('y.pkl','r') 
    weight= pickle.load(o1)
    y = pickle.load(o2)
    plot_pca(weight,y)
    oz = open('z.pkl','r') 
    z = pickle.load(oz)
    mlp_network(z,y)
    
    on = open('net.pkl','r') 
    net = pickle.load(on)
    pred(net,z)
    o1.close()
    o2.close()