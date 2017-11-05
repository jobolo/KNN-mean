#-*- coding: utf-8 -*-  
from numpy import *
import operator
from os import listdir

def classfy0(inX, dataSet, lables, k):
    #距离计算
    dataSetSize=dataSet.shape[0]#得到数组的行数，即知道几个训练数据
    diffMat=tile(inX,(dataSetSize,1))-dataSet#tile沿y轴方向复制4，x轴复制1然后减去之前的数组
    sqDiffMat=diffMat**2#平方
    sqDistance=sqDiffMat.sum(axis=1)#列相加
    distance=sqDistance**0.5#开根号得到距离
    sortedDistIndicies=sqDistance.argsort()#按元素排列
    classCount={}#存放投票结果
    for i in range(k):#距离最近的ｋ个循环投票过程
        votelable=lables[sortedDistIndicies[i]]# 给第ｋ个点标签
        classCount[votelable]=classCount.get(votelable,0)+1#每个标签出现次数
        sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)#对分类结果按照出现次数进行分类并返回最多的结
        return sortedClassCount[0][0] 
'''
if __name__== "__main__":  
    # 导入数据  
    dataset, labels = createDataSet()  
    inX = [0.1, 0.1]  
    # 简单分类  
    className = classfy0(inX, dataset, labels, 3)  
    print 'the class of test sample is %s' %className  
'''
def file2matrix(filename):
    fr=open(filename)#打开文档
    arrayOlines=fr.readlines()#读取行数存到数组中
    numberOfLines=len(arrayOlines)#计算行数
    returnMat=zeros((numberOfLines,3))#初始化矩阵，行为numberOfLines，列为３
    classLableVector=[]#创建数组存储标签
    index=0#初始化索引值为0
    for line in arrayOlines:
        line=line.strip()#删除空格
        listFromLine=line.split('\t')#以\t分割
        returnMat[index,:] = listFromLine[0:3]#提取前３个元素存储到特征矩阵
        classLableVector.append(int(listFromLine[-1]))#添加最后一行元素，存储为整数类型
        index += 1#索引自增
    return returnMat,classLableVector#返回值
def autoNorm(dataSet):
    #归一化数值方法
    minVals=dataSet.min(0)
    #获得最小值
    maxVals=dataSet.max(0)
    #获得最大值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    #返回数组
    m=dataSet.shape[0]
    #获得行数
    normDataSet=dataSet-tile(minVals,(m,1))
    #通过tile函数扩展成m×3的矩阵
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,range,minVals
def datingClassTest():
    hoRatio=0.10
    #随机10%作为测试样例
    datingDataMat,datingLables=file2matrix('datingTestSet2.txt')
    #载入数据
    norMat,ranges,minVals=autoNorm(datingDataMat)
    #数据归一化
    m=norMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classfierResult=classfy0(norMat[i,:],norMat[numTestVecs:m,:],datingLables[numTestVecs:m],3)
        #输入参数:normMat[i,:]为测试样例，表示归一化后的第i行数据  
        #normMat[numTestVecs:m,:]为训练样本数据，样本数量为(m-numTestVecs)个  
        #datingLabels[numTestVecs:m]为训练样本对应的类型标签  
        #k为k-近邻的取值
        print "the classifier came back with:%d,the real answer is:%d"%(classfierResult,datingLables[i])
        if (classfierResult!=datingLables[i]):
            errorCount+=1.0
    print "the total error rate is:%f"%(errorCount/float(numTestVecs))
def classfyPerson():
    resultList=['not at all','in small doses','in large doses']
    percentTats=float(raw_input("percentage of time spent playing video game?"))
    ffMiles=float(raw_input("frequent flier miles earn per yer?"))
    iceCream=float(raw_input("liners of ice cream consumed per year?"))
    datingDataMat,datingLables=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classfierResult=classfy0((inArr-minVals)/ranges,norMat,datingLables,3)
    print"you will probably like this person:",resultList[classfierResult-1]

def img2vector(filename):
    #创建1*1024的数组，循环读出文件前32行，并将每行头32个字符值存储到数组中，返回数组
    returnVector=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector

def handwritingClassTest():
    hwLables=[]
    trainingFileList=listdir('trainingDigits')
    #获取目录内容
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        #从文件名解析分类数字
        hwLables.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest= img2vector('testDigits/%s' % fileNameStr)
        classfierResult = classfy0(vectorUnderTest, trainingMat, hwLables, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classfierResult, classNumStr)
        if(classfierResult!=classNumStr):errorCount+=1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
