import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import pickle
import cmath
import math
import scipy.stats as scita
import scipy.signal as signal
import scipy.io as scio
from numpy import linalg as la
from numpy import *
import lmdb
import caffe

winLen=30

def db(x):
#变换到log空间
	if (x==0):
		ans=0
	else:
		ans=20*(np.log10(x))
	return ans

def butterworth_II(file,Fc):
#去噪
	N  = 2    # Filter order
	Wn = 2*(np.pi)*Fc # Cutoff frequency
	B, A = signal.butter(N, Wn, output='ba')
	ret = signal.filtfilt(B,A,file)
	return ret

def relative_phase(tmp1,tmp2):
#计算相对相位
	tmp=tmp1*np.conjugate(tmp2)
	tmp_1=(tmp.real)/(abs(tmp))
	ret=np.arccos(tmp_1)

	return (ret)




def file_data(filename):
#解析原始数据包
	l=[]
	with open(filename, "rb") as f:
		while 1:
			try:
				flag = 1
				k = pickle.load(f)
				for i in range(90):
					if(abs(k[i]) <= 0):
						flag = 0
				if flag == 0:
					continue
				l.append(k[0:90])
			except Exception as e:
				break
	a = np.array(l).T
	return(a)

def csi_amplitude(file):
	[row,col]=file.shape
	newFile = np.zeros((row,col))
	for i in range(row):
		for j in range(col):
			newFile[i,j]=db(abs(file[i,j]))
			#print (file[i,j])
	ret=np.array(newFile)
	return (ret.real)

def csi_relative_phase(file):
#计算数据的相对相位
	file=file.reshape(3,30,-1)
	[row,col,other]=file.shape
	csi_ant1=file[0]
	csi_ant2=file[1]
	csi_ant3=file[2]
	rephase1_2=[]
	rephase1_3=[]
	rephase3_2=[]
	rephase_all=[]

	for i in range(col):
		tmp1_2=relative_phase(csi_ant1[i],csi_ant2[i])

		tmp1_3=relative_phase(csi_ant1[i],csi_ant3[i])
		tmp3_2=relative_phase(csi_ant3[i],csi_ant2[i])
		rephase1_2.append(tmp1_2)
		rephase1_3.append(tmp1_3)
		rephase3_2.append(tmp3_2)

	for j in range(30):
		rephase_all.append(rephase1_2[j])
	for j in range(30):
		rephase_all.append(rephase1_3[j])
	for j in range(30):
		rephase_all.append(rephase3_2[j])

	ret=np.array(rephase_all)
	return ret.real

def get_characters(matrix, num, label):
#计算出相应特征值
	max = []
	min = []
	mean = []
	skewness = []
	kurtosis = []
	std = []
	i = 1
	col = matrix.shape[1]
	chunk = int(col / num)
	while (i) * chunk <= col and i <= num:
		tmp = matrix[:, chunk * (i-1):chunk * (i)]
		i = i + 1
		cnt = 0
		max_t = []
		min_t = []
		mean_t = []
		skewness_t = []
		kurtosis_t = []
		std_t = []
		while cnt < 90:
			t = tmp[cnt:]
			max_t.append(np.max(t))
			min_t.append(np.min(t))
			mean_t.append(np.mean(t))
			skewness_t.append(np.mean(scita.skew(t,axis=1, bias=True)))
			kurtosis_t.append(np.mean(scita.kurtosis(t,axis=1, bias=True)))
			std_t.append(np.std(t))
			cnt = cnt + 1
		max.append(max_t)
		min.append(min_t)
		mean.append(mean_t)
		skewness.append(skewness_t)
		kurtosis.append(kurtosis_t)
		std.append(std_t)
	l = []
	for i in range(0, 540):
		l.append(label)
	max = np.array(max).T
	min = np.array(min).T
	mean = np.array(mean).T
	skewness = np.array(skewness).T
	kurtosis = np.array(kurtosis).T
	std = np.array(std).T
	result = np.append(max, min, axis=0)
	result = np.append(result, mean, axis=0)
	result = np.append(result, skewness, axis=0)
	result = np.append(result, kurtosis, axis=0)
	result = np.append(result, std, axis=0)
	return result, l

def classification(input):
#根据标签list转化为标签矩阵（one-hot编码）
	labelOrder=set()
	labelList=[]
	i=0
	for label in input:
		label=label.split('_')[0]
		labelList.append(label)
		labelOrder.add(label)
	labelOrder=list(labelOrder)
	labelCol=len(labelOrder)
	labelMatrix = []
	for label in labelList:
		l = [0] * labelCol
		labelNum = labelOrder.index(label)
		l[labelNum] = 1
		labelMatrix.append(l)
	labelMatrix = np.array(labelMatrix, dtype=object)
	print(labelMatrix.shape)
	return labelMatrix,labelOrder


		
			
	

def train_data(path,num):
#输入为路径和特征值数量
	files= os.listdir(path)
	labelList=[]
	featureList=[]
	for file in files:
		if file == 'over':
			continue
		for x in os.listdir(path + '/' + file):
			print(x)
			csiData=file_data(str(path) + '/' + file + '/' + x)
			csiAmplitude=csi_amplitude(csiData)
			csiAmplitude=butterworth_II(csiAmplitude,0.03)
			csiFeature, labels=get_characters(csiAmplitude,num,file)
			labelList.extend(labels)
			featureList.extend(csiFeature)
	labelTmp=np.array(labelList)
	featureMatrix=np.array(featureList)
	labelMatrix,labelOrder=classification(labelTmp)
	return featureMatrix,labelMatrix,labelOrder


def test_input(path,num):
	files = os.listdir(path)
	print(files)
	featureList = []
	for file in files:
		csiData = file_data(str(path) + '/' + file)
		csiAmplitude = csi_amplitude(csiData)
		csiAmplitude = butterworth_II(csiAmplitude, 0.03)
		csiFeature, labels = get_characters(csiAmplitude, num, file)
		featureList.extend(csiFeature)
	featureMatrix = np.array(featureList)
	return featureMatrix

def turn_to_lmdb():#将csi数据转换为lmdb格式主文件
    data,label,labelOrder1=train_data(rootpath + '\\datatrain', 784)
    N = 1000

# Let's pretend this is interesting data
    X = data
    y = label

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
    map_size = X.nbytes * 10
    env = lmdb.open('mylmdb', map_size=map_size)

    with env.begin(write=True) as txn:
    # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()#处理好这五个变量即可
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(y[i])
            str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
    env.close
