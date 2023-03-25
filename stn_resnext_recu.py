import numpy as np 

import itertools
import h5py as h5py

import matplotlib as mpl
mpl.use('Agg')


from numpy import linalg as LA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import scipy.io
from pandas.plotting import register_matplotlib_converters
from matplotlib import cm
import tensorflow as tf
from tensorflow import keras

from keras.layers import Dense, Dropout, Activation,Flatten
#from tensorflow.keras.applications import imagenet_utils
#from pyimagesearch.gradcam import GradCAM
import os
import random
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard


from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
register_matplotlib_converters()
from sklearn.preprocessing import OneHotEncoder

import requests
import sonnet as snt

print(dir(snt))
print(snt.__version__)

# extra code
import helpers

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
#tf.compat.v1.enable_eager_execution()
#load data set and prepare X_train, X_test, Y_train, Y_test
#dataset is given and presents dictionary whith keys whose format is tuple ('modulation_name',snr_db)
#each key has assigned matrix (1000,2,128) where is 1000 number of observations, 2 is I and Q ()
#first element is real part while the second is imaginary part, and 128 
#is number of samples
train=True
N_samples=1024 #need to fix this why it 65
N=800

#dataset_name='../dataset1024.mat'
#dataset_name='../dataset1024_new.mat'
#dataset_name='../dataset1024_rayleigh_up.mat'
#dataset_name='../dataset1024_rician_up.mat'
#dataset_names=['../dataset1024_rayleigh_fs_200.mat']
#dataset_names=['../dataset1024_rician_fs_200.mat']
#dataset_names=['../dataset1024_noenergynormiq.mat']
#dataset_names=['../dataset1024_iqnoenergy_rayLsTest.mat']
#dataset_names=['../dataset1024_noenergynormiq_rici8.mat']
#dataset_names=['../dataset1024_noenergynormiq_rici8.mat']
#dataset_names=['../dataset1024_noenergynormiq_awgn8.mat','../dataset1024_noenergynormiq_rici8.mat','../dataset1024_noenergynormiq_rayleigh8.mat']
#dataset_names=['../dataset1024_noenergynormiq_awgn8.mat']
#dataset_names=["../dataset1024_noenergynormiq_awgn8.mat", "../dataset1024_noenergynormiq_rici8.mat","../dataset1024_noenergynormiq_rayleigh8.mat"]
#dataset_names=['../dataset1024_iqnoenergy_ray_1500fs_200sa.mat', '../dataset1024_iqnoenergy_rici_1500fs_200sa.mat','../dataset1024_iqnoenergy_awgn_1500fs_200sa.mat','../dataset1024_iqnoenergy_ray_300fs_200sa.mat', '../dataset1024_iqnoenergy_rici_300fs_200sa.mat','../dataset1024_iqnoenergy_awgn_300fs_200sa.mat']

#dataset_names=['../dataset1024_iqnoenergy_awgnLsTest.mat']
dataset_names=['../dataset1024_iqnoenergy_rici_Lstest_200sa.mat']
#dataset_names=['../dataset1024_iqnoenergy_rayLsTest.mat']
#dataset_name='../dataset1024_iqnoenergy_rayAlpha55.mat'
#dataset_name='../dataset1024_iqnoenergy_awgnAlpha55.mat'
#dataset_names=['../dataset1024_iqnoenergy_rayOther_200sa.mat']
#dataset_name='../dataset1024_iqnoenergy_riciAlpha55_200sa.mat'
#dataset_names=['../dataset1024_iqnoenergy_awgn_600fs_200sa.mat']
#dataset_names=['../dataset1024_iqnoenergy_rici_Lstest_200sa.mat','../dataset1024_iqnoenergy_rayLsTest.mat','../dataset1024_iqnoenergy_awgnLsTest.mat']

X=np.array([])
Y=np.array([])
			

X_train=np.array([])
amp_train=np.array([])
phase_train=np.array([])
Y_train=np.array([])
snr_mod_pairs_train=np.array([])

X_test=np.array([])
amp_test=np.array([])
phase_test=np.array([])
Y_test=np.array([])
snr_mod_pairs_test=np.array([])


X_valid=np.array([])
amp_valid=np.array([])
phase_valid=np.array([])
Y_valid=np.array([])
snr_mod_pairs_valid=np.array([])

snrs=[] #unique values of snr values
mods=[] #unique values of modulation formats
number_transmissions=[]
snr_mod_pairs=np.array([])
power_ratios=[]
n_ffts=[]
nsps=[]
bws=[]

max_number_transmissions=5
trainableStage1=False
trainableStage2=False
trainableStage3=False
trainableStage4=False
trainableStage5=False
trainableStage6=False
trainableSTN=False
trainableSTN_last=False
trainableStageDense=True
trainableGrid=False


split_factor_train=0.25 #split datasets to training and testing data
split_factor_valid=0.25


def gen_data():
	global X, Y, X_test, X_train, Y_train,Y_test, Y_valid, X_valid, snrs, mods, snr_mod_pairs, split_factor
	global snr_mod_pairs_test, snr_mod_pairs_train,Train, datasets_name, labels_name
	global power_ratios, train, bws, n_ffts, nsps, nsps_target, n_fft_target
	global snr_mod_pairs_valid
	print ("\n"*2,"*"*10,"Train/test dataset split - Start","*"*10,"\n"*2)
	print ("Doing")
	
	for dataset_name in dataset_names:
		print("dataset is ",dataset_name)
		
		f = h5py.File(dataset_name, 'r')
		ds=f['ds']

		for key in ds.keys():
			#print(key)
			pom=key.split('rer')
			mod=str(pom[0])
			snr=int(pom[1].replace('neg','-'))
			l=pom[2]
			freq=str(pom[3])
			m=pom[4]
			nt=int(pom[5])

			sps=round(float(l)/float(m),3)
			# ['APSK128', 'APSK16', 'APSK256', 'APSK32', 'APSK64', 'BFM', 'BPSK', 'CPFSK', 
			#'DSBAM', 'GFSK', 'OQPSK', 'PAM4', 'PSK8', 'QAM128', 'QAM16', 'QAM256', 'QAM32', 
			#'QAM64', 'QPSK', 'SSBAM'])


			#if mod.find('BPSK')<0:
			#	continue

			#if snr<18:
			#	continue
			#if str(sps)!=str(8.000):
			#	continue


			#if str(sps)!=str(2.000) and str(sps)!=str(4.000) and str(sps)!=str(8.000) and str(sps)!=str(16.000) and str(sps)!=str(32.000):
			#	continue
			#else:
			#	continue

			print ("it is ",sps)
			if(snr not in snrs):
				snrs.append(snr)

			if(mod not in mods):
				mods.append(mod)

			if (sps not in nsps):
				nsps.append(sps)



			values=np.array(ds.get(key))
			shape=values.shape
			#print(shape)
			total_len_a=shape[2]
			values=np.transpose(values,(2,1,0))
			#print("after tran ",values.shape)
			values=values[:,0:2,:]
			#print("after tran ",values.shape)
			# if dataset_name.find("../dataset1024_iqnoenergy_rici_Lstest_200sa.mat")<0:
			# 	#print("no it is rici")
			# 	split_factor_train=0.05
			# 	split_factor_valid=0.05
			# else:
			# 	#print("it is rici")
			# 	split_factor_train=0.25
			# 	split_factor_valid=0.25
			
			train_len=int(round(split_factor_train*total_len_a))
			valid_len=int(round(split_factor_valid*total_len_a))
			test_len=int(total_len_a-(train_len+valid_len))

			if train_len>total_len_a:
				print("Split factor cannot be higher than 1")
				exit()

			b=np.full((total_len_a,1),mod)
			c=np.full((total_len_a,4),[mod,snr,sps,nt])
			

			indices=np.arange(total_len_a)
			np.random.seed(10000)
			np.random.shuffle(indices)

			if X.size == 0:
				X=np.array(values)
				Y=np.array(b)
				snr_mod_pairs=np.array(c)

				X_train=np.array(values[indices[0:train_len]])
				Y_train=np.array(b[indices[0:train_len]])
				snr_mod_pairs_train=np.array(c[indices[0:train_len]])

				X_test=np.array(values[indices[train_len:train_len+test_len]])
				Y_test=np.array(b[indices[train_len:train_len+test_len]])
				snr_mod_pairs_test=np.array(c[indices[train_len:train_len+test_len]])

				X_valid=np.array(values[indices[train_len+test_len:total_len_a]])
				Y_valid=np.array(b[indices[train_len+test_len:total_len_a]])
				snr_mod_pairs_valid=np.array(c[indices[train_len+test_len:total_len_a]])
			else:
				train=False
				test_tf=True
				if (train==True):
					X_train=np.vstack((X_train,values[indices[0:train_len]]))
					Y_train=np.append(Y_train,b[indices[0:train_len]],axis=0)
					snr_mod_pairs_train=np.append(snr_mod_pairs_train, c[indices[0:train_len]],axis=0)

					
					X_valid=np.vstack((X_valid, values[indices[train_len+test_len:total_len_a]]))
					Y_valid=np.append(Y_valid,b[indices[train_len+test_len:total_len_a]],axis=0)
					snr_mod_pairs_valid=np.append(snr_mod_pairs_valid, c[indices[train_len+test_len:total_len_a]],axis=0)

				if (test_tf==True):
					X_test=np.vstack((X_test, values[indices[train_len:train_len+test_len]]))
					Y_test=np.append(Y_test,b[indices[train_len:train_len+test_len]],axis=0)
					snr_mod_pairs_test=np.append(snr_mod_pairs_test, c[indices[train_len:train_len+test_len]],axis=0)

		f.close()

	snrs.sort()
	mods.sort()
	nsps.sort()

	indices=np.arange(len(X_train))
	np.random.seed(100000)
	np.random.shuffle(indices)
	
	X_train=np.array(X_train[indices])
	Y_train=np.array(Y_train[indices])
	snr_mod_pairs_train=np.array(snr_mod_pairs_train[indices])
	#print(X_train)
	#print(Y_train)

	
	indices=np.arange(len(X_test))
	np.random.seed(100000)
	np.random.shuffle(indices)
	
	X_test=np.array(X_test[indices])
	Y_test=np.array(Y_test[indices])
	snr_mod_pairs_test=np.array(snr_mod_pairs_test[indices])

	indices=np.arange(len(X_valid))
	np.random.seed(100000)
	np.random.shuffle(indices)
	
	X_valid=np.array(X_valid[indices])
	Y_valid=np.array(Y_valid[indices])
	snr_mod_pairs_valid=np.array(snr_mod_pairs_valid[indices])


	
	print("SNR values are ",snrs)
	print("Modulation formats are ",mods)
	print("nsps are ",nsps)

	print("\n\nComplete datasets have shapes such:")
	print("Input dataset: ",X.shape)
	print("Output dataset: ", Y.shape)
	print("SNR Modulation pairs: ",snr_mod_pairs.shape)

	print("\n\nTrain datasets have shapes such:")
	print("Train Input datasets: ",X_train.shape)
	print("Train Output datasets: ", Y_train.shape)
	print("Train SNR Modulation pairs: ", snr_mod_pairs_train.shape)

	print("\n\nTest datasets have shapes such: ")
	print("Test Input datasets: ",X_test.shape)
	print("Test Output datasets: ",Y_test.shape)
	print("Test SNR Modualtion pairs: ",snr_mod_pairs_test.shape)

	print("\n\nValid datasets have shapes such: ")
	print("Valid Input datasets: ",X_valid.shape)
	print("Valid Output datasets: ",Y_valid.shape)
	print("Valid SNR Modualtion pairs: ",snr_mod_pairs_valid.shape)

	print("\n"*2,"*"*10,"Train/test dataset split - Done","*"*10,"\n"*2)
		
	


def to_one_hot(yy):
	#print (yy)
	yy=list(yy)
	#print(yy)
	yy1=np.zeros([len(yy),max(yy)+1])
	yy1[np.arange(len(yy)),yy]=1
	return yy1


def encode_labels():
	global Y_train,Y_test,Y_valid, snr_mod_pairs_test,snr_mod_pairs_train,mods
	print("\n"*2,"*"*10,"Label binary encoding - Start","*"*10,"\n"*2)
	print("Doing...")

	# onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
	# Y_train = onehot_encoder.fit_transform(Y_train[..., np.newaxis])
	# Y_test = onehot_encoder.fit_transform(Y_test[..., np.newaxis])
	# Y_valid = onehot_encoder.fit_transform(Y_valid[..., np.newaxis])
	Y_train=to_one_hot(map(lambda x:mods.index(x),Y_train))
	Y_test=to_one_hot(map(lambda x:mods.index(x),Y_test))
	Y_valid=to_one_hot(map(lambda x:mods.index(x),Y_valid))

	print("\n","*"*10,"Label binary encoding - Done","*"*10,"\n"*2)

def transform_input():
	global X_test, X_train,X_valid,Train,amp_train,phase_train,amp_test,phase_test, amp_valid,phase_valid
	print ("\n"*2,"*"*10,"Input dataset transformation-Start","*"*10,"\n"*2)
	print ("Doing...")
	train=False
	transform=False
	if (train==True and transform==True):
		print("we do transform")
		for sample in X_train:

			for i in range(0,sample.shape[1]):
				
				sig_amp=LA.norm([sample[0][i],sample[1][i]])
				sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
				sample[0][i]=sig_amp
				sample[1][i]=sig_phase


	if (transform==True):	
		for sample in X_test:
			
			for i in range(0,sample.shape[1]):
				sig_amp=LA.norm([sample[0][i],sample[1][i]])
				sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
				sample[0][i]=sig_amp
				sample[1][i]=sig_phase

	if (train==True and transform == True):
		print("we do transform") 

		for sample in X_valid:
			
			for i in range(0,sample.shape[1]):
				sig_amp=LA.norm([sample[0][i],sample[1][i]])
				sig_phase=np.arctan2(sample[1][i],sample[0][i])/np.pi
				sample[0][i]=sig_amp
				sample[1][i]=sig_phase

	print("\n\nInput datasets after transformation have shapes such:")
	print("Train Input datasets: ",X_train.shape)
	print("Test Input datasets: ",X_test.shape)
	print("Valid Input datasets: ",X_valid.shape)


	X_train=np.transpose(X_train,(0,2,1))
	X_test=np.transpose(X_test,(0,2,1))
	X_valid=np.transpose(X_valid,(0,2,1))

	print("after transpose")
	print("Train Input datasets: ",X_train.shape)
	print("Test Input datasets: ",X_test.shape)



	X_train=np.reshape(X_train,(-1,1,N_samples,2))
	X_test=np.reshape(X_test,(-1,1,N_samples,2))
	X_valid=np.reshape(X_valid,(-1,1,N_samples,2))


	#print(amp_train)
	#print(phase_train)

	print("after reshape ")
	print("Train Input datasets: ",X_train.shape)
	print("Test Input datasets: ",X_test.shape)
	print("Valid Input datasets: ",X_valid.shape)




	print("\n"*2,"*"*10,"Input dataset transformation-Done","*"*10,"\n"*2)



def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_examples(images, cls_true, cls_pred,extra,snr):
	global mods
	n_examples = images.shape[0]
	print ("Number of examples is ",n_examples)
	print ("images shape is ",images.shape)

	for i in range(0,n_examples):
		fig=plt.figure(figsize=(15,10))
		i_samples=images[i,0,:,0]
		q_samples=images[i,0,:,1]
		print ("i samples shape is ",i_samples.shape)

		plt.scatter(i_samples,q_samples)
		cls = cls_true[i]
		pred = cls_pred[i]
		mod_cls=mods[cls]
		mod_pred=mods[pred]

		plt.savefig("./images/"+str(mod_pred)+"_realmod_"+str(mod_cls)+str(extra)+str(snr)+"_ray.png")



def localizer(input_layer, constraints):
	global trainableSTN, trainableSTN_last
	if constraints is not None:
		num_params = constraints.num_free_params
	else:
		num_params = 6

	x=input_layer

	x=keras.layers.Conv2D(64,(1,3),padding='same',trainable=trainableSTN)(x)
	x=keras.layers.MaxPooling2D(pool_size=(1,2),trainable=trainableSTN)(x)
	#x=keras.layers.Conv2D(64,(1,3),padding='same',trainable=trainableStage1)(x)
	#x=keras.layers.MaxPooling2D(pool_size=(1,2))(x)

	x=keras.layers.Conv2D(32,(1,3),padding='same',trainable=trainableSTN)(x)
	#stage 1	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableSTN)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableSTN)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableSTN)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableSTN)(x)

	#stage 2	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableSTN)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableSTN)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableSTN)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableSTN, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableSTN)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableSTN)(x)


	#stage 3
	
	#x=keras.layers.Conv2D(32,(1,1),padding='same',trainable=trainableStage1)(x)	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN_last, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableSTN_last)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN_last, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableSTN_last, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableSTN_last)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableSTN_last, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableSTN_last)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableSTN_last)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableSTN_last, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableSTN_last)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableSTN_last)(x)

	#x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage1)(x)

	

	x=keras.layers.GlobalAveragePooling2D(trainable=trainableSTN_last)(x)


	graph = keras.layers.Dense(units=num_params,trainable=trainableSTN_last, activation='tanh', bias_initializer='random_normal', activity_regularizer='l2')(x)
	# hard code the affine transformation!
	theta = keras.layers.Dense(units=num_params,trainable=trainableSTN_last, bias_initializer='random_normal')(graph)
	def add(theta):        
		identity = tf.constant([[0.7, -0.7, 0.1, 0.3, 0.7, 0.2]], dtype=tf.float32)
		theta = theta + identity
		return theta

	theta = keras.layers.Lambda(add,trainable=trainableSTN_last)(theta)
	print(theta)
	return theta

def cnn_model_fn(input_layer, dense_units, num_classes):
	global trainableStage1, trainableStage2, trainableStage3, trainableStage4, trainableStage5
	global trainableStage6,trainableStageDense
	graph = input_layer

	x=graph

	x=keras.layers.Conv2D(64,(1,3),padding='same',trainable=trainableStage1)(x)
	x=keras.layers.MaxPooling2D(pool_size=(1,2),trainable=trainableStage1)(x)
	#x=keras.layers.Conv2D(64,(1,3),padding='same',trainable=trainableStage1)(x)
	#x=keras.layers.MaxPooling2D(pool_size=(1,2))(x)

	x=keras.layers.Conv2D(32,(1,3),padding='same',trainable=trainableStage1)(x)
	#stage 1	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage1, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage1)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage1, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage1, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage1)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage1, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage1)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage1)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage1, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage1)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage1)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage1)(x)

	#stage 2
	
	#x=keras.layers.Conv2D(32,(1,1),padding='same',trainable=trainableStage1)(x)	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage2, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage2)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage2, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage2, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage2)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage2, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage2)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage2)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage2, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage2)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage2)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage2)(x)

	#stage 3
	
	#x=keras.layers.Conv2D(32,(1,1),padding='same',trainable=trainableStage3)(x)	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage3, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage3)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage3, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage3, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage3)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage3, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage3)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage3)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage3, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage3)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage3)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage3)(x)

	#stage 4
	
	#x=keras.layers.Conv2D(32,(1,1),padding='same',trainable=trainableStage1)(x)	
	
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage4, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage4)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage4, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage4, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage4)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage4, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage4)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage4)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage4, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage4)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage4)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage4)(x)

	#stage 5
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage5, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage5)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage5, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage5, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage5)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage5, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage5)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage5)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage5, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage5)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage5)(x)

	x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage5)(x)


	#stage 6
	x_id=x

	x1=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage6, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x1=keras.layers.Activation('relu',trainable=trainableStage6)(x1)
	x1=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage6, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x1)

	x2=keras.layers.Conv2D(16,(1,1),padding='valid',trainable=trainableStage6, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x2=keras.layers.Activation('relu',trainable=trainableStage6)(x2)
	x2=keras.layers.Conv2D(16,(1,3),padding='same',trainable=trainableStage6, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x2)

	x=keras.layers.Concatenate(axis=3,trainable=trainableStage6)([x1,x2])
	x=keras.layers.Activation('relu',trainable=trainableStage6)(x)
	x=keras.layers.Conv2D(32,(1,1),padding='valid',trainable=trainableStage6, kernel_regularizer=l2(0.0001),kernel_initializer=keras.initializers.VarianceScaling(seed=0.0))(x)
	x=keras.layers.Activation('relu',trainable=trainableStage6)(x)
	x=x_id+x
	x=keras.layers.Activation('relu',trainable=trainableStage6)(x)

	#x=keras.layers.MaxPooling2D((1,2),trainable=trainableStage1)(x)

	#stage 7

	x=keras.layers.GlobalAveragePooling2D(trainable=trainableStage6)(x)



	
	#x=keras.layers.Flatten(trainable=trainableStage6)(x)
	#x=keras.layers.Dense(128,activation='selu',trainable=trainableStageDense)(x)
	#x=keras.layers.Dense(128,activation='selu',trainable=trainableStageDense)(x)

	#dense = keras.layers.Dense(num_classes)(x)

	#logits = keras.layers.Activation('softmax', name='y')(dense)
	logits=x

	return logits

def standard_cnn(**pm):
	X = keras.layers.Input(shape=pm['input_shape'], name='X')
	x = cnn_model_fn(X, pm['dense_units'], pm['num_classes'])

	dense = keras.layers.Dense(pm['num_classes'])(x)
	output = keras.layers.Activation('softmax', name='y')(dense)

	model = keras.Model(inputs=X, outputs=output)
	opt = keras.optimizers.Adam()
	model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

	return model

class SpatialTransformLayer(snt.AbstractModule):
	"""affine transform layer.
	Constructor requires output_shape, with optional
	constraints and boundary.
	"""

	def __init__(self, output_shape,constraints=None,name='stn_layer'):

		super().__init__(name=name)
		if constraints is None:
			constraints = snt.AffineWarpConstraints.no_constraints(num_dim=2)

		self._constraints = constraints
		self._output_shape = output_shape
		self.__name__ = name

	def _build(self, inputs):

		U, theta = inputs
		grid = snt.AffineGridWarper(source_shape=U.get_shape().as_list()[1:-1],output_shape=self._output_shape,constraints=self._constraints)(theta)
		V = tf.contrib.resampler.resampler(U, grid, name='resampler')

		return V

# define our keras model
def stn_model_fn(**pm):
	global N_samples, trainableStageDense, trainableGrid
	# input layer
	U = keras.layers.Input(shape=pm['input_shape'], name='X')

	# create localizationnetwork
	theta = localizer(U, pm['constraints'])
	
	grid_resampler = SpatialTransformLayer(output_shape=pm['output_shape'], constraints=pm['constraints'])
	# feed input U and parameters theta into the grid resampler
	V = keras.layers.Lambda(grid_resampler, output_shape=pm['output_shape'], name='V',trainable=trainableGrid)([U, theta])

	print("v shape is ",V.get_shape())
	#V=U+V
	
	
	# now input V to a standard cnn
	logits = cnn_model_fn(V, pm['dense_units'], pm['num_classes'])
	main_logits=cnn_model_fn(U,pm['dense_units'], pm['num_classes'])
	x=keras.layers.Concatenate()([logits,main_logits])

	x=keras.layers.Dense(128,'relu',trainable=trainableStageDense)(x)
	#x=keras.layers.Dense(256,'relu')(x)

	num_classes=pm['num_classes']

	dense = keras.layers.Dense(num_classes)(x)

	logits = keras.layers.Activation('softmax', name='y')(dense)

	
	model = keras.Model(inputs=U, outputs=logits)

	opt = keras.optimizers.Adam()
	model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

	return model


################################Main Program##########################################
#1. step - load dataset, split to train and test datasets, encode labels as binary vectors
parse = True
if parse:
	gen_data()
	np.save("X_train.npy",X_train)
	np.save("Y_train.npy",Y_train)
	np.save("X_test.npy",X_test)
	np.save("Y_test.npy",Y_test)
	np.save("X_valid.npy",X_valid)
	np.save("Y_valid.npy",Y_valid)
	np.save("mods.npy",mods)
	np.save("snrs.npy",snrs)
	np.save("nsps.npy",nsps)
	np.save("snr_mod_pairs_test.npy",snr_mod_pairs_test)
	np.save("snr_mod_pairs_train.npy",snr_mod_pairs_train)
	np.save("snr_mod_pairs_valid.npy",snr_mod_pairs_valid)
else:
	X_train = np.load("X_train.npy")
	Y_train = np.load("Y_train.npy")
	X_test  = np.load("X_test.npy")
	Y_test  = np.load("Y_test.npy")
	X_valid = np.load("X_valid.npy")
	Y_valid = np.load("Y_valid.npy")
	mods = np.load("mods.npy").tolist()
	snrs = np.load("snrs.npy").tolist()
	nsps = np.load("nsps.npy").tolist()
	snr_mod_pairs_test=np.load("snr_mod_pairs_test.npy")
	snr_mod_pairs_train=np.load("snr_mod_pairs_train.npy")
	snr_mod_pairs_valid=np.load("snr_mod_pairs_valid.npy")

#gen_data()
encode_labels()

#2. step - feature extraction (calculate amplitude and phase)
transform_input()

# create some boundaries on the affine warp
constraints = snt.AffineWarpConstraints([[None, None, None], [None, None, None]])

model_parameters = {
	'input_shape': (1,N_samples, 2),
	'batch_size': 256,
	'output_shape': (1,N_samples),
	'constraints': constraints,
	'dense_units': 128,
	'num_classes': len(mods)
}


K.clear_session()
tf.set_random_seed(0)

#normal_model = standard_cnn(**model_parameters)

#normal_model.summary()
#model = standard_cnn(**model_parameters)

model = stn_model_fn(**model_parameters)
model.summary()

cp_save_path_old = '/home/eperenda/multiple_signals/stn/pure_ray_weights_best_STN'
cp_save_path_channel='/home/eperenda/multiple_signals/stn/tf_channelBest_ray_weights_best_STN'
cp_save_path_channelAware='/home/eperenda/multiple_signals/stn/channelAware_ray_weights_best_STN'
cp_save_path='/home/eperenda/multiple_signals/stn/tf_channelAware_ray_weights_best_STN'

#cp_save_path=cp_save_path_channel
#print(cp_save_path)

checkpoint = ModelCheckpoint(cp_save_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

#model.load_weights(cp_save_path_channelAware)

train=False
filepath_normal = 'keras_normal_cnn'
filepath_stn='model_stn'
tb_log_dir = '/home/eperenda/multiple_signals/stn/l_log' 
if train:
	#normal_model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), epochs=80, batch_size=256)
	#normal_model.save(filepath_normal)
	model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid), epochs=80, batch_size=256, 
		callbacks=[	checkpoint,
					TensorBoard(log_dir=tb_log_dir)])
	#model.save(filepath_stn)
	#model=keras.models.load_model(cp_save_path)# ,custom_objects={'SpatialTransformLayer':SpatialTransformLayer})
	model.load_weights(cp_save_path)


else:
	#normal_model=keras.models.load_model(filepath_normal)
	#model=keras.models.load_model(filepath_stn ,custom_objects={'SpatialTransformLayer':SpatialTransformLayer})
	model.load_weights(cp_save_path)




print ("\n"*2,"*"*10,"Model Training - Done","*"*10,"\n"*2)

#exit()
# for mod in mods:
# 	for sps in nsps:
# 		for snr in snrs:
# 			print ("Predicting for SNR of ",snr," and sps of ",sps," and mod ", mod, " \n"*2)
# 			indices=[]
# 			i=0
# 			j=0
# 			for snr_mod in snr_mod_pairs_test:

# 				if (snr_mod[0]==mod and snr_mod[1] == str(snr) and  snr_mod[2]== str(sps)):
# 					indices.append(i)
				
# 				i=i+1
# 			print ("Total number test data is ", len(indices))
# 			if len(indices) == 0:
# 				print("continue")
# 				continue

# 			X_test_1=X_test[indices]
# 			Y_test_1=Y_test[indices]
# 			y_pred=model.predict(X_test_1)

		
# 			#y_pred = model.predict(X_test_1)
# 			cls_pred = np.argmax(y_pred, axis=1)
# 			cls_true = np.argmax(Y_test_1, axis=1)

# 			resampler = keras.Model(inputs=model.input, outputs=model.get_layer('V').output)
# 			bad_preds = np.argwhere(cls_pred == cls_true).flatten()
# 			if len(bad_preds)<11:
# 				continue;
# 			idic=[2,6,10]
# 			examples = bad_preds[idic]
# 			U_images = X_test_1[examples]
# 			V_images = resampler.predict(U_images)
# 			print ("shape V images is ", V_images.shape)

# 			plot_examples(U_images, cls_true[examples], cls_pred[examples],'U_img',snr)
			
# 			plot_examples(V_images, cls_true[examples], cls_pred[examples],'V_img',snr)
			


for sps in nsps:

	times_s={}
	acc={}

	for snr in snrs:
		print ("Predicting for SNR of ",snr," and sps of ",sps," \n"*2)
		indices=[]
		i=0
		j=0
		for snr_mod in snr_mod_pairs_test:

			if (snr_mod[1] == str(snr) and  snr_mod[2]== str(sps)):
				#print(snr_mod[1])
				#print(snr_mod[0])
				indices.append(i)
				
			i=i+1
		print ("Total number test data is ", len(indices))
		if len(indices) == 0:
			print("continue")
			continue

		X_test_1=X_test[indices]
		Y_test_1=Y_test[indices]
		#print(Y_test_1)
		
		y_pred=model.predict(X_test_1)

		
		#print("erma")
		y_el=[]
		y_pred_el=[]

		for i in range(1,len(y_pred)):
			y_pred_el.append(y_pred[i-1].argmax())

		for i in range(1,len(y_pred)):
			y_el.append(Y_test_1[i-1].argmax())

		#print(y_pred_el)
		#print("real")
		#print(y_el)
		#exit()
		cnf_matrix=confusion_matrix(y_el, y_pred_el)
		#print(cnf_matrix)

		cor=np.trace(cnf_matrix)
		#print "snr: ",snr,"cor:",cor
		cor_new=np.sum(np.diag(cnf_matrix))
		#print "snr: ",snr,"cor new:",cor_new
		sum_all=np.sum(cnf_matrix)
		#print "snr: ",snr,"sum:",sum_all
		acc[snr]=float(cor)/float(sum_all)

		#plt.matshow(cnf_matrix)
		#plt.colorbar()
		
		# Plot non-normalized confusion matrix
		# plt.figure()
		# plot_confusion_matrix(cnf_matrix,classes=mods, title='Confusion matrix, without normalization '+str(snr))

		# plt.gcf().subplots_adjust(bottom=0.15)
		# plt.savefig("./images/er_confmatsps32_lp"+str(snr)+".pdf")

		# Plot normalized confusion matrix
		#plt.figure(figsize = (12,10))
		#plot_confusion_matrix(cnf_matrix,classes=mods, normalize=True, title='Normalized confusion matrix '+str(snr))
		#plt.savefig("./images/spsoshea/er_confmatsps_"+str(sps)+"_snr_"+str(snr)+".png")

	#plt.show()
	print ("\noverall accuracy for sps ", sps, " is ",acc, " \n\n")

print (mods)

