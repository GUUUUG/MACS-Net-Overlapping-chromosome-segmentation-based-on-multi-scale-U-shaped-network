import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                                                  ###

#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import utilities
import os


from MACS_Net import MACS_Net
from MACS_Net_1 import MACS_Net_1
from CENet_ceshi import CENet_ceshi
from MACS_Net_3layers import MACS_Net_3layers
#from MACS_Net_3layers_TEST import MACS_Net_3layers_TEST
from MACS_Net_4layers import MACS_Net_4layers
#from UNet import UNet
from ARMSNet_duibi.UNet_R_4l import UNet_R_4l
from ARMSNet_duibi.UNet_AR_4l import UNet_AR_4l
from ARMSNet_duibi.UNet_PR_4l import UNet_PR_4l
from ARMSNet_duibi.UNet_DAR_4l import UNet_DAR_4l
from ARMSNet_duibi.UNet_4l import UNet_4l

from model import Deeplabv3





from keras.models import Model
import tensorflow as tf
from Loss_accuracy import LossHistory
import math
from keras.optimizers import Adam
import random
#import cv2
from keras.layers import Conv2D
#from keras.callbacks import Callback
#from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import Callback

#import warnings
from keras import backend as K
from test import binary_crossentropy

from keras.utils import multi_gpu_model
from time import *
# Load data

xdata = np.load('/home/guangjie.wang/new/data/xdata_128x128.npy')   ############################
labels = np.load('/home/guangjie.wang/new/data/ydata_128x128_0123_onehot.npy')

number = 1
a=np.load('/home/guangjie.wang/new/data_cls_new/'+str(number)+'/data_cls_4.npy')                  ################ 4
a=a.tolist()
b=np.load('/home/guangjie.wang/new/data_cls_new/'+str(number)+'/data_cls_1.npy')                 ################  1
b=b.tolist()


x = xdata[a]
y = labels[a]
x_test = xdata[b]
y_test = labels[b]

#weight = []
#for i in range(4):
#    posative = np.sum(labels[:,:,:,i]) 
#    negative = 220102656-posative
#    inter = round(float(negative)/float(posative),2)
#    mi = math.log(inter,1000)
#    if mi < 0:
#        mi = (1+np.abs(mi))**-1
#        weight.append(mi)
#    else:
#        weight.append(1+mi)
    
#ix = []
#for i in range(len(y)):
#    #print(i)
#    value = y[i,:,:,3].sum()
#    if value == 0:
#        ix.append(i)
#print(len(ix))        
#length = np.arange(len(y)).tolist()
#z = []
#for m in length:
#	if m not in ix:
#		z.append(m)
#x = x[z]
#y = y[z]

#ix = []
#for i in range(len(y_test)):
#    #print(i)
#    value = y_test[i,:,:,3].sum()
#    if value == 0:
#        ix.append(i)       
#length = np.arange(len(y_test)).tolist()
#z = []
#for m in length:
 # if m not in ix:
#		z.append(m)
#x_test = x_test[z]
#y_test = y_test[z]

#x_f = np.fliplr(x)
#y_f = np.fliplr(y)
#x_test_f = np.fliplr(x_test)
#y_test_f = np.fliplr(y_test)
#
#x = np.stack((x,x_f),axis=0).reshape(-1,128,128,1)
#y = np.stack((y,y_f),axis=0).reshape(-1,128,128,4)

##############################################################################################################################################################################
##############################################################################################################################################################################
#Name = './MACSNet_4layers_zidaiweighted_original.h5'                                        ################ model 2
#Name = '/home/guangjie.wang/new/MACSNet_duibi/CENet_1.h5'
Name = '/home/guangjie.wang/new/ARMSNet_duibi/duibi_h5/UNet_4l_1.h5'
Name2 = '/home/guangjie.wang/new/ARMSNet_duibi/duibi_h5/UNet_PR_'+str(number)+'.h5'                       ################  MODEL


#model = CENet_ceshi(input_shape=(128,128,1))
#model = MACS_Net_1(input_shape=(128,128,1))
#model = Deeplabv3(input_shape=(128,128,1), classes=4)  
#model = UNet(input_shape=(128,128,1))

#model = UNet_R_4l(input_shape=(128,128,1))
#model = UNet_AR_4l(input_shape=(128,128,1))
#model = UNet_PR_4l(input_shape=(128,128,1))
#model = UNet_DAR_4l(input_shape=(128,128,1))
model = UNet_4l(input_shape=(128,128,1))

#model = MACS_Net(input_shape=(128,128,1))
#model = MACS_Net_4layers(input_shape=(128,128,1))
#model = MACS_Net_3layers(input_shape=(128,128,1))
#model = MACS_Net_3layers_TEST(input_shape=(128,128,1))


#model = multi_gpu_model(model, gpus=2)                          
model.compile(loss='binary_crossentropy', optimizer='adam')   ##loss='binary_crossentropy' , optimizer='adam'   optimizer=Adam(lr=1e-4)  default  lr=1e-3, beta_1=0.9, beta_2=0.999
model.load_weights(Name)                                                           ###############################################  3


###############################################  predict img
ix = 119
img = x_test[ix,:,:,0].reshape(1,128,128,1)
label = y_test[ix,:,:,3]
#print(np.sum(label))
img_pred = model.predict(img).round()
plt.xticks(())
plt.yticks(())
plt.imshow(x_test[ix,:,:,0])
plt.savefig('./img.png')
plt.imshow(label)
plt.savefig('./label.png')
plt.show()
plt.imshow(img_pred[0,:,:,3])
plt.savefig('./pred.png')
plt.show()

testIOU = utilities.IOU(img_pred, y_test[ix,:,:,:].reshape(1,128,128,4))
print('Testing IOU: ' + str(testIOU))
##############################################                  predict iou
#y_pred_test = model.predict(x_test).round()
#testIOU = utilities.IOU(y_pred_test, y_test)
#print('Testing IOU: ' + str(testIOU))
################################################                  predict chrom iou
#y_pred_test = model.predict(x_test).round()
#testIOU = utilities.IOU_One(y_pred_test, y_test)
#print('Testing Chrom IOU: ' + str(testIOU))
###############################################                  predict Accuracy
#y_pred_test = model.predict(x_test).round()
#testIOU = utilities.global_chrom_Accuracy(y_pred_test, y_test)
#print('Testing Chrom Acc: ' + str(testIOU))
###############################################                  predict iou_set
#y_pred_test = model.predict(x_test).round()
#testIOU = utilities.IOU_set(y_pred_test, y_test)
#np.save('/home/guangjie.wang/new/IOU_SET/IOU_test_UNet_'+str(number),testIOU)

#for layer in model.layers[:-3]:
#    layer.trainable = False
#for layer in model.layers[-3:]:
#    layer.trainable = True
    



# Specify the number of epochs to run
#num_epoch = 1                                                                                           ############################################### 4                                                 
#for i in range(num_epoch):
#    print('epoch:', i)
#    # Fit
#    #history = LossHistory()
#    check_point = ModelCheckpoint(Name2, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)  #
#    callback = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode='min')
#    
#    history = model.fit(x, y, epochs=100, validation_split=0.2, batch_size=32, callbacks=[check_point, callback])       ##########################         [check_point, callback]
#                                       
#    # Calculate mIOU
#    model.load_weights(Name2) 
#    y_pred_train = model.predict(x).round()
#    trainIOU = utilities.IOU(y_pred_train, y)
#    print('value: ',np.sum(y_pred_train[0,:,:,3]))
#    print('value: ',np.sum(y[0,:,:,3]))
#    print('Training IOU: ' + str(trainIOU))    
#    y_pred_test = model.predict(x_test).round()
#    testIOU = utilities.IOU(y_pred_test, y_test)
#    print('Testing Overlap IOU: ' + str(testIOU))
##    
##    y_pred_test = model.predict(x_test).round()
##    testIOU = utilities.IOU_One(y_pred_test, y_test)
##    print('Testing Chrom IOU: ' + str(testIOU))
##    
##    y_pred_test = model.predict(x_test).round()
##    testIOU = utilities.global_chrom_Accuracy(y_pred_test, y_test)
##    print('Testing Chrom Acc: ' + str(testIOU))
#
#    
#    
#    
#    #fig = plt.figure()
#    #plt.plot(history.history['loss'],label='training loss')
#    #plt.plot(history.history['val_loss'],label='val loss')
#    #plt.title('model loss')
#    #plt.ylabel('loss')
#    #plt.xlabel('epoch')
#    #plt.legend(loc='upper right')
#    #fig.savefig('/home/guangjie.wang/new/pic/6_'+str(i)+'.png')      ####this is My_loss
#    
#    #y_pred_train = model.predict(x).round()
#    #trainIOU = utilities.IOU_set(y_pred_train, y)
#    #np.save('/home/guangjie.wang/new/IOU_SET/IOU_train_better_'+str(i),trainIOU)
#    #y_pred_test = model.predict(x_test).round()
#    #testIOU = utilities.IOU_set(y_pred_test, y_test)
#    #np.save('/home/guangjie.wang/new/IOU_SET/IOU_test_better_'+str(i),testIOU)

