# MACS-Net-Overlapping-chromosome-segmentation-based-on-multi-scale-U-shaped-network
MACS Net: Overlapping chromosome segmentation based on multi-scale U-shaped network

## 1.数据集的制作
采用Kaggle网站中由Jeanpat制作的重叠染色体数据集，其链接为  
https://github.com/jeanpat/DeepFISH/blob/master/dataset/LowRes_13434_overlapping_pairs.h5  
再利用xx.py对数据集进行预处理，生成128×128尺寸的数据集  

## 2.训练  
打开 `trainModel.py`   
修改数据集路径  
```
#—————————————— Load data ————————————————————————————#  
xdata = np.load('./data/xdata_128x128.npy')  
labels = np.load('./data/ydata_128x128_0123_onehot.npy')  
```  
修改数据集划分文件的路径  
```
#—————————————— Load the class of data ——————————————————————#  
number = 1      # 选择五折交叉验证中哪一折
a=np.load('./data_cls_new/'+str(number)+'/data_cls_4.npy')            
a=a.tolist()
b=np.load('./data_cls_new/'+str(number)+'/data_cls_1.npy')                 
b=b.tolist()
```
修改预训练模型的路径  
```
#—————————————— Pretrained model ———————————————————————#  
Name = './h5/MACSNet_1.h5'                                      
#Name = './h5/CENet_1.h5'
#Name = './h5/UNet_1.h5'
```

修改存储模型的路径  
```
#—————————————— Temporary model ————————————————————————#  
Name_tem = './MACSNet_'+str(number)+'.h5'  
```

修改训练的次数以及迭代轮数
```
#—————————————— Fit model ————————————————————————#  
num_epoch = 1                                                                                                                                     
for i in range(num_epoch):
    print('epoch:', i)
    # Fit
    check_point = ModelCheckpoint(Name_tem, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)  
    callback = EarlyStopping(monitor="val_loss", patience=30, verbose=0, mode='min')   #修改早停阈值
    
    history = model.fit(x, y, epochs=300, validation_split=0.2, batch_size=32, callbacks=[check_point, callback]) #修改迭代次数
```

## 3.预测
打开`predict.py`  
修改权重路径  
```
#—————————————— Pretrained model ———————————————————————#  
Name = './h5/MACSNet_1.h5'                                      
#Name = './h5/CENet_1.h5'
#Name = './h5/UNet_1.h5'  
model.load_weights(Name)   
```

预测单张图片中重叠部分的IoU  
```
#—————————————— predict single img ———————————————————————#  
ix = 250    
img = x_test[ix,:,:,0].reshape(1,128,128,1)
label = y_test[ix,:,:,3]

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
testIOU = utilities.IOU(img_pred, y_test[ix,:,:,:].reshape(1,128,128,4))  # 计算重叠部分的IoU
print('Testing IOU: ' + str(testIOU))
```
计算测试集中重叠部分的IoU  
```
#—————————————— predict iou ——————————————————————————#  
y_pred_test = model.predict(x_test).round()
testIOU = utilities.IOU(y_pred_test, y_test)
print('Testing IOU: ' + str(testIOU))
```

计算测试集中分割出的独立染色体的IoU  
```
#—————————————— predict chrom iou ———————————————————————#  
y_pred_test = model.predict(x_test).round()
testIOU = utilities.IOU_One(y_pred_test, y_test)
print('Testing Chrom IOU: ' + str(testIOU))
```
计算测试集中分割出的独立染色体的Acc  
```
#—————————————— predict Accuracy ———————————————————————#  
y_pred_test = model.predict(x_test).round()
testIOU = utilities.global_chrom_Accuracy(y_pred_test, y_test)
print('Testing Chrom Acc: ' + str(testIOU))
```
