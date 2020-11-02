# MACS-Net-Overlapping-chromosome-segmentation-based-on-multi-scale-U-shaped-network
MACS Net: Overlapping chromosome segmentation based on multi-scale U-shaped network

## 1.数据集的制作
采用Kaggle网站中由Jeanpat制作的重叠染色体数据集，其链接为  
https://github.com/jeanpat/DeepFISH/blob/master/dataset/LowRes_13434_overlapping_pairs.h5  
再利用xx.py对数据集进行预处理，生成128×128尺寸的数据集  

## 2.训练  
打开 `trainModel.py`  
修改数据集路径  
—————————————— Load data ————————————————————————————  
```
xdata = np.load('./data/xdata_128x128.npy')  
```
```
labels = np.load('./data/ydata_128x128_0123_onehot.npy')  
```
