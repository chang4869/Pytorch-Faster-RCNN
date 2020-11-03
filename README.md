# Impelemnt of Faster-RCNN
## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python >=3.6 and Pytorch >=1.6**

1. Install [Pytorch](https://pytorch.org/)

2. The data set should be in VOC format，the data directory should look like this:
   |---your dataset name
       |---Annotations
       |---JPEGImages


## Train the model
<font face="Times New Roman" size=4>

   ```
   sh train.sh
   ```
</font>

## Inference the model
<font face="Times New Roman" size=4>

   ```
   python demo.py
   ```
</font>
