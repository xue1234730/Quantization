# Deep Neural Network Quantization via Layer-Wise Optimization Using Limited Training Data
https://csyhhu.github.io/data/L-DNQ.pdf

## 简介
现有的方法大多依赖于有监督的训练过程来获得满意的性能，获取大量的标记训练数据，这可能不适合实际部署。这篇文章提出了一种新的分层量化深度神经网络方法，它只需要有限的训练数据(原始数据集的1%)。L-DNQ的目标是最小化原始网络和量化网络之间的最终层输出误差。

1)对每一层参数进行量化，层输出与原始全精度参数相似：为每一层制定了一个二次优化问题，以最小化量化层输出和全精度层输出之间的误差。

2)量化可以在一个封闭的解中得到，无需基于梯度的训练：量化权值作为离散约束，导致离散优化问题。为解决这一问题，采用可选方向乘数方法(ADMM)方法，将连续变量和离散约束解耦，分别更新它们。

3)量化后的整体预测性能有理论保证。

4)整个过程只消耗训练数据的一小部分实例(实验中训练数据集的1%)。


## 使用


### 要求
PyTorch > 0.4.0

TensorFlow > 1.3.0

### 确定数据集
在代码 `utils/dataset.py` 的第 39 行（CIFAR10）和 70 行（ImageNet）更改数据集路径。

### Pre-trained Model
L-DNQ 需要 pre-trained model。
- CIFAR10 Dataset

运行 `train_base_model.py` 代码， 得到一个包含 pre-trained 的文件夹`ResNet20`:
    
    cd L-DNQ
    python train_base_model.py
    python train_base_model.py --resume --lr=0.01 # If needed

- ImageNet Dataset

从 torchvision model zoo 下载 pre-trained model 到文件夹 `ResNet18-ImageNet`。

    cd L-DNQ
    mkdir ResNet18-ImageNet
    wget https://download.pytorch.org/models/resnet18-5c106cde.pth
    
### 量化
在 CIFAR10/ImageNet 数据集量化 ResNet 网络。  To reproduce ResNet20 quantization using CIFAR10:

    python main.py
    
To reproduce ResNet18 quantization using ImageNet:

    python main.py --model_name=ResNet18-ImageNet --dataset_name=ImageNet 
    
To reproduce other experiments, please change the network structure accordingly in the code.

### 改变数据集大小
此代码中仅使用原始数据集的1%，如果需要改变的话，将 `main.py` 中

`get_dataloader(dataset_name, 'limited', batch_size = 128, ratio=0.01)` 改变为

`get_dataloader(dataset_name, 'limited', batch_size = 128, ratio=Whatever you want, resample=True)`

Set `resample` to `False` after new selected dataset is generated. 

如果想使用整个数据集： `get_dataloader(dataset_name, 'train', batch_size = 128)`

### 改变量化位数

改变参数 `--kbits` （3,5,7,9,11）. For example, 5 means the quantization bits are: $0, \pm \alpha, \pm 2*\alpha, \pm 4*\alpha$, totally 5 bits.