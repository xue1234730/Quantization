# Trained Ternary Quantization

https://arxiv.org/pdf/1612.01064v1.pdf

## 简介

TTQ将浮点权重量化为三值权重。在以前的三值网络 [TWN (Ternary weight networks)](https://arxiv.org/abs/1605.04711v1)中，将权重量化为：<br>
![image](https://user-images.githubusercontent.com/58316204/120437891-3a52cf80-c3b3-11eb-89a0-8ce2ca237477.png)<br>
阈值的计算公式为：<br>
![image](https://user-images.githubusercontent.com/58316204/120437960-522a5380-c3b3-11eb-9af5-ef9fbc1d1501.png)<br>
本文中，作者将权重量化为：<br>
![image](https://user-images.githubusercontent.com/58316204/120438174-9ddcfd00-c3b3-11eb-9497-52cfbd8f1c9d.png)<br>
其中，尺度因子不是某层权重元素绝对值的均值，而是设定为参数 Wl，通过反向传播学习获得。计算出权重的梯度为：<br>
![image](https://user-images.githubusercontent.com/58316204/120438234-b1886380-c3b3-11eb-9fdd-0cfcc86958b5.png)<br>
其中<br>
![image](https://user-images.githubusercontent.com/58316204/120438264-bb11cb80-c3b3-11eb-96e2-14643109c43e.png)<br>
与TWN不同的是，它的阈值设置为<br>
![image](https://user-images.githubusercontent.com/58316204/120438446-f57b6880-c3b3-11eb-81be-c58aca69deed.png)<br>

## 使用

### 训练全精度模型

```
python train_base_model.py -m ResNet20 -d CIFAR10
```
By using `train_base_model.py`, it will generate a folder named as `../Results/model-dataset` (such as 
`../Results/ResNet20-CIFAR10`) in the upper level folder, with the pretrain network named as `model-dataset-pretrained.pth`

### TTQ

```
python TTQ.py -m ResNet20 -d CIFAR10 -tf 0.05
```

### Results Visualization
Specify the training log path you want to see in `visualize_training_log.py`
```
python visualize_training_log.py
```

## Experiment
All layers are ternarized:

| Model    | Dataset |  Quantized Acc | FP Acc |
| :-------:|:-------:|:-------------:|:--------:|
| ResNet20 | CIFAR10 | 90.3| 91.5|


## Customization
Build your own model with `nn.Conv2d`/`nn.Linear` replaced by `TTQ_CNN`/`TTQ_Linear`.
