# Quantization

## 为什么量化？

通常我们用各种现有成熟的深度学习框架，如TensorFlow、Pytorch或mxnet等搭建网络模型时，参数和中间运行数据都是默认为float32类型的，然而在移动嵌入式设备上，由于内存大小、功耗等限制，直接部署原始的浮点网络是不切实际的，所以就需要对原始浮点模型进行压缩，减少参数所需的内存消耗。通常的方法有剪枝、知识蒸馏、量化、矩阵分解，其中量化方法是使用最为普遍的，因为将32bit的参数量化为8bit，需要消耗的内存直接就缩减到了原始的1/4，而其他方法针对不同任务，不同模型，在满足性能要求的情况下实际能节省多少资源都是无法保证的。

## 什么是量化?

量化有若干相似的术语。低精度（Low precision）可能是最通用的概念。常规精度一般使用 FP32（32位浮点，单精度）存储模型权重；低精度则表示 FP16（半精度浮点），INT8（8位的定点整数）等等数值格式。不过目前低精度往往指代 INT8。

优点1：加快运算速度。当把FP32转变为INT8表示时，在不考虑系统有浮点加速模块时，定点运算要比浮点运算快。

优点2：减少存储空间。若将FP32转变为INT8位表示时，存储空间减小到了1/4大小。

缺点：在用低带宽数值近似表示时，会造成一些精度损失。值得高兴的是，神经网络的参数大多是冗余的（或者说是对噪声的容忍度），所以当在近似变换时对精度的影响不是特别大。

## 如何量化？

神经网络量化的实现需要在常见操作（卷积，矩阵乘法，激活函数，池化，拼接等）前后分别加上quantize和dequantize操作，quantize操作将input从浮点数转换成8 位整数，dequantize操作把output从8 位整数转回浮点数。

FP32的tensor量化为INT8需要4步：缩放；取整；偏移；反量化。

![image](https://github.com/xue1234730/Quantization/blob/main/quantization.jpg)

## 4.量化实现

1.Data-Free Quantization Through Weight Equalization and Bias Correction

代码：DFQ

2.Post training 4-bit quantization of convolutional networks for rapid-deployment

代码：PTQ

3.NICE: Noise Injection and Clamping Estimation for Neural Network Quantization

代码： NICE

4.PACT: Parameterized Cliping Activation for Quantized Neural Networks

代码： PACT

5.Scalable Methods for 8-bit Training of Neural Networks

代码： convnet
