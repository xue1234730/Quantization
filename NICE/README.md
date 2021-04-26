# NICE
https://arxiv.org/pdf/1810.00162.pdf<br>
## 介绍
NICE 方法有4个特点：
1. 训练过程中注入噪声<br>
2. 基于统计的参数初始化和激活截断<br>
3. 渐进训练方案<br>
4. 不量化网络第一层和最后一层<br>
前三点在[quantize.py](quantize.py)中实现。<br>

### Noise Injection
为了模拟量化的效果，文章在网络权值中加入了随机噪声。已知均匀的噪声分布可以很好地近似精细量化器的量化误差;也适用于相对粗糙的量化。此外，一定量的随机权值扰动似乎对训练算法的整体收敛有一个正则化效果。<br>
为了在噪声注入中达到类似 drpout 的效果，使用伯努利分布量化部分权重，并在其他权重上添加噪声。根据经验，选择 M\~Ber(0.05)，在反向传播过程中，权重 W 不使用量化值 Qw，而使用 W' = (1-M)Qw + M(W-e)，其中 e\~Uni(-range/2,range/2)，range为量化范围。<br>

### Gradual Quantization
为了提高 NICE 对更深层网络的可扩展性，从逐步向量化参数集添加子集开始，让网络其余部分适应这些变化。<br>
方法：讲网络划分为 N 个大小相等的层块 {B1,B2,...Bn}，在第 i 阶段，将噪声注入到 Bi 块的权重中，此时前面的块 {B1,...Bi-1} 被量化，后面的块 {Bi+1,...Bn} 保持全精度。
只使用一次渐进过程，即当第 N 个阶段结束时，在其余的训练时期使用 STE 对所有层量化和训练。<br>
在将噪声注入到层数 Bn 的块中完成训练后，继续对完全量化网络进行几个epoch的训练，直到收敛。对于预先训练的要进行量化的网络，文章发现最优的块大小是具有相应激活的单层。

### Clamping and Quantization
1. 权重量化<br>
将权重截断在 [-Cw,Cw] 之间，即： Wc = Clamp(W,-Cw,Cw) = max(-Cw, min(x,Cw))，其中参数 Cw 在每层都独自定义，初始化为：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/NICE/fig/WInit.PNG)<br>
截断权重 Wc 量化为 Bw 位：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/NICE/fig/WeightsClamp.PNG)<br>
2. 激活量化<br>
ReLU 被 ClampReLU 代替，将激活截断： Ac = Clamp(a,0,Ca)，a 为每层的线性输出，Ca 为截断范围，初始化为：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/NICE/fig/AInit.PNG)<br>
将截断后的激活 Ac 量化为 Ba 位：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/NICE/fig/AClamp.PNG)<br>
由于 round 函数不可微，所以使用 STE 更新梯度。<br>
3. 偏置量化<br>
偏置的范围取决于每层权重和激活的范围，对于每一层， Cb 都初始化为：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/NICE/fig/bInit.PNG)<br>
Bb 为偏置的量化位宽，以与权重相同的方式被截断和量化

## 使用
使用 Resnet-Cifar10，W5A5，将 Cifar10 数据集存放在目录 ./results 中或使用 --datapath "datapath" 设置存放路径。<br>
```
python3 main.py --model resnet --depth 18 --bitwidth 5 --act-bitwidth 5 --step 21 --gpus 0 --epochs 120 -b 256 --dataset cifar10 --start-from-zero --resume none --learning_rate=0.01 --quant_start_stage=0 --quant_epoch_step=3 --schedule 300
```
>* Prec@1 82.06

## Additional Knowledge
### Uniform Additive Noise
噪声<br>
实际的图像经常会受到一些随机的影响而退化，可以理解为原图受到干扰和污染。通常就就把这个退化成为噪声（noise）。在我们采集、传输或者处理的过程中都有可能产生噪声，因此噪声的出现是多方面的原因。<br>
在信号处理中我们用概率特征来描述噪声，即噪声出现的概率呈现为什么样子就为什么样子的噪声。就比如本篇的主角高斯噪声，其在信号中出现、分布的概率成高斯分布（正态分布）。也许你会疑问噪声概率分布应是无规律的，现实中也确实噪声严格上是不规律的分布，但是为了简化计算，人们常将噪声理想化，这种理想型的概率分布模型来描述的噪声称为白噪声。白噪声在信号中是典型的，其特点是具有常量功率谱。而高斯噪声属于白噪声的一个特例。<br></br>
加性零均值高斯噪声<br>
1. 加性<br>
加性噪声的表达式: 
f(x,y)=g(x,y)+v(x,y)<br>
g为原图像信号，v为噪声信号，两者关系是独立的，这表示原图信号如何变化与噪声无光，噪声不依赖于图像信号，这种独立与信号的噪声便称为加性噪声（additive noise）。<br>
2. 零均值<br>
均值为零，即噪声为高斯分布，高斯分公式中的均值为0，即高斯分布的对称轴为0。
