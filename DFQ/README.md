# DFQ
关于 Data Free Quantization 的 Pytorch 实现
## 介绍
该论文提出了一种不需要额外数据来 finetune 恢复精度的离线 8bit 量化方法，它利用了 relu 函数的尺寸等价缩放的特性来调整不同channel的权重范围，并且还能纠正量化过程中引入的偏差。

### 问题
1.模型预训练完之后某些层的权重参数不同通道之间的数据方差很大如下图所示，利用常见的 per-layer 量化策略(即整个层的参数作为一个 tensor 进行量化)，则会使得值较小的通道直接全部被置为 0，导致精度的下降，per-channel 的方法可以解决这个问题，但是在硬件实现上因为要针对每一个通道都有自己独立的缩放系数和偏移值考虑，会导致很多额外的开销，得不偿失；

2.FP32->INT8 量化过程中带来的 noise 是有偏误差，会导致不同模型不同程度的性能下降，目前的方法基本依赖于 finetune；

解决办法: Data-Free Quantization

算法流程：
Cross-layer equalization --> Bias absorption --> Quantization --> Bias correction

### Cross-layer equalization
目前最常用的激活函数是 ReLU，他满足如下等式：f(sx) = sf(x)，并且该等式可以适用于一切分段线性函数。量化需要对参数 Tensor 进行缩放，如果不同通道的缩放系数差异很大的话就会导致巨大的量化误差，所以可以利用 ReLU 函数的缩放特性来进行上下层之间不同通道缩放系数的调整：若 Layer 1 的 weight 在输出通道的 range 方差很大，那么可以给 range 大的层乘以小的系数，range 小的层乘以大的系数，使得各个通道的 range 尽量一致，等于一个给定的范围，这样量化的时候整个 tensor 量化的误差就会最小，同时为了保证数值内部的等价，Layer 1 乘完的系数在 Layer 2 要除回，具体体现在 Layer 1 weight 的第 i 个输出通道乘了系数 s，对应 Layer 2 weight 的第 i 个输入通道就要除以系数 s。

论文中证明并讲述了每个 layer 的固定 range 的取法，具体最优化证明可以看论文的 appendix A：Layer 1 权重的输出通道和 Layer 2 的输入通道一一对应，令 ri 为第 i 个通道的权重范围，即 [-ri,ri] ，这里采用对称量化的方法比较好说明，假设第 i 个通道的 Layer 1 和 Layer 2 权重的范围分别是 r1 和 r2，那么如何取 si 可以使得两个 Layer 的第 i 层量化比较均衡，最好的方法就是令最后两个的范围 r = sqrt(r1*r2) ，那么 s = r/r1 = sqrt(r2/r1).

这里讲的是两层之间的 range 均衡，在真正的实验中是每相邻两层均衡，然后不断迭代着算下去的，可以设置一个量化误差阈值，如果把整个网络迭代完一遍之后的权重量化误差在一个预先设定的范围之内就可以停止迭代，或者设定一个最大迭代次数，达到最大次数即停止。

这部分在 dfq.py 中实现。

### Bias absorption
上面考虑的是如何使得每层 weight 的量化误差尽可能小，但在实际量化过程中，影响比较大的还有 activation 的量化效果。尤其在本层 weight 的某通道缩放系数大于1的时候，activation 的该通道数据范围会增大，这有可能会导致 activation 各通道之间的数据范围差异变大，从而导致较大的量化误差。解决办法是采用 ReLU 函数的又一等价变换： r(Wx+b-c) = r(Wx+b)-c ，这里的 r() 是 ReLU 函数，c 是一个正数，可以用裕量来解释，即原始的 Wx+b>c，就满足上述等式。

这部分在 dfq.py 中实现。

### Quantization
这部分在 utils.quantize.py 中实现.

UniformQuantize 按照缩放、取整、偏移将权重量化为 [qmin,qmax] 即 [-128,127] 或 [0,255] 之间的整数。

QuantConv2d 和 QuantLinear 量化输入，权重和激活值。

QuatNConv2d 和 QuantNLinear 只量化输入。

### Bias correction
本节讲述的是如何校正量化(四舍五入的不可逆性)带来的偏差。之前的研究认为量化带来的误差是无偏的，不影响输出的均值，但是本文发现权重的量化误差会导致输出的有偏误差，从而改变下一层输入数据的分布，在其他量化方法中方法中，可以直接计算 activation 量化前后的误差，加入 loss 中作为一个正则项，但是本文基于 Data-Free 的立场，决定继续采用网络中的已有信息(其实就是每层的权重和BN的各个参数信息)，所以需要从数学形式上进行推导。


## 代码运行

run the equalized model by:
```
python main_cls.py --quantize --relu --equalize
```

run the equalized and bias-corrected model by:
```
python main_cls.py --quantize --relu --equalize --correction
```

## 实验结果
On detection task  
Tested with [MobileNetV2 SSD-Lite model](https://github.com/qfgaohao/pytorch-ssd)

<table>
</th><th>Pascal VOC 2007 test set</th></tr>
<tr><td>

model      | precision
-----------|------    
Original   | 68.70 
DFQ        | 65.39 

</td></tr> </table>


