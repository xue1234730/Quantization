# Scalable Methods for 8-bit Training of Neural Networks
https://arxiv.org/pdf/1805.11046.pdf

## 介绍
对训练完毕的网络模型进行定点量化可以提升模型在推理过程中的计算效率，但是对于如何确定最优的量化比特数以及量化方案尚无定论。文章首先通过理论分析指出，在网络训练过程中，除部分特定的操作外，大部分操作对于模型权重精度的下降并不敏感。基于这一结论，本文提出对模型权重、各层特征图以及梯度信号进行量化，并且维护了两个量化精度不同的梯度信号，在不损失精度的情况下最大程度地提升计算效率。同时，由于batch normalization层对于量化精度要求更高，本文提出了Range BN层以提升对量化误差的容忍度。

### Range Batch-Normalization
对于 nxd 维，输入为 x = (x(1),x(2),...,x(d)) 的层，传统 batch norm 将每一维归一化为<br>
![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/1.PNG)<br>
本文中，作者使用 Range BN 代替传统BN操作。假设输入服从高斯分布，则输入的范围与标准差大小相关。Range BN 使用尺度调整C(n)乘以输入值的范围来接近标准偏差σ。即：<br>
![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/2.PNG)<br>
其中，![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/Cn.PNG) ，range(·) = max(·)-min(·)

### Quantized Back-Propagation
1.Quantization methods：文章使用 GEMMLOWP 量化结构，其中激活的最大值和最小值是由 Range BN　计算的。
2.Gradients Bifurcation：在反向传播算法中，从最后一层开始递归计算梯度,每一层都需要导出两组梯度来执行递归更新。<br>
损失函数L的梯度:<br>
![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/4.PNG)<br>
每一层activation的梯度:<br>
![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/5.PNG)<br>
每层权重梯度更新:<br>
![image](https://github.com/xue1234730/Quantization/blob/main/convnet/fig/6.PNG)<br>
3.Straight-Through Estimator:文章使用了 STE 方法来通过离散变量近似微分。这是处理离散变量的精确导数几乎处处为零这一问题的最简单和硬件友好的方法。

### 使用

    
    python main.py --model resnet --model-config "{'depth': 18, 'quantize':True}" --save resnet18_8bit -b 64
    

Dependencies:

- [pytorch](<http://www.pytorch.org>)
- [torchvision](<https://github.com/pytorch/vision>) to load the datasets, perform image transforms
- [pandas](<http://pandas.pydata.org/>) for logging to csv
- [bokeh](<http://bokeh.pydata.org>) for training visualization


## Data
- Configure your dataset path with ``datasets-dir`` argument
- To get the ILSVRC data, you should register on their site for access: <http://www.image-net.org/>


## Model configuration

Network model is defined by writing a <modelname>.py file in <code>models</code> folder, and selecting it using the <code>model</code> flag. Model function must be registered in <code>models/\_\_init\_\_.py</code>
The model function must return a trainable network. It can also specify additional training options such optimization regime (either a dictionary or a function), and input transform modifications.

e.g for a model definition:

```python
class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.model = nn.Sequential(...)

        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
                'weight_decay': 5e-4, 'momentum': 0.9},
            {'epoch': 15, 'lr': 1e-3, 'weight_decay': 0}
        ]

        self.data_regime = [
            {'epoch': 0, 'input_size': 128, 'batch_size': 256},
            {'epoch': 15, 'input_size': 224, 'batch_size': 64}
        ]
    def forward(self, inputs):
        return self.model(inputs)
        
 def model(**kwargs):
        return Model()
```


# Citation

```
@inproceedings{hoffer2018fix,
  title={Fix your classifier: the marginal value of training the last weight layer},
  author={Elad Hoffer and Itay Hubara and Daniel Soudry},
  booktitle={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=S1Dh8Tg0-},
}
```
