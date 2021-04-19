# PTQ
https://arxiv.org/pdf/1810.05723v3.pdf<br>
## 介绍

神经网络的量化造成的精度损失可以通过训练来补偿，但需要完整的数据集信息(大都涉及隐私等)，而且训练很耗时，故此一些统称为训练后量化的方法被提出，这些方法只需要量化权重和激活值，不需要重新训练。但是，低于8bit的量化会导致显著的精度下降，因此作者研究了CNN训练后的4bit量化。<br>
作者提出了三种训练后量化的方法：ACIQ、Per-channel bit allocation、Bias-correction。

### ACIQ: Analytical Clipping for Integer Quantization
通过最小化均方误差逼近最佳剪切值。论文中提出，经过研究发现对权重进行clip并无优势，因此该方法仅用于激活值的量化。<br>
假设X是高精度的随机变量，f(x)是其概率密度函数，E(x) = 0 (不失一般性，因为可以随时添加或减去该均值)，总位宽为M，要将值量化为 2^M 个离散值。
首先，定义一个剪裁函数 clip(x , α) , x∈R：<br>
![image](https://user-images.githubusercontent.com/58316204/115141275-e9d71b00-a06d-11eb-9f3c-faa30c4ad914.png)<br>
范围为 [−α , α]。将其等分为 2^M 个量化区域，相邻的量化步长为：<br>
![image](https://user-images.githubusercontent.com/58316204/115141333-3b7fa580-a06e-11eb-89a9-7aa684e56bd0.png)<br>
假设每个量化值都位于量化区域中点，则可得到X与量化后的值Q(X)的均方误差：<br>
![image](https://user-images.githubusercontent.com/58316204/115141370-741f7f00-a06e-11eb-925f-796e93fee89f.png)<br>
其中，第一项和第三项是剪裁函数 clip 预期的均方误差(对于关于0对称的分布来说二者相等)，称为剪裁噪声；第二项则是将 [−α , α] 量化为 2^M 个量化区域的均方误差，称为量化噪声。<br>
作者通过构造分段函数来近似密度函数 f(x)，并且证明了量化噪声和剪裁噪声满足如下关系式(剪裁噪声是在 Laplace(0，b) 的条件下)：<br>
![image](https://user-images.githubusercontent.com/58316204/115141423-b1840c80-a06e-11eb-8064-5f238def6a93.png)<br>
![image](https://user-images.githubusercontent.com/58316204/115141428-b5b02a00-a06e-11eb-9286-bec3108ec6ed.png)<br>
所以有：<br>
![image](https://user-images.githubusercontent.com/58316204/115141439-c1035580-a06e-11eb-99ba-bd38106497bc.png)<br>
为找到最优的 α，令其导数为零：<br>
![image](https://user-images.githubusercontent.com/58316204/115141443-c95b9080-a06e-11eb-9622-574bb0a35c28.png)<br>
M = 2，3，4 时，可求得 α∗ = 2.83b， 3.89b， 5.03b；实践中，可以估算 b=E(|X-E(X)|)。<br>

求解的源代码：[mse_analysis.py](mse_analysis.py)<br>
<br/>

### Per-channel bit allocation
该方法可用于量化权重和激活值。不限制所有通道都用4位表示，而是允许一些通道具有更高的位宽，而限制其他通道具有更低的位宽。唯一的要求是，写入或从内存中读取的总比特数保持不变(即保持每个通道的平均位宽为4)，使得总体的量化噪声的均方误差最小。<br>
给定n个通道，假设通道i的取值范围为 [−α , α] ，根据公式(5)的量化噪声，引入拉格朗日乘子 λ ：<br>
![image](https://user-images.githubusercontent.com/58316204/115141706-62d77200-a070-11eb-8ad6-fc29d0ec8102.png)<br>
令对 Mi 和 λ 的导数分别为0：<br>
![image](https://user-images.githubusercontent.com/58316204/115141743-8d292f80-a070-11eb-9ec7-05442106cab4.png)<br>
![image](https://user-images.githubusercontent.com/58316204/115141746-91554d00-a070-11eb-90e5-0132b1327927.png)<br>
然后作者证明了下列规则：<br>
![image](https://user-images.githubusercontent.com/58316204/115141759-a03bff80-a070-11eb-8038-e66d15aa1acc.png)<br>
两边取对数：<br>
![image](https://user-images.githubusercontent.com/58316204/115141770-aaf69480-a070-11eb-86b5-1ae3859a201f.png)<br>
[bit_allocation_synthetic.py](bit_allocation_synthetic.py)<br/>
<br/>

### Bias correction
该方法用于量化权重。主要思想为通过一种简单的方法补偿权重量化前后的均值和方差的偏差。<br>
作者观察到权重量化后其均值和方差存在固有偏差，即<br>
![image](https://user-images.githubusercontent.com/58316204/115141828-f5781100-a070-11eb-9bce-b146401b4cd3.png)<br>
通道c的校正常数为：<br>
![image](https://user-images.githubusercontent.com/58316204/115141841-09bc0e00-a071-11eb-94c9-ff9e4d33ab56.png)<br>
然后就可以用于更新权重：<br>
![image](https://user-images.githubusercontent.com/58316204/115141845-10e31c00-a071-11eb-9570-7f55f73825cf.png)<br>
[bias_correction.ipynb](bias_correction.ipynb)<br/>
<br/>


### Quantization
本代码中还使用了 [GEMMLOWP](https://github.com/google/gemmlowp/blob/master/doc/quantization.md) 量化方案并用 pytorch 实现。作者通过使用 ACIQ 减小通道的范围并为每个通道分配 bits，优化了这个量化方案。
[int_quantizer.py](pytorch_quantizer/quantization/qtypes/int_quantizer.py)
<br/><br/>

## 硬件要求
NVIDIA GPU / cuda support

## 数据
- 运行本代码需要用到 [ILSVRC2012](http://www.image-net.org/) 数据集。
- 可以通过 --data "PATH_TO_ILSVRC" 来设置数据集路径，或将数据集复制到 ~/datasets/ILSVRC2012.

## 代码运行
- 创建 Python3 虚拟环境
```
virtualenv --system-site-packages -p python3 venv3
. ./venv3/bin/activate
```
- 安装
```
pip install torch torchvision bokeh pandas sklearn mlflow tqdm
```
- build kernels<br>
To improve performance GEMMLOWP quantization was implemented in cuda and requires to compile kernels.
```
cd kernels
./build_all.sh
cd ../
```
- 运行
>*Post-training quantization of Res50.*
- Experiment W4A4 naive:
```
python inference/inference_sim.py -a resnet50 -b 512 -pcq_w -pcq_a -sh --qtype int4 -qw int4
```
>* Prec@1 62.154 Prec@5 84.252
- Experiment W4A4 + ACIQ + Bit Alloc(A) + Bit Alloc(W) + Bias correction:
```
python inference/inference_sim.py -a resnet50 -b 512 -pcq_w -pcq_a -sh --qtype int4 -qw int4 -c laplace -baa -baw -bcw
```
>* Prec@1 73.330 Prec@5 91.334
<br/>

## 实验结果
![experiments](fig/experiments.png)
<br>

# Additional knowladge
## Post-Training Quantization
量化后的神经网络中的参数通常还需要进行调整,这可以通过对模型进行再训练来完成，这个过程称为量化感知训练训练(Quantization-Aware Training，QAT)，或者不进行再训练，这个过程通常被称为训练后量化(Post-Training Quantization，PTQ)。在 QAT 中，预先训练的模型被量化，然后使用训练数据进行微调，以调整参数和恢复精度下降。在 PTQ 中，使用校准数据(例如，一小部分训练数据)对预训练模型进行校准，以计算裁剪范围和比例因子。然后，根据标定结果对模型进行量化。校准过程通常与QAT的微调过程并行进行。<br><br/>
在 PTQ 中，所有的权值和激活量化参数都是确定的，无需对神经网络模型进行再训练。因此，PTQ 是一种快速量化神经网络模型的方法。然而，与 QAT 相比，这通常是以较低的精度为代价的。

## The Laplace distribution && Gaussian distribution
拉普拉斯分布：如果随机变量的概率密度函数分布，那么它就是拉普拉斯分布，记为 x-Laplace（μ,b)，其中，μ 是位置参数，b 是尺度参数。如果 μ = 0，那么，正半部分恰好是尺度为 1/b(或者b，看具体指数分布的尺度参数形式) 的指数分布的一半。<br>
![image](https://user-images.githubusercontent.com/58316204/115142884-b3ea6480-a076-11eb-8d64-aaa0fc5de04e.png)<br>
高斯分布（正态分布）：若随机变量X服从一个数学期望为 μ、方差为 σ^2 的正态分布，记为 N(μ，σ^2)。其概率密度函数为正态分布的期望值 μ 决定了其位置，其标准差 σ 决定了分布的幅度。当 μ = 0，σ = 1 时的正态分布是标准正态分布。
![image](https://user-images.githubusercontent.com/58316204/115142968-1a6f8280-a077-11eb-89f6-cf5024aeab3e.png)<br>

## GEMMLOWP Quantization scheme
GEMM为通用矩阵乘<br>
![image](https://user-images.githubusercontent.com/58316204/115143477-2a3c9600-a07a-11eb-9780-20ad8c41fe5d.png)<br>
LOWP（Low-precision）是指输入和输出矩阵项都是最多8位的整数，标量类型是 uint8_t。<br>
gemmlowp 允许对基于 uint8 值的矩阵执行计算，但是这些矩阵仅在以某种方式近似于实数矩阵的情况下才有用。

## Resnet
![image](https://user-images.githubusercontent.com/58316204/115180847-29067a00-a109-11eb-9e85-8b57b65fef9e.png)<br>
![image](https://user-images.githubusercontent.com/58316204/115180860-2e63c480-a109-11eb-97fc-948ed361a934.png)<br>

