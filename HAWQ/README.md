# HAWQ: Hessian AWare Quantization
https://arxiv.org/pdf/2011.10680.pdf
## 简介

文章贡献：<br>
1.模型推理过程仅包括整数乘法，加法和移位，而无需任何浮点运算/转换或整数除法。

2.混合精度量化作为一个整数线性规划问题，在模型扰动和内存占用/延迟之间取得平衡。

3.在TVM开发第一个开源4位/比特和混合精度的量化工具，ResNet50模型部署到T4 GPU，与INT8量化相比， INT4的平均速度提高了1.45倍。

4.ResNet50模型INT8精度量化的准确度达到77.58%，比之前的整数量化性能高2.68%；而混合精度INT4/8量化比INT8的推理延迟降低23%，而准确度仍然达到76.73%。

### Quantized Matrix Multiplication and Convolution
HAWQ使用均匀量化，对权重使用对称量化，对激活使用非对称量化，以及所有比例因子都使用静态量化。<br>
假设一个隐藏层的输入为 h，权重张量为 W，然后经过 ReLU 激活函数。首先将 h 和 W 量化为 Shqh 和 Swqw，其中 Sh 和 Sw 为实值量化比例，qh 和 qw 是相应的量化整数值。最后的输出结果 α 为：α = SwSh(qw * qh)，其中 qw * qh 为低精度整数的矩阵乘法(或卷积操作)。然后将 α 解量化并传送到下一层，其中 Sa 是预先计算的输出激活的比例因子。：<br>
![images](https://github.com/xue1234730/Quantization/blob/main/HAWQ/fig/3.PNG)<br>
在HAWQ-V3中，qw∗qh操作是通过低精度纯整数乘法和INT32累加来执行的，最终的INT32结果通过  （SwSh）/Sa 缩放来量化。后者是浮点缩放，需要与累积的结果相乘(以INT32精度计算)。在这个阶段，简单的实现需要浮点乘法。然而，这可以通过强制缩放为二进制数来避免。二进制数是 b/2^c 格式的有理数，其中b、c是两个整数。因此，使用INT32整数乘法和移位可以有效地执行上述的二元缩放。给定一个特定的 （SwSh）/Sa，用 DN(Dyadic Number) 表示可以计算相应的b和c的函数:<br>
![images](https://github.com/xue1234730/Quantization/blob/main/HAWQ/fig/4.PNG)<br>
### Batch Normalization
传统的 Batch Normalization 操作为：<br>
![images](https://github.com/xue1234730/Quantization/blob/main/HAWQ/fig/BN.PNG)<br>
在推理过程中这些参数都是固定的，因此 BN 操作可以与卷积融合，同时文章还使用纯整数方法将其量化，避免了量化 BN 参数导致的显著精度退化的问题。首先保持Conv和BN层展开，并允许BN统计数据更新。在几个epoch之后，我们将运行的统计数据冻结在BN层，并折叠CONV层和BN层。<br>
![images](https://github.com/xue1234730/Quantization/blob/main/HAWQ/fig/Fusion.PNG)<br>

## 使用
### 安装

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* **To install HAWQ** and develop locally:
```bash

pip install -r requirements.txt
```

### 代码
uniform 8-bit quantization for resnet50 on ImageNet. （修改数据集存放路径"--data /path/to/imagenet/ " 以及结果保存路径 “--save-path /path/to/checkpoints/”）
```
export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 --epochs 1 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --pretrained --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 0.0001 --fix-BN --checkpoint-iter -1 --quant-scheme uniform8
```
评估模型（修改数据集存放路径"--data /path/to/imagenet/ " 以及结果保存路径 “--save-path /path/to/checkpoints/”）
```
export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 --epochs 90 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 1 --checkpoint-iter -1 --quant-scheme bops_0.5 --resume /path/to/resnet50_bops0.5/checkpoint.pth.tar --resume-quantize -e
```

## Related Works
  - [HAWQ-V3: Dyadic Neural Network Quantization (ICML 2021)](https://arxiv.org/abs/2011.10680)
  - [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks (NeurIPS 2020)](https://proceedings.neurips.cc//paper/2020/file/d77c703536718b95308130ff2e5cf9ee-Paper.pdf)
  - [HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)


## License
HAWQ is released under the [MIT license](LICENSE).
