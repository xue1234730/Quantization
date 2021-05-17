# PACT: Parameterized Clipping Activation for Quantized Neural Networks
https://arxiv.org/abs/1805.06085

## 介绍
该方法提出一个新的激活函数，即PACT(PArameterized Clipping Activation)，其作用在训练阶段，即该方案是从头开始训练的。

作者发现：在运用权重量化方案来量化 activation 时，在大规模图像分类任务上（如ImageNet），传统RELU的量化结果和全精度结果相差较大。相较于 weight 基本在 ( 0 , 1 ) ，activation的值是无限大的，所以提出截断式 RELU 的激活函数。该截断的上界，即文中的 α 是可学习的参数，这保证了每层都能有不一样的量化范围。新添的参数 α 在训练时使用 L2 正则化，使其快速收敛的同时，保持一个较小的值，以限制量化时产生的量化误差。

PACT激活函数,相当于relu函数截断上界，输出限制在 [0,α]:<br>
![image](https://user-images.githubusercontent.com/58316204/117118081-e7412900-adc2-11eb-8e01-af26261d9fed.png)<br>
将其线性量化为k位:<br>
![image](https://user-images.githubusercontent.com/58316204/117118180-0dff5f80-adc3-11eb-8c5b-959a5c4f5115.png)<br>

代码：
```
y = torch.clamp(x, min = 0, max = alpha.item())
scale = (2**k - 1) / alpha
y_q = torch.round( y * scale) / scale
```
 
α相当于loss中的参数，可以通过反向传播进行学习(STE)：<br>
![image](https://user-images.githubusercontent.com/58316204/117118234-1eafd580-adc3-11eb-86fd-3abb115ca70b.png)

## Additional Knowledge

# DoReFa量化
代码中还使用了[DoReFa](https://arxiv.org/pdf/1606.06160.pdf)量化方法

代码：
```
def quantize_k(r_i, k):
  scale = (2**k - 1)
  r_o = torch.round( scale * r_i ) / scale
  return r_o

class DoReFaQuant(Function):
	@staticmethod
	def forward(ctx, r_i, k):
    tanh = torch.tanh(r_i).float()
	  r_o = 2*quantize_k( tanh / (2*torch.max(torch.abs(tanh)).detach()) + 0.5 , k) - 1
	  return r_o
  
  @staticmethod
	def backward(ctx, dLdr_o):
    return dLdr_o, None
```
