# DFQ
关于Data Free Quantization的Pytorch实现
## Results
On detection task  
- Tested with [MobileNetV2 SSD-Lite model](https://github.com/qfgaohao/pytorch-ssd)

<table>
</th><th>Pascal VOC 2007 test set (mAP with 07 metric)  </th></tr>
<tr><td>

model      | precision
-----------|------    
Original   | 68.70 
DFQ        | 65.39 

</td></tr> </table>

## Usage
There are 6 arguments, all default to False
  1. quantize: whether to quantize parameters and activations.  
  2. relu: whether to replace relu6 to relu.  
  3. equalize: whether to perform cross layer equalization.  
  4. correction: whether to apply bias correction
  5. clip_weight: whether to clip weights in range [-15, 15] (for convolution and linear layer)

run the equalized model by:
```
python main_cls.py --quantize --relu --equalize
```

run the equalized and bias-corrected model by:
```
python main_cls.py --quantize --relu --equalize --correction
```

## TODO
- [x] cross layer equalization
- [x] high bias absorption
- [x] data-free bias correction
- [x] test with detection model
- [x] test with classification model


## Acknowledgment
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/ricky40403/PyTransformer
- https://github.com/qfgaohao/pytorch-ssd
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/xxradon/PytorchToCaffe
