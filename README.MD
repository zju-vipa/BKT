Trans-Net
==========
Code for [https://ojs.aaai.org/index.php/AAAI/article/view/16222](Visual Boundary Knowledge Translation, AAAI2021). 

### Installation
For packages, 

```
pip install -r requirements.txt
```

Note that pydensecrf need to be installed from source, 
```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### Datasets

####  file organization
Note that ```full_data.txt``` maps input image path to ground-truth mask path and we split data into ``` train , val , test ```  in the code. All dataset are organized as follows:
```
/datasets
    /[HumanMatting]
        /Images
        /Labels
        full_data.txt
```

#### dataset source
Birds: [http://www.vision.caltech.edu/visipedia/CUB-200-2011.html](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

HumanMatting: [https://github.com/aisegmentcn/matting_human_datasets](https://github.com/aisegmentcn/matting_human_datasets)

Flowers: [http://www.robots.ox.ac.uk/~vgg/data/flowers/](http://www.robots.ox.ac.uk/~vgg/data/flowers/)

MSRA10K and MSRA-B: [https://mmcheng.net/msra10k/](https://mmcheng.net/msra10k/)

ECSSD and CSSD: [http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)

DUT-OMRON:  [http://saliencydetection.net/dut-omron/](http://saliencydetection.net/dut-omron/)

PASCAL-Context: [https://www.cs.stanford.edu/~roozbeh/pascal-context/](https://www.cs.stanford.edu/~roozbeh/pascal-context/)

HKU-IS: [https://i.cs.hku.hk/~gbli/deep_saliency.html](https://i.cs.hku.hk/~gbli/deep_saliency.html)

SOD: [http://elderlab.yorku.ca/SOD/SOD.zip](http://elderlab.yorku.ca/SOD/SOD.zip)

THUR15K: [https://mmcheng.net/code-data/](https://mmcheng.net/code-data/)

SIP1K: [https://mmcheng.net/code-data/](https://mmcheng.net/code-data/)


### SOTA Comparison
CAC: [https://github.com/akwasigroch/coattention_object_segmentation](https://github.com/akwasigroch/coattention_object_segmentation)

ReDO:  [https://github.com/mickaelChen/ReDO](https://github.com/mickaelChen/ReDO)

SG-One: [https://github.com/xiaomengyc/SG-One](https://github.com/xiaomengyc/SG-One)

PANet: [https://github.com/kaixin96/PANet](https://github.com/kaixin96/PANet)

ALSSS: [https://github.com/hfslyc/AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg)

USSS: [https://github.com/tarun005/USSS_ICCV19](https://github.com/tarun005/USSS_ICCV19)

Unet/FPN/LinkNet/PSPNet/PAN: [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

DeepLab_V3+: [https://github.com/jfzhang95/pytorch-deeplab-xception.git](https://github.com/jfzhang95/pytorch-deeplab-xception.git)

### How to run

``python main.py --device_id=[gpu_id]``

##### Noted that you need to set dataset folder in `` main.py``

## Acknowledgements

Part of the code is heavily borrowed from the official code 
release of Deeplab_v3+([https://github.com/jfzhang95/pytorch-deeplab-xception.git](https://github.com/jfzhang95/pytorch-deeplab-xception.git))
