# SemiMAR
A Pytorch implementation of SemiMAR: Semi-Supervised Learning for CT Metal Artifact Reduction
![image](https://github.com/zjk1988/SemiMAR/blob/main/overview.png)

[Conference version](https://github.com/zzr-idam/UVM-Net/blob/main/fw3.png): ICASSP 2023 Oral Presentation

[Journal version](https://ieeexplore.ieee.org/abstract/document/10239528/): IEEE Journal of Biomedical and Health Informatics Featured Article

## Abstract
Metal artifacts lead to CT imaging quality degradation. With the success of deep learning (DL) in medical imaging, a number of DL-based supervised methods have been developed for metal artifact reduction (MAR). Nonetheless, fully-supervised MAR methods based on simulated data do not perform well on clinical data due to the domain gap. Although this problem can be avoided in an unsupervised way to a certain degree, severe artifacts cannot be well suppressed in clinical practice. Recently, semi-supervised metal artifact reduction (MAR) methods have gained wide attention due to their ability in narrowing the domain gap and improving MAR performance in clinical data. However, these methods typically require large model sizes, posing challenges for optimization. To address this issue, we propose a novel semi-supervised MAR framework. In our framework, only the artifact-free parts are learned, and the artifacts are inferred by subtracting these clean parts from the metal-corrupted CT images. Our approach leverages a single generator to execute all complex transformations, thereby reducing the model's scale and preventing overlap between clean part and artifacts. To recover more tissue details, we distill the knowledge from the advanced dual-domain MAR network into our model in both image domain and latent feature space. The latent space constraint is achieved via contrastive learning. We also evaluate the impact of different generator architectures by investigating several mainstream deep learning-based MAR backbones. Our experiments demonstrate that the proposed method competes favorably with several state-of-the-art semi-supervised MAR techniques in both qualitative and quantitative aspects.

## Prerequisites
-------------

This repository needs the following system settings:

 - Python 3.6 
 - Pytorch 1.6.0
 - CUDA 10.1
 - Matlab R2017b



## Acknowledgement
The authors would like to thank Dr. Haofu Liao (liaohaofu@gmail.com) for providing the codes framework. The authors would also like to thank Dr. Yanbo Zhang (yanbozhang007@gmail.com) and Dr. Hengyong Yu (hengyong_yu@uml.edu) for providing the artifact synthesis code used in this repository. 


## Citation
If you find this work useful for your research, please cite our paper:
```
@@ARTICLE{10239528,
  author={Wang, Tao and Yu, Hui and Wang, Zhiwen and Chen, Hu and Liu, Yan and Lu, Jingfeng and Zhang, Yi},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={SemiMAR: Semi-Supervised Learning for CT Metal Artifact Reduction}, 
  year={2023},
  volume={27},
  number={11},
  pages={5369-5380},
  keywords={Mars;Metals;Computed tomography;Image reconstruction;Generators;Decoding;Image quality;Computed tomography;metal artifact reduction;semi-supervised learning;contrastive learning},
  doi={10.1109/JBHI.2023.3312292}}
```
