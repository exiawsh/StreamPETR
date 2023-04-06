[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2303.11926)
![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/PETR)
<div align="center">
<h1>StreamPETR</h1>
<h3>Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection</h3>
</div>


<div align="center">
  <img src="figs/framework.png" width="550"/>
</div><br/>

## Introduction

This repository is an official implementation of StreamPETR.

## Getting Started

Please follow our documentation step by step. For the convenience of developers and researchers, we also add notes for developers to better convey the implementations of PF-Track and accelerate your adaptation of our framework. If you like my documentation and help, please recommend our work to your colleagues and friends.

1. [**Environment Setup.**](./docs/setup.md)
2. [**Data Preparation.**](./docs/data_preparation.md)
3. [**Training and Inference.**](./docs/training_inference.md)

## Model Zoo
<div align="center">
  <img src="figs/fps.png" width="550"/>
</div><br/>


| Model | Setting |Pretrain| Lr Schd | Training Time | NDS| mAP|FPS-pytorch | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: | :---: | :---: |
|StreamPETR| V2 - 900q | FCOS3D | 24ep | 13 hours | 57.1 | 48.3 | 12.5 |[config](projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py) |[model](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_vov_flash_800_bs2_seq_24e.pth)/[log](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_vov_flash_800_bs2_seq_24e.log) |
|StreamPETR| R50 - 900q | ImageNet | 90ep | 36 hours | 53.7 | 43.2 | 26.7 |[config](projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py) |[model](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_r50_flash_704_bs2_seq_90e.pth)/[log](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_vov_flash_800_bs2_seq_24e.log) |
|StreamPETR| R50 - 428q | [NuImg](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | 60ep | 26 hours | 54.6 |44.9 | 31.7 |[config](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth)/[log](https://github.com/exiawsh/storage/releases/download/untagged-117fd2925d1b0f8de361/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.log) |


The detailed results can be found in the training log. For other results on nuScenes val set, please see [Here](docs/training_inference.md).
## Results on NuScenes Test Set.
| Model | Setting |Pretrain|NDS| mAP|
| :---: | :---: | :---: | :---: | :---:|
|StreamPETR| V2-99 - 900q | DD3D | 63.6| 55.0 |

**Notes**: 
- FPS is measured on NVIDIA RTX 3090 GPU with batch size of 1 (containing 6 view images) and FP32. 
- The training time is measured with 8x 2080ti GPUs.

## Currently Supported Features

- [x] StreamPETR code (also including PETR and Focal-PETR)
- [x] Flash Attention
- [x] Checkpoints
- [x] Sliding window training
- [x] Efficient training in streaming video
- [ ] TensorRT Inference
- [ ] 3D object tracking

## Acknowledgements

We thank these great works and open-source codebases:

* 3D Detection. [MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [SOLOFusion](https://github.com/Divadi/SOLOFusion).
* Multi-object tracking. [MOTR](https://github.com/megvii-research/MOTR), [PF-Track](https://github.com/TRI-ML/PF-Track).


## Citation

If you find StreamPETR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@article{wang2023exploring,
  title={Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection},
  author={Wang, Shihao and Liu, Yingfei and Wang, Tiancai and Li, Ying and Zhang, Xiangyu},
  journal={arXiv preprint arXiv:2303.11926},
  year={2023}
}
```