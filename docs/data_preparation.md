# Data Preparation

## Dataset
**1. Download nuScenes**

Download the [nuScenes dataset](https://www.nuscenes.org/download) to `./data/nuscenes`.

## 2. Creating infos file

We modify data preparation in `MMDetection3D`, which addtionally creates 2D annotations and temporal information for training/evaluation. 
```shell
python tools/create_data_nusc.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes2d --version v1.0
```

Using the above code will generate `nuscenes2d_infos_temporal_{train,val}.pkl`.
We also privided the processed [train](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_train.pkl), [val](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_val.pkl) and [test](https://github.com/exiawsh/storage/releases/download/v1.0/nuscenes2d_temporal_infos_test.pkl) pkl.

## Pretrained Weights
```shell
cd /path/to/StreamPETR
mkdir ckpts
```
Please download the pretrained weights to ./ckpts. To verify the performance on the val set, we provide the pretrained V2-99 [weights](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth). The V2-99 is pretrained on DDAD15M ([weights](https://tri-ml-public.s3.amazonaws.com/github/dd3d/pretrained/depth_pretrained_v99-3jlw0p36-20210423_010520-model_final-remapped.pth)) and further trained on nuScenes **train set** with FCOS3D.  For the results on test set in the paper, we use the DD3D pretrained [weights](https://github.com/exiawsh/storage/releases/download/v1.0/dd3d_det_final.pth). The nuImage pretrained weights of R50 model can be found [here](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth). 



* After preparation, you will be able to see the following directory structure:  

**Folder structure**
```
StreamPETR
├── projects/
├── mmdetection3d/
├── tools/
├── configs/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes2d_temporal_infos_train.pkl
|   |   ├── nuscenes2d_temporal_infos_val.pkl
```