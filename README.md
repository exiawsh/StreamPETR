<div align="center">
<h1>StreamPETR</h1>
<h3>Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection</h3>
</div>

    
## Catalog

- [ ] StreamPETR code
- [ ] PETR code
- [ ] Focal-PETR code
- [ ] 3D object tracking
- [ ] Flash Attention
- [ ] Batch size 2 training for temporal models
- [ ] Efficient training in streaming video
- [ ] Checkpoints
- [ ] TensorRT Inference

<!-- ## Introduction -->
This repository is an official implementation of StreamPETR.
## Main Results
We provide some results on nuScenes **val set** with pretrained models. These model are trained on 8x 2080ti **without cbgs**.
| config            | mAP      | NDS     |FPS-Pytorch    |   config |   download |
|:--------:|:----------:|:---------:|:--------:|:--------:|:-------------:|
| StreamPETR-R50-704x256   | 45.0%     | 55.0%    | 31.7  | |  

**Notes**: 
- FPS is measured on NVIDIA RTX3090 GPU with batch size of 1 (containing 6 view images).

## Code Release
We expect to uploda the code and pretrained models at mid-April. Please stay tuned. 
