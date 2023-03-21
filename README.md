<div align="center">
<h1>StreamPETR</h1>
<h3>Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection</h3>
</div>


<div align="center">
  <img src="figs/framework.png"/>
</div><br/>
<div align="center">
  <img src="figs/fps.png"/>
</div><br/>
    
## Roadmap

- [ ] StreamPETR code
- [ ] PETR code
- [ ] Focal-PETR code
- [ ] Flash Attention
- [ ] Checkpoints
- [ ] Batch size 2 training for temporal models
- [ ] Efficient training in streaming video
- [ ] TensorRT Inference
- [ ] 3D object tracking

<!-- ## Introduction -->
This repository is an official implementation of StreamPETR.
## Main Results
We provide some results on nuScenes **val set** with pretrained models. These model are trained on 8x 2080ti.
| config            | mAP      | NDS     |FPS-Pytorch    |   config |   download |
|:--------:|:----------:|:---------:|:--------:|:--------:|:-------------:|
| StreamPETR-R50-704x256   | 45.0%     | 55.0%    | 31.7  | |  

**Notes**: 
- FPS is measured on NVIDIA RTX3090 GPU with batch size of 1 (containing 6 view images) and fp32 (Pytorch). 

## Code Release
We expect to upload the code and pretrained models at mid-April. Please stay tuned. 
