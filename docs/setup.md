# Environment Setup

## Base Environments  
Python >= 3.8 \
CUDA == 11.2 \
PyTorch == 1.9.0 \
mmdet3d == 1.0.0rc6 \
[flash-attn](https://github.com/HazyResearch/flash-attention) == 0.2.2

**Notes**: 
- [flash-attn](https://github.com/HazyResearch/flash-attention) is an optional requirement, which can speedup and save GPU memory. If your device (e.g. TESLA V100) is not compatible with the flash-attn, you can skip its installation and comment the relevant [code](../projects/mmdet3d_plugin/models/utils).
- It is also possible to consider installing version 1.9.0 + of Pytorch, but you need to find the appropriate flash-attn version (e.g. 0.2.8 for CUDA 11.6 and pytorch 1.13).


## Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation


**a. Create a conda virtual environment and activate it.**
```shell
conda create -n streampetr python=3.8 -y
conda activate streampetr
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```
**c. Install flash-attn (optional).**
```
pip install flash-attn==0.2.2
```

**d. Clone StreamPETR.**
```
git clone https://github.com/exiawsh/StreamPETR
```

**e. Install mmdet3d.**
```shell
pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd ./StreamPETR
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```