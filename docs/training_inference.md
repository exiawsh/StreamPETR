# Train & inference
## Train
You can train the model following:

```bash
tools/dist_train.sh projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py 8 --work-dir work_dirs/stream_petr_r50_flash_704_bs2_seq_24e/
```

**Notes**: 
- We provide training config both in [sliding window](../projects/configs/StreamPETR/stream_petr_r50_flash_704_bs1_8key_2grad_24e.py) and [streaming video](../projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py). The results reported in our paper is trained with sliding window, while the sliding window training consumes huge GPU memory and training time. So we provide an additional training manner in streaming video (following [SOLOFusion](https://github.com/Divadi/SOLOFusion)). 

## Evaluation
You can evaluate the detection model following:
```bash
tools/dist_test.sh projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py work_dirs/stream_petr_vov_flash_800_bs2_seq_24e/latest.pth 8 --eval bbox
```

You can evaluate the tracking model following:
```bash
python nusc_tracking/pub_test --version v1.0-trainval --checkpoint {PATH_RESULTS.JSON} --data_root {PATH_NUSCENES}
```

## Estimate the inference speed of StreamPETR
The latency includes data-processing, network (FP32) and post-processing. Noting that \"workers_per_gpu\" may affect the speed because we include data processing time.
```bash
python tools/benchmark.py projects/configs/test_speed/stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py
```

## Visualize
You can generate the reault json following:
```bash
./tools/dist_test.sh projects/configs/StreamPETR/stream_petr_vov_flash_800_bs2_seq_24e.py work_dirs/stream_petr_vov_flash_800_bs2_seq_24e/latest.pth 8 --format-only
```
You can visualize the 3D object detection following:
```bash
python3 tools/visualize.py
# please change the results_nusc.json path in the python file
```

## Training Recipes
Here we provide some training tricks, which may further boost the performance of our model. In the further, we will try them and provide the improved baseline of our model.
1. The training of [streaming video](../projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py) converges relatively slowly (Sliding window with 60 epoches == streaming video with 90 epoches), but it still **saves 4x training hours**. The results in our paper were early trained using sliding window (8x frame window size).

2. To achieve SOTA results, we have modified the loss weights and Hungarian matching weights for the bounding box regression. Specifically, we change the `x,y weight from 1.0 to 2.0`. We find it works well on sparse query based designs.
3. The learning rate of backbone has significant impact for small models. For most 2D pretrained (e.g. R50-Nuimage or V2-99-FCOS) and large backbone (e.g. VIT-Base), we suggest setting it to 0.1. `For small IN1k-pretrained models (e.g. R50-IN1k), 0.25 or 0.5 is better`.
4. For small IN1k models, the results are not stable, Sync-BN can obtain stable results, while resulting in slightly longer training time. You can enable it by additionally setting `SyncBN=True` and changing the norm config to      `norm_cfg=dict(type='BN2d', requires_grad=True),
norm_eval=False`.
5. When training longer (e.g. 60ep), 300+128 queries has similar results to 644+256 queries, which is friendly to deployment.
6. The `dropout ratio for Transformer` may be sub-optimal.
7. `EMA` may boost the performance.
8. The `feedforward_channels for Transformer` can be set smaller (e.g. 512), it can improve the inference speed and has little impact on the accuracy.
9. Single frame detector pre-training.
10. If your device does not support Flash attention, change the config to `dict(type='PETRMultiheadAttention',
          embed_dims=256,
          num_heads=8,
          dropout=0.1,
          fp16=True,)`.
11. Adjuting the learning rate:

| Num_gpus * Batch_size | Learning Rate|
| :---: | :---: |
|8| 2e-4 |
|16| 4e-4 |
|32| 6e-4 |
|64| TBA |

## Detection Results
| Model | Setting |Pretrain| Lr Schd | NDS| mAP| Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
|PETR| R50 - 900q | ImageNet | 24ep | 34.9 | 30.9 |[config](../projects/configs/PETRv1/petrv1_r50_flash_704_24e.py) |[log](https://github.com/exiawsh/storage/releases/download/v1.0/petrv1_r50_flash_704_24e.log)|
|FocalPETR| R50 - 900q | ImageNet | 24ep | 36.6| 33.1 |[config](../projects/configs/PETRv1/focal_petrv1_r50_flash_704_24e.py) |[log](https://github.com/exiawsh/storage/releases/download/v1.0/focal_petrv1_r50_flash_704_24e.log)|
|StreamPETR| R50 - 900q | ImageNet | 24ep | 47.6 | 37.5 |[config](../projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_24e.py) |[log](https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_24e.log)|