# ViT-Large
The ViT-Large architecture we employed is [EVA-02](https://arxiv.org/abs/2303.11331). Following EVA02, the most attention layers utilize a 16x16 window size. For global attention, we set the attention window size in accordance with the shorter edge. Consequently, for 800x320, the maximum window would be 20x20, for 1600x800, the maximum window expands to 50x50. 

To save memory, we have adopted "grad checkpoint" and "flash attention" in backbone. This is highly pivotal as the "grad checkpoint" contributes to a reduction of up to 75% in memory occupation (40G -> 11G). Further, the impact of flash attention becomes significantly pronounced at higher resolutions. These code can download from https://github.com/baaivision/EVA/blob/master/EVA-02/det/detectron2/modeling/backbone/vit.py. We don't need external FPN and the setting is following:

```bash
sim_fpn=dict(
        scale_factors=[4, 2, 1, 0.5, 0.25],
        in_channels=1024,
        out_channels=256,
        out_indices=[2, 3, 4, 5, 6],
        )
img_backbone=dict(
    type='ViT',
    img_size=320,
    patch_size=16,
    window_size=16,
    global_window_size=20,
    in_chans=3,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4*2/3,
    window_block_indexes = (
    list(range(0, 2)) + list(range(3, 5)) + list(range(6, 8)) + list(range(9, 11)) + list(range(12, 14)) + list(range(15, 17)) + list(range(18, 20)) + list(range(21, 23))
    ),
    sim_fpn=None,
    # sim_fpn=sim_fpn, #Only for RepDETR3D
    qkv_bias=True,
    drop_path_rate=0.3,
    # use_act_checkpoint=False,
    # xattn=False,
    use_act_checkpoint=True,
    xattn=True,
    )
```
## Pretrain weights
You can download the pretrained weights from EVA02. The object365 [weights](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_coco_det_sys_o365.pth) will have a better performance but is not necessary. The ImageNet [weights](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/cls/in21k/eva02_L_pt_m38m_medft_in21k_p14.pt) only lower 1% NDS than object365. So I think the [Rope](https://arxiv.org/pdf/2104.09864.pdf) and CLIP distillation make the most contributions.  After downloaded the weights, you can transform the weights by:

```bash
import torch

pretrain_dict = torch.load('ckpts/eva02_L_coco_det_sys_o365.pth', map_location=torch.device('cpu'))
pretrain_dict = pretrain_dict["model"]
print(pretrain_dict.keys())
remapped_dict = {}
for k,v in pretrain_dict.items():
    if "backbone.net" in k:
        remapped_dict[k.replace("backbone.net.", "img_backbone.")] = v
    if "backbone.simfp" in k:
        remapped_dict[k.replace("backbone.", "img_backbone.adapter.")] = v
torch.save(remapped_dict,'ckpts/eva02_L_coco_det_sys_o365_remapped.pth')
```

## Layer-wise decay
You can following the setting of vov99 to train the ViT-Large directly. The performance will lower than layer-wise decay on 800x320 (~ 0.8% NDS), but will not obvious on 1600x800. So I suggest you use the "x0.1 backbone" firstly. For layer-wise decay, we set the learning rate of head 4x than backbone. The config can be find here:

```bash
optimizer = dict(constructor='LearningRateDecayOptimizerConstructor',     
    type='AdamW', 
    lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-7,
    paramwise_cfg={'decay_rate': 0.9,
                'head_decay_rate': 4.0,
                'decay_type': 'vit_wise',
                'num_layers': 24,
                })
```


## Results and json files
For the results of ViT-Large on val set, here is the result of StreamPETR (800x320):

```bash
（1） with x0.1 backbone lr
Evaluating bboxes of pts_bbox
mAP: 0.5152
mATE: 0.5755
mASE: 0.2539
mAOE: 0.3004
mAVE: 0.2332
mAAE: 0.2044
NDS: 0.6009
Eval time: 105.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.679   0.395   0.143   0.057   0.241   0.202
truck   0.477   0.579   0.186   0.063   0.194   0.207
bus     0.520   0.665   0.191   0.053   0.374   0.315
trailer 0.292   0.916   0.217   0.489   0.162   0.184
construction_vehicle    0.188   0.964   0.436   0.776   0.127   0.326
pedestrian      0.568   0.578   0.280   0.353   0.277   0.142
motorcycle      0.535   0.553   0.250   0.344   0.358   0.256
bicycle 0.549   0.438   0.260   0.438   0.133   0.004
traffic_cone    0.690   0.338   0.305   nan     nan     nan
barrier 0.654   0.328   0.273   0.130   nan     nan

（2） with layer wise decay
Evaluating bboxes of pts_bbox
mAP: 0.5304
mATE: 0.5636
mASE: 0.2546
mAOE: 0.3016
mAVE: 0.2400
mAAE: 0.2072
NDS: 0.6085
Eval time: 141.8s

Per-class results:
Object Class    AP      ATE     ASE     AOE     AVE     AAE
car     0.683   0.387   0.142   0.060   0.251   0.197
truck   0.479   0.601   0.187   0.060   0.209   0.228
bus     0.556   0.659   0.190   0.049   0.374   0.280
trailer 0.330   0.789   0.223   0.558   0.169   0.177
construction_vehicle    0.208   0.951   0.432   0.839   0.126   0.341
pedestrian      0.583   0.558   0.279   0.336   0.264   0.138
motorcycle      0.553   0.521   0.249   0.310   0.384   0.288
bicycle 0.563   0.463   0.254   0.352   0.143   0.009
traffic_cone    0.692   0.354   0.310   nan     nan     nan
barrier 0.656   0.354   0.281   0.149   nan     nan
```

**Notes**: 
- **I don't encourage anyone use future frames to achieve a better performance on the leaderboard.** In fact, we spent a lot of energy to get back to the first place withoutfuture frames. We hope that everyone can return to the correct research road. Please observe this rule !
- The RepDETR3D will acheive a higher performane than original StreamPETR. If any one use RepDETR3D as the baseline, please cite StreamPETR and use StreamPETR-D instead of RepDETR3D.
- If you want to use ViT-Large in your own methods, please also cite StreamPETR. We welcome anyone achieve a higher performance by using this setting!

Thank you very much!

