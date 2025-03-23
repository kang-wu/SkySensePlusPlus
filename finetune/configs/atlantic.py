crop_size = (
    256,
    256,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        537.9411629981602,
        615.7886221108977,
        343.4481583821405,
        3010.641650390625,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    std=[
        367.4598430230881,
        254.2473100510193,
        187.5437562223154,
        921.0792775874182,
    ],
    type='SegDataPreProcessor')
data_root = 'rs_datasets/'
dataset_type = 'AtlanticDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=1000, save_best='mIoU',max_keep_ckpts=1,
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    testeval=dict(type='TestEvalLoopHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
mean = [
    537.9411629981602,
    615.7886221108977,
    343.4481583821405,
    3010.641650390625,
]
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=512,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        drop_rate=0.1,
        embed_dims=1024,
        img_size=crop_size,
        in_channels=4,
        interpolate_mode='bilinear',
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        norm_eval=False,
        num_heads=16,
        num_layers=24,
        out_indices=(
            5,
            11,
            17,
            23,
        ),
        patch_size=4,
        qkv_bias=True,
        type='VisionTransformer',
        with_cls_token=False),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            537.9411629981602,
            615.7886221108977,
            343.4481583821405,
            3010.641650390625,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
        std=[
            367.4598430230881,
            254.2473100510193,
            187.5437562223154,
            921.0792775874182,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            512,
            512,
            512,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    neck=dict(
        in_channels=[
            1024,
            1024,
            1024,
            1024,
        ],
        out_channels=512,
        scales=[
            1,
            1,
            1,
            1,
        ],
        type='MultiLevelNeck'),
    pretrained="pretrain/skysensepp_release_s2.pth",
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=6e-05, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=10000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(seed=20240315)
resume = False
std = [
    367.4598430230881,
    254.2473100510193,
    187.5437562223154,
    921.0792775874182,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'deforestation_atlantic/deforestation_atlantic_test.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='AtlanticDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=10000, type='IterBasedTrainLoop', val_interval=500)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'deforestation_atlantic/deforestation_atlantic_train.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='AtlanticDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadSingleRSImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'deforestation_atlantic/deforestation_atlantic_val.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(type='LoadSingleRSImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='AtlanticDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'save/atlantic_skysensepp'
