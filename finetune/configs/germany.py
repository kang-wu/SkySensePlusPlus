crop_size = (
    24,
    24,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        2482.0061841829206,
        2456.642580060208,
        2667.8229979675334,
        2744.9377076257624,
        3620.1499158373827,
        4063.9126981046647,
        3922.2406108776354,
        4264.908986788407,
        2453.0070206816135,
        1774.0019119673998,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        24,
        24,
    ),
    std=[
        2392.1256366526068,
        2100.1364646122875,
        2262.6154840764625,
        2353.899770400333,
        2089.598452203458,
        2057.1247114077073,
        2013.2108514271458,
        2041.0248949410561,
        1380.4643757742374,
        1243.547946113518,
    ],
    ts_size=30,
    type='RSTsSegDataPreProcessor')
data_root = 'rs_datasets/'
dataset_type = 'GermanyCropDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=1,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=20, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
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
    2482.0061841829206,
    2456.642580060208,
    2667.8229979675334,
    2744.9377076257624,
    3620.1499158373827,
    4063.9126981046647,
    3922.2406108776354,
    4264.908986788407,
    2453.0070206816135,
    1774.0019119673998,
]
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=3,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=18,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        downscale_indices=[
            -1,
        ],
        drop_path_rate=0.0,
        drop_rate=0.1,
        embed_dims=1024,
        img_size=(
            24,
            24,
        ),
        in_channels=10,
        init_cfg=dict(
            checkpoint=
            'pretrain/skysensepp_mmcvt_s2.pth',
            type='Pretrained'),
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
        bgr_to_rgb=False,
        mean=[
            2482.0061841829206,
            2456.642580060208,
            2667.8229979675334,
            2744.9377076257624,
            3620.1499158373827,
            4063.9126981046647,
            3922.2406108776354,
            4264.908986788407,
            2453.0070206816135,
            1774.0019119673998,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            24,
            24,
        ),
        std=[
            2392.1256366526068,
            2100.1364646122875,
            2262.6154840764625,
            2353.899770400333,
            2089.598452203458,
            2057.1247114077073,
            2013.2108514271458,
            2041.0248949410561,
            1380.4643757742374,
            1243.547946113518,
        ],
        ts_size=30,
        type='RSTsSegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            1024,
            1024,
            1024,
            1024,
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
        num_classes=18,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    neck=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=1024,
        in_channels=[
            768,
            768,
            768,
            768,
        ],
        in_channels_ml=[
            1024,
            1024,
            1024,
            1024,
        ],
        init_cfg=dict(
            checkpoint=
            'pretrain/skysensepp_mmcvt_fusion.pth',
            type='Pretrained'),
        input_dims=1024,
        mlp_ratio=4,
        num_heads=16,
        num_layers=24,
        out_channels=768,
        out_channels_ml=1024,
        output_cls_token=True,
        qkv_bias=True,
        scales=[
            4,
            2,
            1,
            0.5,
        ],
        scales_ml=[
            1,
            1,
            1,
            1,
        ],
        ts_size=30,
        type='FusionMultiLevelNeck',
        with_cls_token=True),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            cls_token=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            pos_embed=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=2000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=2000,
        by_epoch=False,
        end=20000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(seed=20240311)
resume = False
static_graph = True
std = [
    2392.1256366526068,
    2100.1364646122875,
    2262.6154840764625,
    2353.899770400333,
    2089.598452203458,
    2057.1247114077073,
    2013.2108514271458,
    2041.0248949410561,
    1380.4643757742374,
    1243.547946113518,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        'rs_datasets/germany_crop/germany_crop_val.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(data_key='image', ts_size=30, type='LoadTsImageFromNpz'),
            dict(
                data_key='image',
                reduce_zero_label=True,
                type='LoadAnnotationsNpz'),
            dict(type='PackSegInputs'),
        ],
        type='GermanyCropDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(data_key='image', ts_size=30, type='LoadTsImageFromNpz'),
    dict(data_key='image', reduce_zero_label=True, type='LoadAnnotationsNpz'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            0,
            1000,
        ),
        (
            4000,
            2000,
        ),
        (
            8000,
            4000,
        ),
    ],
    max_iters=20000,
    type='IterBasedTrainLoop',
    val_interval=2000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file=
        'rs_datasets/germany_crop/germany_crop_train.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(data_key='image', ts_size=30, type='LoadTsImageFromNpz'),
            dict(
                data_key='image',
                reduce_zero_label=True,
                type='LoadAnnotationsNpz'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='GermanyCropDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(data_key='image', ts_size=30, type='LoadTsImageFromNpz'),
    dict(data_key='image', reduce_zero_label=True, type='LoadAnnotationsNpz'),
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
        'rs_datasets/germany_crop/germany_crop_val.json',
        data_prefix=dict(img_path='images', seg_map_path='idx_labels'),
        data_root='rs_datasets/',
        pipeline=[
            dict(data_key='image', ts_size=30, type='LoadTsImageFromNpz'),
            dict(
                data_key='image',
                reduce_zero_label=True,
                type='LoadAnnotationsNpz'),
            dict(type='PackSegInputs'),
        ],
        type='GermanyCropDataset'),
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
work_dir = 'save_germany'