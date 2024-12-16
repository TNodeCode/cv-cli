from mmengine import Config
from mmengine.runner import Runner


class MMDetModels:
  models = {
    "faster_rcnn": ["faster-rcnn_r50_fpn_1x_coco", "faster-rcnn_r101_fpn_1x_coco", "faster-rcnn_x101-32x4d_fpn_1x_coco", "faster-rcnn_x101-64x4d_fpn_1x_coco"],
    "cascade_rcnn": ["cascade-rcnn_r50_fpn_1x_coco", "cascade-rcnn_r101_fpn_1x_coco", "cascade-rcnn_x101-32x4d_fpn_1x_coco", "cascade-rcnn_x101-64x4d_fpn_1x_coco"],
    "deformable_detr": ["deformable-detr_r50_16xb2-50e_coco", "deformable-detr-refine_r50_16xb2-50e_coco", "deformable-detr-refine-twostage_r50_16xb2-50e_coco"],
    "yolox": ["yolox_nano_8xb8-300e_coco", "yolox_tiny_8xb8-300e_coco", "yolox_s_8xb8-300e_coco", "yolox_m_8xb8-300e_coco", "yolox_l_8xb8-300e_coco", "yolox_x_8xb8-300e_coco"],
  }

  @staticmethod
  def get_available_models():
    return MMDetModels.models
  
  @staticmethod
  def get_config():
    models = MMDetModels.get_available_models()
    DATASET_DIR="data/spine/"
    DATASET_CLASSES=['spine']

    MODEL_TYPE="cascade_rcnn"
    MODEL_NAME=models[MODEL_TYPE][3]
    BATCH_SIZE=2
    NUM_CLASSES=1
    EPOCHS=36
    WORK_DIR=f"work_dirs/{MODEL_TYPE}/{MODEL_NAME}"

    ANN_TRAIN="instances_train2017_mixed.json"
    ANN_VAL="instances_val2017_mixed.json"
    ANN_TEST="instances_test2017_mixed.json"

    OPTIMIZER="ADAMW"

    cfg = Config.fromfile(f"mmdetection/configs/{MODEL_TYPE}/{MODEL_NAME}.py")

    if OPTIMIZER=="SGD":
      cfg.optim_wrapper.optimizer = {
        'type': 'SGD',
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 0.0001,
      }
    else:
      cfg.optim_wrapper.optimizer = {
        'type': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 0.05,
      }

    # Here we can define image augmentations used for training.
    # see: https://mmdetection.readthedocs.io/en/v2.19.1/tutorials/data_pipeline.html
    train_pipeline = [
      dict(type='LoadImageFromFile', backend_args=None),
      dict(type='LoadAnnotations', with_bbox=True),
      dict(type='Resize', scale=(1333, 800), keep_ratio=True),
      dict(type='RandomFlip', prob=0.5, direction="horizontal"), 
      dict(type='RandomFlip', prob=0.5, direction="vertical"),
      dict(type='RandomFlip', prob=0.5, direction="diagonal"),
      dict(type='RandomAffine', max_rotate_degree=25.0, max_translate_ratio=0.1, scaling_ratio_range=(0.5, 1.5), max_shear_degree=5.0),
      dict(type='PackDetInputs')
    ]


    if MODEL_TYPE == "yolox":
      # YoloX uses MultiImageMixDataset, has to be configured differently
      cfg.train_dataloader.dataset.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.dataset.ann_file=f"annotations/{ANN_TRAIN}"
      cfg.train_dataloader.dataset.dataset.data_prefix.img="train2017/"
      cfg.train_dataloader.dataset.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    else:
      cfg.train_dataloader.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.ann_file=f"annotations/{ANN_TRAIN}"
      cfg.train_dataloader.dataset.data_prefix.img="train2017/"
      cfg.train_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    cfg.val_dataloader.dataset.data_root=DATASET_DIR
    cfg.val_dataloader.dataset.data_prefix.img="val2017/"
    cfg.val_dataloader.dataset.ann_file=f"annotations/{ANN_VAL}"
    cfg.val_evaluator.ann_file=f"{DATASET_DIR}annotations/{ANN_VAL}"
    cfg.val_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    cfg.test_dataloader.dataset.data_root=DATASET_DIR
    cfg.test_dataloader.dataset.data_prefix.img="test2017/"
    cfg.test_dataloader.dataset.ann_file=f"annotations/{ANN_TEST}"
    cfg.test_evaluator.ann_file=f"{DATASET_DIR}annotations/{ANN_TEST}"
    cfg.train_cfg.max_epochs=EPOCHS
    cfg.default_hooks.logger.interval=10
    if MODEL_TYPE == "faster_rcnn":
      cfg.model.roi_head.bbox_head.num_classes=NUM_CLASSES
    elif MODEL_TYPE == "cascade_rcnn":
      cfg.model.roi_head.bbox_head[0].num_classes=NUM_CLASSES
      cfg.model.roi_head.bbox_head[1].num_classes=NUM_CLASSES
      cfg.model.roi_head.bbox_head[2].num_classes=NUM_CLASSES
    elif MODEL_TYPE == "deformable_detr":
      cfg.model.bbox_head.num_classes=NUM_CLASSES
    elif MODEL_TYPE == "yolox":
      cfg.model.bbox_head.num_classes=NUM_CLASSES
    cfg.train_dataloader.batch_size=BATCH_SIZE
    cfg.val_dataloader.batch_size=BATCH_SIZE
    cfg.test_dataloader.batch_size=BATCH_SIZE
    cfg.work_dir=WORK_DIR
    cfg.resume = True
    return cfg

  @staticmethod
  def train():
    cfg = MMDetModels.get_config()
    runner = Runner.from_cfg(cfg)
    runner.train()