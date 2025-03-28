from mmengine import Config
from mmengine.runner import Runner
from dynaconf import Dynaconf
from cvsdk.mmdet.config import TrainingConfig


class MMDetModels:
  """MMDetection models class.
  """
  models = {
    "faster_rcnn": ["faster-rcnn_r50_fpn_1x_coco", "faster-rcnn_r101_fpn_1x_coco", "faster-rcnn_x101-32x4d_fpn_1x_coco", "faster-rcnn_x101-64x4d_fpn_1x_coco"],
    "cascade_rcnn": ["cascade-rcnn_r50_fpn_1x_coco", "cascade-rcnn_r101_fpn_1x_coco", "cascade-rcnn_x101-32x4d_fpn_1x_coco", "cascade-rcnn_x101-64x4d_fpn_1x_coco"],
    "deformable_detr": ["deformable-detr_r50_16xb2-50e_coco", "deformable-detr-refine_r50_16xb2-50e_coco", "deformable-detr-refine-twostage_r50_16xb2-50e_coco"],
    "yolox": ["yolox_nano_8xb8-300e_coco", "yolox_tiny_8xb8-300e_coco", "yolox_s_8xb8-300e_coco", "yolox_m_8xb8-300e_coco", "yolox_l_8xb8-300e_coco", "yolox_x_8xb8-300e_coco"],
  }

  @staticmethod
  def get_available_models() -> dict[str, list[str]]:
    """Get available models.

    Returns:
        dict[str, list[str]]: Dictionary of available models.
    """
    return MMDetModels.models
  
  @staticmethod
  def get_config(config_file: str, envvar_prefix: str = "", load_from: str | None = None) -> Config:
    """Load a configuration.

    Args:
        config_file (str): YAML training configuration file
        envvar_prefix (str, optional): Prefix for environment variables. Defaults to "".
        load_from (str | None, optional): Path to checkpoint file with pretrained weights. Defaults to None.

    Returns:
        Config: MMDetection configuration
    """
    settings = Dynaconf(
        envvar_prefix=envvar_prefix,
        settings_files=[config_file],
        lowercase_envvars=True,
    )
    
    config_data = {k.lower(): v for k, v in settings.items()}
    config = TrainingConfig(**config_data)

    DATASET_DIR=config.dataset_dir
    DATASET_CLASSES=config.dataset_classes

    MODEL_TYPE=config.model_type
    MODEL_NAME=config.model_name
    BATCH_SIZE=config.batch_size
    NUM_CLASSES=len(config.dataset_classes)
    EPOCHS=config.epochs
    WORK_DIR=f"{config.work_dir}/{MODEL_TYPE}/{MODEL_NAME}"

    ANN_TRAIN=config.annotations_train
    ANN_VAL=config.annotations_val
    ANN_TEST=config.annotations_test

    OPTIMIZER=config.optimizer

    cfg = Config.fromfile(f"mmdetection/configs/{MODEL_TYPE}/{MODEL_NAME}.py")
    print("LOAD FROM", load_from)
    cfg.load_from = load_from

    if OPTIMIZER=="SGD":
      cfg.optim_wrapper.optimizer = {
        'type': 'SGD',
        'lr': config.lr,
        'momentum': config.momentum,
        'weight_decay': config.weight_decay,
      }
    else:
      cfg.optim_wrapper.optimizer = {
        'type': 'AdamW',
        'lr': config.lr,
        'weight_decay': config.weight_decay,
      }

    # Here we can define image augmentations used for training.
    # see: https://mmdetection.readthedocs.io/en/v2.19.1/tutorials/data_pipeline.html
    train_pipeline = [
      dict(type='LoadImageFromFile', backend_args=None),
      dict(type='LoadAnnotations', with_bbox=True),
      dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    ]

    train_pipeline += config.augmentations

    train_pipeline += [
      dict(type='PackDetInputs')
    ]

    if MODEL_TYPE == "yolox":
      # YoloX uses MultiImageMixDataset, has to be configured differently
      cfg.train_dataloader.dataset.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.dataset.ann_file=f"annotations/{ANN_TRAIN}"
      cfg.train_dataloader.dataset.dataset.data_prefix.img=f"{config.train_dir}/"
      cfg.train_dataloader.dataset.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    else:
      cfg.train_dataloader.dataset.data_root=DATASET_DIR
      cfg.train_dataloader.dataset.ann_file=f"annotations/{ANN_TRAIN}"
      cfg.train_dataloader.dataset.data_prefix.img=f"{config.train_dir}/"
      cfg.train_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    cfg.val_dataloader.dataset.data_root=DATASET_DIR
    cfg.val_dataloader.dataset.data_prefix.img=f"{config.val_dir}/"
    cfg.val_dataloader.dataset.ann_file=f"annotations/{ANN_VAL}"
    cfg.val_evaluator.ann_file=f"{DATASET_DIR}annotations/{ANN_VAL}"
    cfg.val_dataloader.dataset.update({'metainfo': {'classes': DATASET_CLASSES}})
    cfg.test_dataloader.dataset.data_root=DATASET_DIR
    cfg.test_dataloader.dataset.data_prefix.img=f"{config.test_dir}/"
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
  def train(config_file: str, load_from: str | None = None):
    cfg = MMDetModels.get_config(config_file=config_file, load_from=load_from)
    runner = Runner.from_cfg(cfg)
    runner.train()