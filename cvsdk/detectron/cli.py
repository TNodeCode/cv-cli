import click
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset


@click.command()
@click.argument('train_dataset_path', type=click.Path(exists=True))
@click.argument('val_dataset_path', type=click.Path(exists=True))
@click.argument('test_dataset_path', type=click.Path(exists=True))
@click.option('--dataset-format', type=click.Choice(['coco', 'voc', 'lvis', 'custom']), required=True, help='Format of the dataset (coco, voc, lvis, custom).')
@click.option('--epochs', default=10, help='Number of epochs to train the model.')
@click.option('--learning-rate', default=0.001, help='Learning rate for training.')
@click.option('--batch-size', default=2, help='Batch size for training.')
@click.option('--resume', type=click.Path(exists=True), help='Path to existing weight file to resume training.')
@click.option('--save-path', default='./model_final.pth', help='Path to save the trained model weights.')
def train(train_dataset_path, val_dataset_path, test_dataset_path, dataset_format, epochs, learning_rate, batch_size, resume, save_path):
    """
    Train a Faster R-CNN model using Detectron2.
    
    TRAIN_DATASET_PATH: Path to the training dataset.
    VAL_DATASET_PATH: Path to the validation dataset.
    TEST_DATASET_PATH: Path to the test dataset.
    """
    # Register datasets
    if dataset_format == 'coco':
        register_coco_instances("train_dataset", {}, f"{train_dataset_path}/annotations.json", train_dataset_path)
        register_coco_instances("val_dataset", {}, f"{val_dataset_path}/annotations.json", val_dataset_path)
        register_coco_instances("test_dataset", {}, f"{test_dataset_path}/annotations.json", test_dataset_path)
    else:
        raise NotImplementedError("Only COCO format is implemented. VOC, LVIS, and custom formats need implementation.")

    cfg = get_cfg()
    cfg.merge_from_file("path/to/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.DATASETS.TEST = ("val_dataset",)  # Set validation dataset for evaluation during training
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = epochs * 1000  # Adjust based on your dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Set the number of classes based on your dataset
    
    if resume:
        cfg.MODEL.WEIGHTS = resume
    else:
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

    # Create trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=bool(resume))
    
    # Add validation evaluation hook    
    val_evaluator = COCOEvaluator("val_dataset", cfg, False, output_dir="./output/")
    trainer.register_hooks(
        [hooks.EvalHook(0, lambda: trainer.test(cfg, trainer.model, evaluators=[val_evaluator]))]
    )

    trainer.train()

    # Save the final model weights
    trainer.save_model(save_path)

@click.command()
@click.argument('model_weights', type=click.Path(exists=True))
@click.argument('test_dataset_path', type=click.Path(exists=True))
@click.option('--dataset-format', type=click.Choice(['coco', 'voc', 'lvis', 'custom']), required=True, help='Format of the dataset (coco, voc, lvis, custom).')
def test(model_weights, test_dataset_path, dataset_format):
    """
    Test a trained Faster R-CNN model using Detectron2.
    
    MODEL_WEIGHTS: Path to the trained model weights.
    TEST_DATASET_PATH: Path to the test dataset.
    """
    if dataset_format == 'coco':
        register_coco_instances("test_dataset", {}, f"{test_dataset_path}/annotations.json", test_dataset_path)
    else:
        raise NotImplementedError("Only COCO format is implemented for testing. VOC, LVIS, and custom formats need implementation.")

    cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

    # Create trainer
    trainer = DefaultTrainer(cfg)
    
    cfg = get_cfg()
    cfg.merge_from_file("path/to/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_weights
    cfg.DATASETS.TEST = ("test_dataset",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the threshold for this model
    cfg.DATALOADER.NUM_WORKERS = 2

    # Perform evaluation    
    evaluator = COCOEvaluator("test_dataset", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "test_dataset")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == '__main__':
    cli = click.Group(commands={'train': train, 'test': test})
    cli()