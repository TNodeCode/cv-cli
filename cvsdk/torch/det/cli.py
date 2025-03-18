import os
import torch
import click
from torchvision.transforms import v2

from cvsdk.torch.det.detectors import *


@click.group()
def torchdet():
    """CLI for training and managing a YOLO model on a custom dataset."""
    pass


@torchdet.command()
@click.option('--root-dir', type=click.Path(exists=True), default='./data/coco', help='Path to dataset')
@click.option('--epochs', type=int, default=10, help='Number of epochs for training')
@click.option('--batch-size', type=int, default=2, help='Batch size')
@click.option('--lr', type=float, default=0.001, help='Learning rate')
@click.option('--img-size', type=int, default=640, help='Image size for training')
def train(root_dir: str, epochs: int, batch_size: int, lr: float, img_size: int):
    dataset_name = "7s"
    num_classes = 10
    data_root_dir = root_dir
    train_data_dir = os.path.join(data_root_dir, "train2017")
    train_annotation_file = os.path.join(data_root_dir, "annotations", "instances_train2017.json")
    val_data_dir = os.path.join(data_root_dir, "val2017")
    val_annotation_file = os.path.join(data_root_dir, "annotations", "instances_val2017.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_transforms = v2.Compose([
        v2.ToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = v2.Compose([
        v2.ToTensor(),
    ])
    detector = FasterRCNNV2Detector(
        num_classes=num_classes,
        device=device,
        root_dir=root_dir,
    )
    detector.train(
        n_epochs = epochs,
        lr = float(lr),
        batch_size = 16,
        start_epoch = 0,
        resume = None,
        save_every = 10,
        lr_step_every = 10,
        num_classes = num_classes,
        device=device,
        log_dir=os.path.join(root_dir, "logs", dataset_name, detector.name),
        train_data_dir = train_data_dir,
        train_annotation_file = train_annotation_file,
        train_transforms = train_transforms,
        val_data_dir = val_data_dir,
        val_annotation_file = val_annotation_file,
        val_transforms = val_transforms,
        val_batch_size=2,
        n_batches_validation=2,
        test_data_dir = None,
        test_annotation_file = None,
        test_transforms = None,    
    )