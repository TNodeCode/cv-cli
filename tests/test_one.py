import os
import pytest
from pathlib import Path
from PIL import Image as PILImage
from cvsdk.model import Dataset, Image, BoundingBox
from cvsdk.model.loaders.coco import CocoLoader
from cvsdk.model.loaders.yolo import YOLOLoader
from cvsdk.model.loaders.mot import MOTLoader


@pytest.fixture
def detection_dataset():
    """
    Fixture to create a mock MOT17 dataset structure for testing.
    """
    dataset = Dataset(
        images=[
            Image(
                id=1,
                file_name="image1.jpg",
                width=512,
                height=512,
                bounding_boxes=[
                    BoundingBox(xmin=100, ymin=150, width=50, height=80, category_id=1),
                    BoundingBox(xmin=10, ymin=20, width=50, height=30, category_id=2)
                ]
            ),
            Image(
                id=2,
                file_name="image2.jpg",
                width=512,
                height=512,
                bounding_boxes=[
                    BoundingBox(xmin=45, ymin=150, width=50, height=150, category_id=1),
                    BoundingBox(xmin=230, ymin=350, width=90, height=120, category_id=2)
                ]
            )
        ],
        categories={1: "cat", 2: "dog"},
        task_type="detection"
    )
    return dataset

def test_coco_export(detection_dataset):
    """
    Test importing a MOT17 dataset.
    """
    os.makedirs("tmp/datasets/coco", exist_ok=True)
    coco_dict = CocoLoader.export_dataset(detection_dataset, output_path=Path("tmp/datasets/coco/test.json"))
    print("COCO DICT", coco_dict)


def test_yolo_export(detection_dataset):
    """
    Test exporting a dataset to YOLO format.
    """
    os.makedirs("tmp/datasets/yolo", exist_ok=True)
    YOLOLoader.export_dataset(detection_dataset, output_dir="tmp/datasets/yolo")


def test_mot_export(detection_dataset):
    """
    Test exporting a dataset to MOT17 format.
    """
    os.makedirs("tmp/datasets/mot", exist_ok=True)
    MOTLoader.export_dataset(detection_dataset, output_dir="tmp/datasets/mot")