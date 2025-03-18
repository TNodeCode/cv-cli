from cvsdk.format.coco import CocoExporter, CocoImporter
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask
from cvsdk.model.validation import COCOObjectDetectionValidation
from cvsdk.cvat.cli import cvat as cvat_cli
from cvsdk.mmdet.cli import mmdet as mmdet_cli
from cvsdk.yolo.cli import yolo as yolo_cli
from cvsdk.fiftyone.cli import fiftyone as fo_cli
from cvsdk.torch.det.cli import torchdet as torchdet_cli
from structlog import get_logger
import click


logger = get_logger()


@click.group()
def cli() -> None:
    """Main CLI group."""
    pass

# Add groups to CLI
cli.add_command(mmdet_cli)
cli.add_command(yolo_cli)
cli.add_command(fo_cli)
cli.add_command(cvat_cli)
cli.add_command(torchdet_cli)


# Entry point of CLI
if __name__ == "__main__":
    cli()


def validate():
    dataset: Dataset = CocoImporter.from_coco_json("data/spine/annotations/test_cocoformat_all.json", task_type="detection")
    dataset2: Dataset = CocoImporter.from_coco_json("data/spine/annotations/test_cocoformat_all.json", task_type="detection")
    result = COCOObjectDetectionValidation.validate(ground_truth=dataset, detections=dataset2)
    print(result)


def import_coco():
    dataset: Dataset = CocoImporter.from_coco_json("data/spine/annotations/test_cocoformat_all.json")
    CocoExporter.to_coco_json(dataset=dataset, output_path="demo.json")

    # Create a dataset
    dataset: Dataset = Dataset(images=[], categories={0: "cat", 1: "dog"})

    # Create an image
    image = Image(id=1, file_name="image_001.png", width=640, height=640)
    
    # Create a bounding box
    bbox1 = BoundingBox(xmin=100, ymin=100, xmax=200, ymax=200)
    bbox2 = BoundingBox(xmin=300, ymin=300, xmax=400, ymax=400)

    # Create a segmentation mask
    mask1 = SegmentationMask(segmentation=[10,50,30,80,20,10], category_id=1)
    mask2 = SegmentationMask(segmentation=[300,250,200,150,400,300], category_id=1)

    # add bounding boxes to image
    image.bounding_boxes.append(bbox1)
    image.bounding_boxes.append(bbox2)

    # add segmentation masks to image
    image.segmentation_masks.append(mask1)
    image.segmentation_masks.append(mask2)
    
    # add image to dataset
    dataset.images.append(image)

def conf():
    #from cvsdk.config import settings
    from dynaconf import Dynaconf

    settings = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=['settings.toml', '.secrets.toml'],
    )

    settings2 = Dynaconf(
        envvar_prefix="DYNACONF",
        settings_files=['faster_rcnn.toml', '.secrets.toml', 'configs/training.yml'],
    )

    print(settings.name)
    print(settings.mysec)
    print(settings2.model)