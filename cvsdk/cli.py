from cvsdk.format.coco import CocoExporter, CocoImporter
from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask
from cvsdk.model.validation import COCOObjectDetectionValidation
from cvsdk.cvat.cli import cvat as cvat_cli
from cvsdk.mmdet.cli import mmdet as mmdet_cli
from cvsdk.yolo.cli import yolo as yolo_cli
from cvsdk.fiftyone.cli import fiftyone as fo_cli
from cvsdk.torch.det.cli import torchdet as torchdet_cli
from cvsdk.inspection.cli import inspect as inspection_cli
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
cli.add_command(inspection_cli)


# Entry point of CLI
if __name__ == "__main__":
    cli()