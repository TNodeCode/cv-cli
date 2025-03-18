import os
import click
import fiftyone as fo


@click.group()
def fiftyone():
    """CLI for training and managing a YOLO model on a custom dataset."""
    pass


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
def app(root_dir, annotations, images):
    """
    Starts a FiftyOne application to inspect a computer vision object detection dataset.
    """
    # Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Launch FiftyOne app
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")
    
    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
