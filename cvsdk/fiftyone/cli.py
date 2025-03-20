import os
import click
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd


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


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
@click.option('--detections', type=str, required=True, help='CSV file containing model detections')
def detections(root_dir, annotations, images, detections):
    """
    Starts a FiftyOne application to inspect a computer vision object detection dataset.
    """
    # Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Parse the detections CSV
    detections_path = os.path.join(root_dir, detections)
    detections_df = pd.read_csv(detections_path)

    # Group detections by image filename
    grouped_detections = detections_df.groupby('filename')

    # Iterate over samples and add detections
    for sample in dataset:
        filename = os.path.basename(sample.filepath)
        if filename in grouped_detections.groups:
            detections = []
            for _, row in grouped_detections.get_group(filename).iterrows():
                # Assuming the CSV contains 'label', 'xmin', 'ymin', 'xmax', 'ymax', and 'confidence' columns
                label = row['label']
                confidence = row['confidence']
                # Convert absolute coordinates to relative [0, 1] bounding box
                bounding_box = [
                    row['xmin'] / sample.metadata.width,
                    row['ymin'] / sample.metadata.height,
                    (row['xmax'] - row['xmin']) / sample.metadata.width,
                    (row['ymax'] - row['ymin']) / sample.metadata.height,
                ]
                detections.append(
                    fo.Detection(
                        label=label,
                        bounding_box=bounding_box,
                        confidence=confidence,
                    )
                )
            # Add detections to the sample
            sample['detections'] = fo.Detections(detections=detections)
            sample.save()

    # Launch FiftyOne app
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")
    
    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")


@fiftyone.command()
@click.option('--root-dir', type=click.Path(exists=True), required=True, help='Root directory containing images and annotations')
@click.option('--annotations', type=str, required=True, help='COCO JSON annotation file')
@click.option('--images', type=str, required=True, help='Directory where images are stored')
def embeddings(root_dir, annotations, images):
    # Step 1: Load the COCO dataset into FiftyOne
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=os.path.join(root_dir, images),
        labels_path=os.path.join(root_dir, annotations),
    )

    # Step 2: Load embeddings from the .npy file
    # TODO load from npy file
    embeddings = np.random.normal(size=(len(dataset), 128))

    # Ensure the number of embeddings matches the number of samples
    assert len(embeddings) == len(dataset), "Mismatch between embeddings and samples"

    # Step 3: Associate embeddings with samples
    for sample, embedding in zip(dataset, embeddings):
        sample["embedding"] = embedding
        sample.save()

    # Step 4: Visualize embeddings
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method="umap",  # or "tsne", "pca" TODO add pacmap and create CLI parameter for this
        num_dims=2,
        brain_key="embedding_viz",
    )

    # Launch the FiftyOne app to explore the embeddings
    session = fo.launch_app(dataset)

    print("FiftyOne app is running. Press CTRL-C to stop.")
    
    try:
        # Keep the script running until CTRL-C is pressed
        session.wait()
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
