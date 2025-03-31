import os
import click
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import basename, splitext, join, expanduser
from cvsdk.inspection.backbones import load_backbone, get_available_backbones
from cvsdk.inspection.image_loader import get_image_from_url, get_image_from_fs, load_image_folder_as_tensor
from cvsdk.inspection.visualize import visualize_image
from cvsdk.inspection.factorization import compute_dff, scale_explanation
from structlog import get_logger

logger = get_logger()


@click.group()
def inspect():
    """
    Main command group for the CLI application.
    """
    pass


@inspect.command()
def available_models():
    backbones = get_available_backbones()
    for b in backbones:
        print("-", b)


def cmd_extract_features(model, feature_key, image_dir, output_filename: str):
    logger.info(
        f"Extract features from model {model} from layer {feature_key}")

    backbone = load_backbone(model)
    backbone.model.eval()
    logger.info("Model loaded")

    # Load images as tensor
    filenames = glob(image_dir)
    input_batch = load_image_folder_as_tensor(filenames=filenames, resize=None)
    logger.info("Loaded images as tensor: " + str(input_batch.shape))

    # Generate feature tensor
    image_features = backbone.get_features(input_batch, feature_key)
    if (type(image_features) == tuple):
        image_features = image_features[0]
        image_features = image_features.unsqueeze(1)
    logger.info("Extracted image features: " + str(image_features.shape))

    # Save feature tensors
    if not output_filename:
        output_filename = f"{image_dir}/{backbone.name}_{feature_key}.npz"
    np.savez_compressed(output_filename, image_features=image_features)
    logger.info("Feature tensors saved: " + str(output_filename))


@inspect.command()
@click.argument('model', type=str)
@click.argument('feature_key', type=str, default=None)
@click.argument('image_dir', type=str)
@click.argument('output_filename', type=str)
def extract_features(model, feature_key, image_dir, output_filename):
    """
    Command for extracting features from a model.

    This command loads images and extracts features from a model, i.e. ResNet50.
    You can specify which model and which feature layer should be used for extraction.


    Call:
        cv inspect extract-features resnet50 4 "data/7s/train2017/*.jpg" out.npz

    Args:
        model (str): Model name
        feature:key (str): Layer of the model (starting from 0)
        image_dir (str): Glob path pattern
        output_filename (str): Output filename (i.e. out.npz)
    """
    cmd_extract_features(model=model, feature_key=feature_key, image_dir=image_dir, output_filename=output_filename)


@inspect.command()
@click.argument('model', type=str)
@click.argument('feature_keys', type=str, default=None)
@click.argument('image_dir', type=str)
@click.argument('n_components', type=int)
def dff(model, feature_keys, image_dir, n_components):
    feature_keys = feature_keys.split(",")
    features_list = []  # features from different layers are stored here
    max_height, max_width, max_layers = 0, 0, 0

    # Load image features
    for i, feature_key in enumerate(feature_keys):
        feature_file_name = f"tmp/{model}_{feature_key}.npz"

        # Create directory for visualizations
        if not os.path.exists(feature_file_name):
            cmd_extract_features(
                model=model, feature_key=feature_key, image_dir=image_dir, output_filename=feature_file_name)

        # Load image features from fs
        features = np.load(
            feature_file_name, allow_pickle=True)
        features = features["image_features"]
        features_list.append(features)
        logger.info("Loaded image features: " + str(features.shape))

        # Remember which feature map was the largest
        if max_layers < features.shape[1]:
            max_layers = features.shape[1]
        if max_height < features.shape[2]:
            max_height = features.shape[2]
        if max_width < features.shape[3]:
            max_width = features.shape[3]

    # All feature layers need to have the same shape
    for i, f in enumerate(features_list):
        logger.debug("Index", i=i)
        logger.debug("1: features", f=f.shape, t=f.dtype)
        f = scale_explanation(
            np.transpose(f, (0, 2, 3, 1)),
            height=max_height, width=max_width, channels=f.shape[1]
        )
        logger.debug("2: Scaled explanation", f=f.shape)
        #f = np.transpose(
        #    f, (0, 1, 2, 3)
        #)
        logger.debug("3: scaled features", f=f.shape)
        f = np.concatenate(
            [f for i in range(max_layers // f.shape[1])], axis=1)
        logger.debug("4: concat features", f=f.shape)
        features_list[i] = f

    # Concatenate features of all layers
    for f in features_list:
        logger.debug("resulting features", f=f.shape)
    features = np.concatenate(features_list, axis=1)

    # Compute concepts and explanations
    logger.info("Computing concepts and components ...")
    concepts, explanations = compute_dff(features, n_components=n_components)
    logger.info("Concepts: " + str(concepts.shape) +
                ", Explanations: " + str(explanations.shape))

    # Load image files
    filenames = glob(image_dir)

    # Iterate over image files
    for i, filename in enumerate(filenames):
        logger.info(f"Processing image {i+1} / {len(filenames)}")

        # Load image
        img, rgb_img_float, _ = get_image_from_fs(
            filename,
            resize=None,
        )

        image_basename = basename(filename)
        image_name, image_ext = splitext(image_basename)

        # Scale explanation matrix to image's shape
        logger.debug("Image shape", shape=img.shape, explanations=explanations.shape)
        scaled_explanation = scale_explanation(
            explanations[i], height=img.shape[0], width=img.shape[1], channels=img.shape[2])

        # Create a visualization for the explanation matrix
        visualizations = visualize_image(
            concepts,
            scaled_explanation,
            None,
            rgb_img_float,
            image_weight=0.3
        )

        # Save the visualization
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))
        ax.axis('off')
        ax.imshow(visualizations)

        vis_dir = f"{image_dir}/{model}"
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)

        fig.savefig(f"{vis_dir}/{image_name}_{model}_{'-'.join(feature_keys)}_{n_components}components.png",
                    bbox_inches='tight', pad_inches=0)