from globox import AnnotationSet, Annotation, BoundingBox, COCOEvaluator
from structlog import get_logger
import numpy as np
import pandas as pd
import glob
import re
import os


logger = get_logger()

class IoMinBoundingBox(BoundingBox):
    """This class overwrites the iou method from the BoundingBox class.

    It computes the intersection overminimum when "iou" is called.
    """
    def iou(self, other: BoundingBox) -> float:
        """The Intersection over Minimum (IoM) between two bounding boxes.

        Parameters:
            other: Other bounding box

        Return:
            Intersection over minimum between this and other bounding box
        """
        xmin = max(self._xmin, other._xmin)
        ymin = max(self._ymin, other._ymin)
        xmax = min(self._xmax, other._xmax)
        ymax = min(self._ymax, other._ymax)

        if xmax < xmin or ymax < ymin:
            return 0.0

        intersection = (xmax - xmin) * (ymax - ymin)
        minimum = min(self.area, other.area)

        if minimum == 0.0:
            return 1.0

        return min(intersection / minimum, 1.0)


def get_coco_gt_annotation_set(file_path: str) -> AnnotationSet:
    """Load a COCO annotation set.

    Args:
        file_path (str): path to COCO annotation file

    Returns:
        AnnotationSet: COCO image annotations
    """
    _gt_annotations = AnnotationSet.from_coco(file_path=file_path)
    gt_annotations = AnnotationSet()
    for ann in _gt_annotations:
        gt_annotations.add(ann)
    return gt_annotations


def read_csv_bboxes_as_df(csv_filepath: str) -> pd.DataFrame:
    """Read deections from CSV file.

    Args:
        csv_filepath (str): CSV file path

    Returns:
        pd.DataFrame: Data frame containing detected bounding boxes
    """
    return pd.read_csv(csv_filepath, index_col=None)


def bboxes_df_to_annotation_set(
        df: pd.DataFrame,
        threshold: float = 0.5,
        filename_pattern: str | None = None,
        bbox_class: BoundingBox = BoundingBox,
) -> AnnotationSet:
    """Turn data frame into AnnotationSet object.

    Args:
        df (pd.DataFrame): data frame containing annotations
        threshold (float, optional): _description_. Box overlap threshold
        filename_pattern (str | None, optional): Filename pattern. Defaults to None.
        bbox_class (BoundingBox, optional): Bounding box representationc class. Defaults to BoundingBox.

    Returns:
        AnnotationSet: AnnotationSet object
    """
    pred_annotations = AnnotationSet()
    df = df[df["score"] >= threshold]
    if filename_pattern:
        df = df[df["filename"].str.match(filename_pattern)]
    filenames = df["filename"].drop_duplicates()

    # Iterate over all filenames in the data frame
    for filename in filenames:
        # get all bounding boxes for a given image file
        bboxes = df[df["filename"] == filename]
        boxes = []
        for i, row in bboxes.iterrows():
            # transform bounding boxes into the globox format
            boxes.append(bbox_class(
                label=str(row["class_name"]),
                xmin=int(row["xmin"]),
                ymin=int(row["ymin"]),
                xmax=int(row["xmax"]),
                ymax=int(row["ymax"]),
                confidence=float(row["score"])
            ))
        # TODO compute image sizes
        annotation = Annotation(image_id=filename, image_size=(512,512), boxes=boxes)
        pred_annotations.add(annotation=annotation)
    return pred_annotations



def evaluate_predicted_bboxes(
    gt_annotations: AnnotationSet,
    pred_annotations: AnnotationSet,
    threshold: float = 0.5,
    max_detections: int = 100,
) -> dict:
    """Evaluate annotation set.

    Args:
        gt_annotations (AnnotationSet): Ground truth annotations
        pred_annotations (AnnotationSet): Predictions
        threshold (float, optional): Box overlap threshold. Defaults to 0.5.
        max_detections (int, optional): Maximum number of detections. Defaults to 100.

    Returns:
        dict: Dictionary contaning metrics
    """
    evaluator = COCOEvaluator(
        ground_truths=gt_annotations,
        predictions=pred_annotations,
    )
    evaluation = evaluator.evaluate(
        iou_threshold=threshold,
        max_detections=max_detections
    )

    ap = evaluation.ap()
    ar = evaluation.ar()
    f1 = (2*ap*ar)/(ap+ar) if ap+ar > 0.0 else 0.0
    metrics = {
        "ap": float(ap),
        "ar": float(ar),
        "f1": float(f1),
        "ap_50": float(evaluator.ap_50()),
        "ap_75": float(evaluator.ap_75()),
        "ap_small": float(evaluator.ap_small()) if evaluator.ap_small() else 0.0,
        "ap_medium": float(evaluator.ap_medium()) if evaluator.ap_medium() else 0.0,
        "ap_large": float(evaluator.ap_large()) if evaluator.ap_large() else 0.0,
        "ar_1": float(evaluator.ar_1()),
        "ar_10": float(evaluator.ar_10()),
        "ar_100": float(evaluator.ar_100()),
        "ar_small": float(evaluator.ar_small()) if evaluator.ar_small() else 0.0,
        "ar_medium": float(evaluator.ar_medium()) if evaluator.ar_medium() else 0.0,
        "ar_large": float(evaluator.ar_large()) if evaluator.ar_large() else 0.0,
    }
    return metrics


def evaluate_training_epochs(
    gt_file: str,
    detections_file: str,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate detections.

    Args:
        gt_file (str): Ground truth COCO annotation file
        detections_file (str): CSV detections file
        score_threshold (float, optional): Box overlap threshold. Defaults to 0.5.

    Returns:
        pd.DataFrame: Metrics dataframe
    """
    # In this list we store the metrics of all epochs
    epoch_metrics = []

    # These are the ground truth annotations the predictions are evaluated against
    gt_annotations = get_coco_gt_annotation_set(file_path=gt_file)

    # Read this CSV file
    df = read_csv_bboxes_as_df(detections_file)
    df = df[df['score'] > score_threshold]

    epochs = df['epoch'].unique()

    for epoch in epochs:
        df_epoch = df[df['epoch'] == epoch]
        # Create an annotation set based on that CSV file
        pred_annotations = bboxes_df_to_annotation_set(
            df=df_epoch,
            bbox_class=IoMinBoundingBox,
        )
        # Compute the metrics for the i-th epoch
        metrics = evaluate_predicted_bboxes(
            gt_annotations=gt_annotations,
            pred_annotations=pred_annotations
        )
        metrics |= {"epoch": int(epoch)}
        logger.info(f"Epoch {int(epoch)}", **metrics, n_detections=df_epoch.shape[0])
        # Add metrics of i-th epoch to list
        epoch_metrics.append(metrics)
    df_metrics = pd.DataFrame(epoch_metrics)
    # maximum index and value of 'ap' column
    max_ap_idx, max_ar_idx, max_f1_idx = df_metrics['ap'].idxmax(), df_metrics['ar'].idxmax(), df_metrics['f1'].idxmax()
    logger.info("Best AP", epoch=int(df_metrics.loc[max_ap_idx, 'epoch']), ap=float(df_metrics.loc[max_ap_idx, 'ap']))
    logger.info("Best AR", epoch=int(df_metrics.loc[max_ar_idx, 'epoch']), ar=float(df_metrics.loc[max_ar_idx, 'ar']))
    logger.info("Best F1", epoch=int(df_metrics.loc[max_f1_idx, 'epoch']), f1=float(df_metrics.loc[max_f1_idx, 'f1']))
    return df_metrics


def evaluate(
        gt_file: str,
        detections_file: str,
        results_file: str,
        score_threshold: float = 0.5,
) -> None:
    df_metrics = evaluate_training_epochs(
        gt_file=gt_file,
        detections_file=detections_file,
        score_threshold=score_threshold,
    )
    df_metrics.to_csv(results_file, index=False)
    logger.info(f"Saved results to {results_file}")
