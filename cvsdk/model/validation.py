from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from cvsdk.model import Dataset
from cvsdk.format.coco import CocoExporter
import json
from typing import Dict
from tempfile import NamedTemporaryFile


class COCOObjectDetectionValidation:
    """Computes COCO evaluation metrics for object detection."""

    @staticmethod
    def validate(ground_truth: Dataset, detections: Dataset) -> Dict[str, float]:
        """Compute COCO evaluation metrics.
        
        Args:
            ground_truth (Dataset): Ground truth dataset.
            detections (Dataset): Predicted detections dataset.

        Returns:
            Dict[str, float]: COCO evaluation metrics.
        """
        # Convert ground truth and detections to COCO JSON format
        gt_coco = CocoExporter.to_coco_dict(ground_truth)
        dt_coco = CocoExporter.to_coco_dict(detections)["annotations"]

        # Save COCO JSON files temporarily
        with NamedTemporaryFile(mode="w", delete=False, suffix=".json") as gt_file, NamedTemporaryFile(mode="w", delete=False, suffix=".json") as dt_file:
            json.dump(gt_coco, gt_file, indent=4)
            json.dump(dt_coco, dt_file, indent=4)
            gt_file_path, dt_file_path = gt_file.name, dt_file.name

        # Load COCO API objects
        coco_gt = COCO(gt_file_path)
        coco_dt = coco_gt.loadRes(dt_file_path)

        # Evaluate using COCO metrics
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract COCO metrics
        results = {
            "AP@[50:95]": coco_eval.stats[0],  # Mean AP across IoU thresholds
            "AP@50": coco_eval.stats[1],       # AP at IoU=0.50
            "AP@75": coco_eval.stats[2],       # AP at IoU=0.75
            "AP_small": coco_eval.stats[3],    # AP for small objects
            "AP_medium": coco_eval.stats[4],   # AP for medium objects
            "AP_large": coco_eval.stats[5],    # AP for large objects
            "AR@[50:95]": coco_eval.stats[6],  # Mean AR across IoU thresholds
            "AR@small": coco_eval.stats[7],    # AR for small objects
            "AR@medium": coco_eval.stats[8],   # AR for medium objects
            "AR@large": coco_eval.stats[9]     # AR for large objects
        }

        return results