from cvsdk.model import Dataset, Image, BoundingBox, SegmentationMask, PanopticSegment
import json


class CocoImporter:
    """Imports a COCO JSON file and converts it into the Dataset model."""

    @staticmethod
    def from_coco_json(json_path: str, task_type: str) -> Dataset:
        """Load a COCO JSON file and return a Dataset model."""
        with open(json_path, "r") as f:
            coco_data = json.load(f)

        categories = {c["id"]: c["name"] for c in coco_data["categories"]}

        images = {img["id"]: Image(
            id=img["id"],
            file_name=img["file_name"],
            width=img["width"],
            height=img["height"],
            bounding_boxes=[],
            segmentation_masks=[],
            panoptic_segments=[],
            labels=[]
        ) for img in coco_data["images"]}

        if task_type == "detection" or task_type == "segmentation":
            for ann in coco_data["annotations"]:
                if "bbox" in ann:
                    bbox = BoundingBox(
                        xmin=ann["bbox"][0],
                        ymin=ann["bbox"][1],
                        width=ann["bbox"][2],
                        height=ann["bbox"][3],
                        category_id=ann["category_id"],
                        id=ann["id"]
                    )
                    images[ann["image_id"]].bounding_boxes.append(bbox)

                if "segmentation" in ann and ann["segmentation"]:
                    mask = SegmentationMask(
                        segmentation=ann["segmentation"],
                        category_id=ann["category_id"],
                        id=ann["id"]
                    )
                    images[ann["image_id"]].segmentation_masks.append(mask)

        elif task_type == "panoptic":
            for ann in coco_data["annotations"]:
                mask = PanopticSegment(
                    segment_id=ann["id"],
                    category_id=ann["category_id"],
                    mask=ann["segmentation"]  # Assuming this contains the path to a mask
                )
                images[ann["image_id"]].panoptic_segments.append(mask)

        elif task_type == "classification":
            for ann in coco_data["annotations"]:
                images[ann["image_id"]].labels.append(ann["category_id"])

        return Dataset(images=list(images.values()), categories=categories, task_type=task_type)
