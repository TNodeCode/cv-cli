from cvsdk.model import Dataset
import json


class CocoExporter:
    """Exports a Dataset model to a COCO JSON file."""

    @staticmethod
    def to_coco_dict(dataset: Dataset) -> dict:
        """Convert Dataset model to COCO format."""
        coco_dict = {
            "images": [
                {"id": img.id, "file_name": img.file_name, "width": img.width, "height": img.height}
                for img in dataset.images
            ],
            "annotations": [],
            "categories": [{"id": cat_id, "name": name} for cat_id, name in dataset.categories.items()]
        }

        annotation_id = 1
        for img in dataset.images:
            if dataset.task_type == "detection" or dataset.task_type == "segmentation":
                for bbox in img.bounding_boxes:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": bbox.category_id,
                        "bbox": [bbox.xmin, bbox.ymin, bbox.width, bbox.height],
                        "area": bbox.width * bbox.height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

                for mask in img.segmentation_masks:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": mask.category_id,
                        "segmentation": mask.segmentation,
                        "area": sum([sum(mask.segmentation[i]) for i in range(len(mask.segmentation))]),
                        "iscrowd": 0
                    })
                    annotation_id += 1

            elif dataset.task_type == "panoptic":
                for mask in img.panoptic_segments:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": mask.category_id,
                        "segmentation": mask.mask  # Assuming this contains the path to a mask
                    })
                    annotation_id += 1

            elif dataset.task_type == "classification":
                for label in img.labels:
                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": img.id,
                        "category_id": label
                    })
                    annotation_id += 1
            return coco_dict


    @staticmethod
    def to_coco_file(dataset: Dataset, output_path: str) -> None:
        """Export dataset to COCO file.

        Args:
            dataset (Dataset): the dataset that should be exported
            output_path (str): the path to the coco file
        """
        coco_dict = CocoExporter.to_coco_dict(dataset);
        with open(output_path, "w") as f:
            json.dump(coco_dict, f, indent=4)