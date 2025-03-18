import pandas as pd
from cvsdk.model import Dataset


class ObjectDetectionExporter:
    """Exports an object detection dataset into a Pandas DataFrame."""

    @staticmethod
    def to_df(dataset: Dataset) -> pd.DataFrame:
        """Convert a Dataset model into a Pandas DataFrame.
        
        Args:
            dataset (Dataset): The dataset containing images and bounding boxes.

        Returns:
            pd.DataFrame: A DataFrame with columns: filename, xmin, ymin, xmax, ymax, class.
        """
        data = []

        for img in dataset.images:
            for bbox in img.bounding_boxes:
                data.append({
                    "filename": img.file_name,
                    "xmin": bbox.xmin,
                    "ymin": bbox.ymin,
                    "xmax": bbox.xmin + bbox.width,
                    "ymax": bbox.ymin + bbox.height,
                    "class": dataset.categories.get(bbox.category_id, f"class_{bbox.category_id}")
                })

        return pd.DataFrame(data, columns=["filename", "xmin", "ymin", "xmax", "ymax", "class"])

    @staticmethod
    def to_csv(dataset: Dataset, output_path: str) -> None:
        """Save the dataset as a CSV file.
        
        Args:
            dataset (Dataset): The dataset containing images and bounding boxes.
            output_path (str): The path to save the CSV file.
        """
        df = ObjectDetectionExporter.to_df(dataset)
        df.to_csv(output_path, index=False)
