import pandas as pd
from cvsdk.model import Image, BoundingBox, Dataset


class ObjectDetectionImporter:
    """Imports an object detection dataset from a CSV file into a Pandas DataFrame or Dataset model."""

    @staticmethod
    def load_csv(csv_path: str) -> pd.DataFrame:
        """Load a CSV file into a Pandas DataFrame.
        
        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame with columns: filename, xmin, ymin, xmax, ymax, class.
        """
        df = pd.read_csv(csv_path)

        # Validate necessary columns
        required_columns = {"filename", "xmin", "ymin", "xmax", "ymax", "class"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV file is missing one or more required columns: {required_columns}")

        return df

    @staticmethod
    def from_csv(csv_path: str) -> Dataset:
        """Convert a CSV file into a Dataset model.
        
        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            Dataset: A Dataset model containing images and bounding boxes.
        """
        df = ObjectDetectionImporter.load_csv(csv_path)

        # Create categories dictionary
        unique_classes = df["class"].unique()
        category_map = {class_name: i + 1 for i, class_name in enumerate(unique_classes)}
        categories = {v: k for k, v in category_map.items()}  # Reverse mapping

        images_dict = {}

        for _, row in df.iterrows():
            file_name = row["filename"]
            bbox = BoundingBox(
                xmin=row["xmin"],
                ymin=row["ymin"],
                width=row["xmax"] - row["xmin"],
                height=row["ymax"] - row["ymin"],
                category_id=category_map[row["class"]]
            )

            if file_name not in images_dict:
                images_dict[file_name] = Image(
                    id=len(images_dict) + 1,
                    file_name=file_name,
                    width=0,  # Placeholder, can be updated later
                    height=0,  # Placeholder, can be updated later
                    bounding_boxes=[bbox]
                )
            else:
                images_dict[file_name].bounding_boxes.append(bbox)

        return Dataset(images=list(images_dict.values()), categories=categories, task_type="detection")
