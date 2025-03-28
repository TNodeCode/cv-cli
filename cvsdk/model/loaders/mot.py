import os
import csv
import shutil
from collections import defaultdict
from typing import List
from pathlib import Path
from cvsdk.model import Image, BoundingBox, Dataset
from PIL import Image as PILImage


class MOTLoader:
    """Imports MOT-format annotations and constructs a Dataset model."""

    @staticmethod
    def import_dataset(base_dir: str, categories: dict[int, str]) -> Dataset:
        """Convert multiple MOT17-format sequences to a Dataset.

        Args:
            base_dir (str): Path to the base directory containing sequence folders.
            categories (dict[int, str]): Dictionary of category IDs to names.

        Returns:
            Dataset: Pydantic model representing the dataset.
        """
        images = []
        base_path = Path(base_dir)

        # Iterate over each sequence directory
        for sequence_dir in base_path.iterdir():
            if sequence_dir.is_dir():
                stack_name = sequence_dir.name
                images_dir = sequence_dir / "img1"
                annotations_path = sequence_dir / "gt" / "gt.txt"

                if images_dir.exists() and annotations_path.exists():
                    sequence_images = MOTImporter._process_sequence(
                        images_dir, annotations_path, categories, stack_name
                    )
                    images.extend(sequence_images)

        return Dataset(
            images=images,
            categories=categories,
            task_type="tracking"
        )
    
    def export_dataset(dataset: Dataset, output_dir: str) -> None:
        """
        Export the Dataset model to the MOT17 dataset format.

        Args:
            dataset (Dataset): The dataset to export.
            output_dir (str): The directory where the MOT17 formatted dataset will be saved.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Organize images by stack
        stacks: dict[str, List[Image]] = {}
        for image in dataset.images:
            if image.stack not in stacks:
                stacks[image.stack] = []
            stacks[image.stack].append(image)

        for stack_name, images in stacks.items():
            stack_dir = output_path / stack_name
            img1_dir = stack_dir / 'img1'
            gt_dir = stack_dir / 'gt'
            img1_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            # Copy images to img1 directory
            for image in images:
                src_image_path = Path(image.file_name)
                dst_image_path = img1_dir / src_image_path.name
                shutil.copy(src_image_path, dst_image_path)

            # Create gt.txt file
            gt_file_path = gt_dir / 'gt.txt'
            with gt_file_path.open('w') as gt_file:
                for image in images:
                    frame_number = int(Path(image.file_name).stem)
                    for bbox in image.bounding_boxes:
                        gt_file.write(f"{frame_number},{bbox.id},{bbox.xmin},{bbox.ymin},{bbox.width},{bbox.height},1,-1,-1,-1\n")

            # Create seqinfo.ini file
            seqinfo_path = stack_dir / 'seqinfo.ini'
            with seqinfo_path.open('w') as seqinfo_file:
                seqinfo_file.write(f"[Sequence]\n")
                seqinfo_file.write(f"name={stack_name}\n")
                seqinfo_file.write(f"imDir=img1\n")
                seqinfo_file.write(f"frameRate=30\n")  # Adjust as necessary
                seqinfo_file.write(f"seqLength={len(images)}\n")
                seqinfo_file.write(f"imWidth={images[0].width}\n")
                seqinfo_file.write(f"imHeight={images[0].height}\n")
                seqinfo_file.write(f"imExt=.jpg\n")
    
    @staticmethod
    def _process_sequence(images_dir: str, annotations_path: str, categories: dict[int, str], stack_name: str) -> list[Image]:
        """Process a single MOT sequence.

        Args:
            images_dir (str): Path to the folder containing images (i.e. "./MOT17/train/MOT17-02/img1").
            annotations_path (str): Path to the annotations .txt file (MOT format) (i.e. ./MOT17/train/MOT17-02/gt/gt.txt).
            categories (dict[int, str]): Dictionary of category IDs to names (i.e. {1: "cat", 2: "dog"}).
            stack_name (str): Name of the stack

        Returns:
            List[Image]: List of Image models for the sequence.
        """
        # Organize frames by frame index
        # Organize frames by frame index
        images_dict = {}
        frame_to_files = {int(p.stem): p.name for p in images_dir.glob("*.jpg")}

        # Read annotations and group by frame
        annotations_by_frame = defaultdict(list)

        with open(annotations_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                frame, obj_id, left, top, width, height, conf, cls, vis = row
                frame = int(frame)
                obj_id = int(obj_id)
                left = float(left)
                top = float(top)
                width = float(width)
                height = float(height)
                category_id = int(cls)

                if obj_id == -1:
                    continue  # Skip ignored regions

                bbox = BoundingBox(
                    xmin=left,
                    ymin=top,
                    width=width,
                    height=height,
                    category_id=category_id,
                    id=obj_id
                )
                annotations_by_frame[frame].append(bbox)

        images = []
        for frame, bboxes in annotations_by_frame.items():
            if frame not in frame_to_files:
                continue

            file_name = frame_to_files[frame]
            image_path = images_dir / file_name

            # Read image size
            with PILImage.open(image_path) as img:
                width, height = img.size

            image = Image(
                id=frame,
                file_name=str(image_path),
                width=width,
                height=height,
                stack=stack_name,
                bounding_boxes=bboxes
            )
            images.append(image)

        return images