import cv2
import numpy as np
from PIL import Image as PILImage
import supervision as sv
from typing import Optional, Dict

class SupervisionRenderer:
    """
    A renderer that uses the Supervision library to annotate images.
    
    For object detection, it uses the BoundingBoxAnnotator (and optionally a LabelAnnotator).
    For instance segmentation, it uses the MaskAnnotator.
    """
    def __init__(self, categories: Optional[Dict[int, str]] = None):
        """
        Args:
            categories (Optional[Dict[int, str]]): Mapping from category_id to label. 
                                                   Used when rendering text labels.
        """
        self.categories = categories or {}

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk (BGR format) using OpenCV."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")
        return image

    def render_detection(self, image_path: str, detections: sv.Detections) -> np.ndarray:
        """
        Renders an image with object detection annotations using BoundingBoxAnnotator.
        
        Args:
            image_path (str): Path to the source image.
            detections (sv.Detections): Detections object containing bounding boxes, class IDs, etc.
            
        Returns:
            np.ndarray: Annotated image in BGR format.
        """
        image = self._load_image(image_path)
        annotated = image.copy()

        # Use BoundingBoxAnnotator to draw boxes
        bbox_annotator = sv.BoundingBoxAnnotator(thickness=2)
        annotated = bbox_annotator.annotate(scene=annotated, detections=detections)
        
        # Optionally, add labels if a categories mapping was provided
        if self.categories:
            labels = [self.categories.get(cid, str(cid)) for cid in detections.class_id]
            label_annotator = sv.LabelAnnotator(text_color=sv.Color.white())
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        return annotated

    def render_segmentation(self, image_path: str, detections: sv.Detections) -> np.ndarray:
        """
        Renders an image with instance segmentation annotations using MaskAnnotator.
        
        Args:
            image_path (str): Path to the source image.
            detections (sv.Detections): Detections object containing segmentation masks.
            
        Returns:
            np.ndarray: Annotated image in BGR format.
        """
        image = self._load_image(image_path)
        annotated = image.copy()

        # Use MaskAnnotator to draw instance segmentation masks
        mask_annotator = sv.MaskAnnotator()
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)
        
        return annotated

    def to_pil(self, image_path: str, detections: sv.Detections, task: str = 'detection') -> PILImage:
        """
        Renders an annotated image (detection or segmentation) and returns it as a PIL Image (RGB).
        
        Args:
            image_path (str): Path to the source image.
            detections (sv.Detections): Detections object.
            task (str): 'detection' or 'segmentation' to select the annotator.
            
        Returns:
            PILImage: Annotated image in RGB mode.
        """
        if task == 'detection':
            annotated = self.render_detection(image_path, detections)
        elif task == 'segmentation':
            annotated = self.render_segmentation(image_path, detections)
        else:
            raise ValueError("Task must be either 'detection' or 'segmentation'")
        
        # Convert BGR (OpenCV format) to RGB (PIL format)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def save(self, image_path: str, detections: sv.Detections, filepath: str, task: str = 'detection') -> None:
        """
        Renders the annotated image and saves it to the specified file path.
        
        Args:
            image_path (str): Path to the source image.
            detections (sv.Detections): Detections object.
            filepath (str): Where to save the annotated image.
            task (str): 'detection' or 'segmentation' to select the annotator.
        """
        if task == 'detection':
            annotated = self.render_detection(image_path, detections)
        elif task == 'segmentation':
            annotated = self.render_segmentation(image_path, detections)
        else:
            raise ValueError("Task must be either 'detection' or 'segmentation'")
        
        cv2.imwrite(filepath, annotated)