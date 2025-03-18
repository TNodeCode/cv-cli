from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json


class BoundingBox(BaseModel):
    """Represents a bounding box in an image."""
    xmin: float = Field(..., ge=0)
    ymin: float = Field(..., ge=0)
    width: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    category_id: int
    id: Optional[int] = None


class SegmentationMask(BaseModel):
    """Represents a segmentation mask as a list of polygons."""
    segmentation: List[List[float]]
    category_id: int
    id: Optional[int] = None


class PanopticSegment(BaseModel):
    """Represents a panoptic segmentation mask for an image."""
    segment_id: int
    category_id: int
    mask: str  # Path to the segmentation mask file


class Image(BaseModel):
    """Represents an image in the dataset."""
    id: int
    file_name: str
    width: int
    height: int
    labels: List[int] = []  # Image classification labels (category IDs)
    bounding_boxes: List[BoundingBox] = []
    segmentation_masks: List[SegmentationMask] = []
    panoptic_segments: List[PanopticSegment] = []


class Dataset(BaseModel):
    """Represents an entire dataset."""
    images: List[Image]
    categories: Dict[int, str]
    task_type: str  # "detection", "segmentation", "panoptic", "classification"