"""YOLOv8 Face Detection and Tracking Package."""

__version__ = "0.2.0"
__author__ = "YOLOv8 Face Team"

from .detection import YOLOv8Face
from .tracking import FaceTracker
from .quality import FaceQualityAssessment, FaceQualityLightQnet
from .alignment import FaceAligner

__all__ = [
    "YOLOv8Face",
    "FaceTracker", 
    "FaceQualityAssessment",
    "FaceQualityLightQnet"
    "FaceAligner"
]