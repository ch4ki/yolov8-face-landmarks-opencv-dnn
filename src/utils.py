"""Utility functions for YOLOv8 Face Detection and Tracking."""

import cv2
import numpy as np
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("yolov8_face")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_dirs(base_dir: str) -> dict:
    """
    Create output directories for different types of results.
    
    Args:
        base_dir: Base output directory
        
    Returns:
        Dictionary with created directory paths
    """
    base_path = Path(base_dir)
    dirs = {
        'base': base_path,
        'faces': base_path / 'faces',
        'aligned': base_path / 'aligned',
        'videos': base_path / 'videos',
        'logs': base_path / 'logs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {k: str(v) for k, v in dirs.items()}


def save_face_crops(image: np.ndarray, bboxes: np.ndarray, track_ids: List[int], 
                   output_dir: str, frame_id: int) -> List[str]:
    """
    Save cropped face images.
    
    Args:
        image: Original image
        bboxes: Face bounding boxes
        track_ids: Track IDs for each face
        output_dir: Output directory
        frame_id: Current frame ID
        
    Returns:
        List of saved file paths
    """
    saved_paths = []
    output_path = Path(output_dir)
    
    for i, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            face_crop = image[y1:y2, x1:x2]
            
            # Create track-specific directory
            track_dir = output_path / f"track_{track_id}"
            track_dir.mkdir(exist_ok=True)
            
            # Save face crop
            filename = f"frame_{frame_id:06d}_face_{i}.jpg"
            file_path = track_dir / filename
            cv2.imwrite(str(file_path), face_crop)
            saved_paths.append(str(file_path))
    
    return saved_paths


def load_image_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Load all image paths from a directory.
    
    Args:
        directory: Directory path
        extensions: List of valid extensions (default: common image formats)
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory_path.glob(f"*{ext}"))
        image_paths.extend(directory_path.glob(f"*{ext.upper()}"))
    
    return [str(path) for path in sorted(image_paths)]


def resize_image_keep_aspect(image: np.ndarray, target_size: Tuple[int, int], 
                           pad_color: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image while keeping aspect ratio and pad if necessary.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        pad_color: Padding color (BGR)
        
    Returns:
        Tuple of (resized_image, scale_factor, (pad_x, pad_y))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded, scale, (pad_x, pad_y)


def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Draw FPS counter on image.
    
    Args:
        image: Input image
        fps: FPS value
        position: Text position (x, y)
        
    Returns:
        Image with FPS counter
    """
    result = image.copy()
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(result, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return result


def calculate_face_area(bbox: np.ndarray) -> float:
    """
    Calculate face area from bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Face area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def filter_faces_by_size(bboxes: np.ndarray, scores: np.ndarray, 
                        min_area: float = 1000.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter faces by minimum area.
    
    Args:
        bboxes: Face bounding boxes
        scores: Detection scores
        min_area: Minimum face area
        
    Returns:
        Tuple of (filtered_bboxes, filtered_scores)
    """
    if len(bboxes) == 0:
        return bboxes, scores
    
    areas = np.array([calculate_face_area(bbox) for bbox in bboxes])
    valid_mask = areas >= min_area
    
    return bboxes[valid_mask], scores[valid_mask]


def non_max_suppression_custom(bboxes: np.ndarray, scores: np.ndarray, 
                              iou_threshold: float = 0.5) -> List[int]:
    """
    Custom Non-Maximum Suppression implementation.
    
    Args:
        bboxes: Bounding boxes [N, 4]
        scores: Detection scores [N]
        iou_threshold: IoU threshold
        
    Returns:
        List of indices to keep
    """
    if len(bboxes) == 0:
        return []
    
    # Calculate areas
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    
    # Sort by scores
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


class FPSCounter:
    """FPS counter utility class."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS counter and return current FPS.
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return 0.0
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.last_time = time.time()


class VideoWriter:
    """Enhanced video writer with automatic codec selection."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Video FPS
            frame_size: Frame size (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        
        # Try different codecs
        codecs = ['mp4v', 'XVID', 'MJPG', 'X264']
        
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
                if self.writer.isOpened():
                    break
                self.writer.release()
                self.writer = None
            except:
                continue
        
        if self.writer is None:
            raise RuntimeError(f"Failed to initialize video writer for {output_path}")
    
    def write(self, frame: np.ndarray):
        """Write frame to video."""
        if self.writer is not None:
            self.writer.write(frame)
    
    def release(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def validate_model_file(model_path: str) -> bool:
    """
    Validate that model file exists and is readable.
    
    Args:
        model_path: Path to model file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        path = Path(model_path)
        if not path.exists():
            return False
        
        if not path.is_file():
            return False
        
        if path.suffix.lower() not in ['.onnx', '.pt', '.pth']:
            return False
        
        # Try to read the file
        with open(path, 'rb') as f:
            f.read(1)
        
        return True
    except:
        return False


def get_available_cameras() -> List[int]:
    """
    Get list of available camera indices.
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
        cap.release()
    
    return available_cameras