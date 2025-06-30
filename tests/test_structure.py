"""Test the new code structure and imports."""

import sys
import pytest
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all modules can be imported correctly."""
    try:
        from detection import YOLOv8Face
        from tracking import FaceTracker
        from quality import FaceQualityAssessment
        from alignment import FaceAligner
        from config import get_default_config, AppConfig
        from utils import setup_logging, FPSCounter
        
        # Test that classes can be instantiated (without actual models)
        assert YOLOv8Face is not None
        assert FaceTracker is not None
        assert FaceQualityAssessment is not None
        assert FaceAligner is not None
        
        # Test config
        config = get_default_config()
        assert isinstance(config, AppConfig)
        assert config.detection.conf_threshold == 0.2
        
        # Test utils
        logger = setup_logging("INFO")
        assert logger is not None
        
        fps_counter = FPSCounter()
        assert fps_counter is not None
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_package_imports():
    """Test package-level imports."""
    try:
        import src
        from src import YOLOv8Face, FaceTracker, FaceQualityAssessment, FaceAligner
        
        assert YOLOv8Face is not None
        assert FaceTracker is not None
        assert FaceQualityAssessment is not None
        assert FaceAligner is not None
        
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")


def test_config_validation():
    """Test configuration validation."""
    from config import get_default_config
    
    config = get_default_config()
    
    # Test config structure
    assert hasattr(config, 'detection')
    assert hasattr(config, 'tracking')
    assert hasattr(config, 'quality')
    assert hasattr(config, 'alignment')
    
    # Test detection config
    assert config.detection.conf_threshold > 0
    assert config.detection.iou_threshold > 0
    assert config.detection.model_path.endswith('.onnx')
    
    # Test tracking config
    assert config.tracking.track_threshold > 0
    assert config.tracking.track_buffer > 0
    assert config.tracking.frame_rate > 0
    
    # Test quality config
    assert config.quality.model_path.endswith('.onnx')
    assert config.quality.min_quality_threshold >= 0
    
    # Test alignment config
    assert config.alignment.template_mode in ['arcface', 'default']
    assert config.alignment.output_size > 0


def test_utils_functions():
    """Test utility functions."""
    import numpy as np
    from utils import calculate_face_area, filter_faces_by_size, FPSCounter
    
    # Test face area calculation
    bbox = np.array([10, 10, 50, 50])
    area = calculate_face_area(bbox)
    assert area == 1600  # (50-10) * (50-10)
    
    # Test face filtering
    bboxes = np.array([[10, 10, 50, 50], [0, 0, 10, 10]])  # Areas: 1600, 100
    scores = np.array([0.9, 0.8])
    
    filtered_bboxes, filtered_scores = filter_faces_by_size(bboxes, scores, min_area=500)
    assert len(filtered_bboxes) == 1  # Only first face should remain
    assert len(filtered_scores) == 1
    
    # Test FPS counter
    fps_counter = FPSCounter(window_size=5)
    fps = fps_counter.update()
    assert fps >= 0


if __name__ == "__main__":
    test_imports()
    test_package_imports()
    test_config_validation()
    test_utils_functions()
    print("All tests passed!")