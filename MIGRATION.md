# Migration Guide

This document helps you migrate from the old code structure to the new modular architecture.

## 🔄 Overview of Changes

The project has been restructured for better maintainability, modularity, and extensibility:

### Old Structure
```
├── main.py                  # YOLOv8_face detection class
├── face_tracker.py          # Complete tracking system
├── cropper.py              # Face alignment
├── main_fqa.py             # Face quality assessment
├── fqa.py                  # Quality assessment script
└── aligner_test.py         # Test script
```

### New Structure
```
├── src/                          # Core modules
│   ├── detection.py              # Face detection (YOLOv8Face)
│   ├── tracking.py               # Face tracking (FaceTracker)
│   ├── quality.py                # Quality assessment (FaceQualityAssessment)
│   ├── alignment.py              # Face alignment (FaceAligner)
│   ├── config.py                 # Configuration management
│   └── utils.py                  # Utility functions
├── examples/                     # Example scripts
│   ├── detect_image.py           # Single image detection
│   ├── track_video.py            # Video tracking
│   ├── quality_assessment.py     # Quality assessment
│   └── align_faces.py            # Face alignment
├── main.py                       # Unified CLI interface
└── config.yaml                   # Configuration file
```

## 📝 Code Migration Examples

### 1. Face Detection

**Old Code:**
```python
from main import YOLOv8_face

detector = YOLOv8_face("weights/yolov8n-face.onnx", conf_thres=0.45)
boxes, scores, classids, kpts = detector.detect(image)
result = detector.draw_detections(image, boxes, scores, kpts)
```

**New Code:**
```python
from src.detection import YOLOv8Face

detector = YOLOv8Face("weights/yolov8n-face.onnx", conf_thres=0.45)
boxes, scores, classids, kpts = detector.detect(image)
result = detector.draw_detections(image, boxes, scores, kpts)
```

### 2. Face Tracking

**Old Code:**
```python
from face_tracker import FaceTracker

tracker = FaceTracker("weights/yolov8n-face.onnx")
tracked_faces, landmarks = tracker.detect_and_track(frame)
result_frame = tracker.draw_tracks(frame, tracked_faces)
```

**New Code:**
```python
from src.tracking import FaceTracker

tracker = FaceTracker("weights/yolov8n-face.onnx")
tracked_faces, landmarks = tracker.detect_and_track(frame)
result_frame = tracker.draw_tracks(frame, tracked_faces)
```

### 3. Face Quality Assessment

**Old Code:**
```python
from main_fqa import face_quality_assessment

fqa = face_quality_assessment("weights/face-quality-assessment.onnx")
quality_scores = fqa.detect(face_crop)
```

**New Code:**
```python
from src.quality import FaceQualityAssessment

fqa = FaceQualityAssessment("weights/face-quality-assessment.onnx")
quality_score = fqa.get_quality_score(face_crop)
# or for raw scores:
quality_scores = fqa.assess_quality(face_crop)
```

### 4. Face Alignment

**Old Code:**
```python
from cropper import FaceRecImageCropperAndAligner

aligner = FaceRecImageCropperAndAligner()
aligned_face, bbox, landmarks = aligner.crop_and_align_by_mat(
    image, det, landmarks, template_mode="arcface", image_size=112
)
```

**New Code:**
```python
from src.alignment import FaceAligner

aligner = FaceAligner()
aligned_face, bbox, landmarks = aligner.align_face(
    image, det, landmarks, template_mode="arcface", image_size=112
)
```

## 🚀 Command Line Migration

### Old Commands

**Detection:**
```bash
python main.py --imgpath images/test.jpg --confThreshold 0.45
```

**Tracking:**
```bash
python face_tracker.py video video.mp4
```

**Quality Assessment:**
```bash
python fqa.py --imgfolder images/
```

### New Commands

**Detection:**
```bash
# Unified interface
python main.py detect --image images/test.jpg --conf 0.45 --show

# Or dedicated script
python examples/detect_image.py --image images/test.jpg --conf 0.45 --show
```

**Tracking:**
```bash
# Unified interface
python main.py track --input video.mp4 --output tracked.mp4 --show

# Or dedicated script
python examples/track_video.py --input video.mp4 --output tracked.mp4 --show
```

**Quality Assessment:**
```bash
# Unified interface
python main.py quality --input images/ --top-k 10

# Or dedicated script
python examples/quality_assessment.py --input images/ --top-k 10
```

**Face Alignment:**
```bash
# Unified interface
python main.py align --input images/ --output-dir output/aligned

# Or dedicated script
python examples/align_faces.py --input images/ --output-dir output/aligned
```

## ⚙️ Configuration Migration

### Old Configuration
Configuration was hardcoded in each script.

### New Configuration
Use YAML configuration files:

```yaml
# config.yaml
detection:
  model_path: "weights/yolov8n-face.onnx"
  conf_threshold: 0.2
  iou_threshold: 0.5

tracking:
  track_threshold: 0.5
  track_buffer: 30
  match_threshold: 0.8

quality:
  model_path: "weights/face-quality-assessment.onnx"
  min_quality_threshold: 0.5

alignment:
  template_mode: "arcface"
  output_size: 112
```

Load configuration in code:
```python
from src.config import load_config_from_file

config = load_config_from_file("config.yaml")
detector = YOLOv8Face(config.detection.model_path, 
                     conf_thres=config.detection.conf_threshold)
```

## 🔧 Advanced Migration

### Custom Integration

**Old Code:**
```python
# Custom script using multiple components
from main import YOLOv8_face
from face_tracker import FaceTracker
from main_fqa import face_quality_assessment
from cropper import FaceRecImageCropperAndAligner

detector = YOLOv8_face("weights/yolov8n-face.onnx")
tracker = FaceTracker("weights/yolov8n-face.onnx")
fqa = face_quality_assessment("weights/face-quality-assessment.onnx")
aligner = FaceRecImageCropperAndAligner()
```

**New Code:**
```python
# Custom script using new modular components
from src import YOLOv8Face, FaceTracker, FaceQualityAssessment, FaceAligner

detector = YOLOv8Face("weights/yolov8n-face.onnx")
tracker = FaceTracker("weights/yolov8n-face.onnx")
fqa = FaceQualityAssessment("weights/face-quality-assessment.onnx")
aligner = FaceAligner()
```

### Package Installation

**Old Method:**
```bash
pip install -r requirements.txt  # If available
```

**New Method:**
```bash
# Install as package
pip install -e .

# Or with optional dependencies
pip install -e ".[all]"
```

## 🛠️ Backward Compatibility

The old files are still present but deprecated:

- `main.py` → Now unified CLI interface
- `face_tracker.py` → Deprecated wrapper with warnings
- `cropper.py` → Deprecated wrapper with warnings
- `main_fqa.py` → Deprecated wrapper with warnings
- `fqa.py` → Deprecated wrapper with warnings
- `legacy_main.py` → Old main.py functionality

These files will show deprecation warnings and guide you to the new structure.

## 🧪 Testing Migration

Test your migration with the new structure:

```bash
# Run tests
python -m pytest tests/

# Test imports
python tests/test_structure.py

# Test examples
python examples/detect_image.py --image images/test.jpg --show
python examples/track_video.py --input 0 --show  # Webcam test
```

## 📚 Additional Resources

- **New Documentation:** See updated [README.md](README.md)
- **Configuration:** See [config.yaml](config.yaml) for all options
- **Examples:** Check the [examples/](examples/) directory
- **API Reference:** All classes have comprehensive docstrings

## 🆘 Need Help?

If you encounter issues during migration:

1. Check the deprecation warnings for guidance
2. Review the examples in the `examples/` directory
3. Test with the unified CLI: `python main.py --help`
4. Open an issue if you find bugs or need assistance

## 📈 Benefits of New Structure

- **Modularity:** Each component is independent and reusable
- **Maintainability:** Clear separation of concerns
- **Extensibility:** Easy to add new features
- **Configuration:** Centralized configuration management
- **Testing:** Better test coverage and structure
- **Documentation:** Comprehensive documentation and examples
- **Performance:** Optimized imports and reduced dependencies