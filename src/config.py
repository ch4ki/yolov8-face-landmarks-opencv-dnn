"""Configuration management for YOLOv8 Face Detection and Tracking."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class DetectionConfig:
    """Configuration for face detection."""
    model_path: str = "weights/yolov8n-face.onnx"
    conf_threshold: float = 0.2
    iou_threshold: float = 0.5
    input_size: tuple = (640, 640)


@dataclass
class TrackingConfig:
    """Configuration for face tracking."""
    track_threshold: float = 0.5
    track_buffer: int = 30
    match_threshold: float = 0.8
    frame_rate: int = 30
    max_history_length: int = 30


@dataclass
class QualityConfig:
    """Configuration for face quality assessment."""
    model_path: str = "weights/face-quality-assessment.onnx"
    min_quality_threshold: float = 0.5
    input_size: tuple = (112, 112)


@dataclass
class AlignmentConfig:
    """Configuration for face alignment."""
    template_mode: str = "arcface"  # "arcface" or "default"
    template_scale: Optional[float] = None
    output_size: int = 112
    allow_upscale: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    detection: DetectionConfig
    tracking: TrackingConfig
    quality: QualityConfig
    alignment: AlignmentConfig
    
    # Paths
    weights_dir: str = "weights"
    images_dir: str = "images"
    output_dir: str = "output"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create configuration from dictionary."""
        detection_config = DetectionConfig(**config_dict.get('detection', {}))
        tracking_config = TrackingConfig(**config_dict.get('tracking', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))
        alignment_config = AlignmentConfig(**config_dict.get('alignment', {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['detection', 'tracking', 'quality', 'alignment']}
        
        return cls(
            detection=detection_config,
            tracking=tracking_config,
            quality=quality_config,
            alignment=alignment_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'detection': {
                'model_path': self.detection.model_path,
                'conf_threshold': self.detection.conf_threshold,
                'iou_threshold': self.detection.iou_threshold,
                'input_size': self.detection.input_size,
            },
            'tracking': {
                'track_threshold': self.tracking.track_threshold,
                'track_buffer': self.tracking.track_buffer,
                'match_threshold': self.tracking.match_threshold,
                'frame_rate': self.tracking.frame_rate,
                'max_history_length': self.tracking.max_history_length,
            },
            'quality': {
                'model_path': self.quality.model_path,
                'min_quality_threshold': self.quality.min_quality_threshold,
                'input_size': self.quality.input_size,
            },
            'alignment': {
                'template_mode': self.alignment.template_mode,
                'template_scale': self.alignment.template_scale,
                'output_size': self.alignment.output_size,
                'allow_upscale': self.alignment.allow_upscale,
            },
            'weights_dir': self.weights_dir,
            'images_dir': self.images_dir,
            'output_dir': self.output_dir,
            'log_level': self.log_level,
            'log_file': self.log_file,
        }
    
    def validate_paths(self) -> None:
        """Validate that required paths exist."""
        # Check weights directory
        weights_path = Path(self.weights_dir)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights directory not found: {weights_path}")
        
        # Check detection model
        detection_model_path = Path(self.detection.model_path)
        if not detection_model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {detection_model_path}")
        
        # Check quality model if it's being used
        quality_model_path = Path(self.quality.model_path)
        if not quality_model_path.exists():
            print(f"Warning: Quality assessment model not found: {quality_model_path}")
        
        # Create output directory if it doesn't exist
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)


def get_default_config() -> AppConfig:
    """Get default configuration."""
    return AppConfig(
        detection=DetectionConfig(),
        tracking=TrackingConfig(),
        quality=QualityConfig(),
        alignment=AlignmentConfig()
    )


def load_config_from_file(config_path: str) -> AppConfig:
    """Load configuration from JSON or YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.json':
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    return AppConfig.from_dict(config_dict)


def save_config_to_file(config: AppConfig, config_path: str) -> None:
    """Save configuration to JSON or YAML file."""
    config_path = Path(config_path)
    config_dict = config.to_dict()
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix.lower() == '.json':
        import json
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif config_path.suffix.lower() in ['.yml', '.yaml']:
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML configuration files")
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


# Environment variable overrides
def get_config_from_env() -> Dict[str, Any]:
    """Get configuration overrides from environment variables."""
    env_config = {}
    
    # Detection config
    if os.getenv('YOLO_MODEL_PATH'):
        env_config.setdefault('detection', {})['model_path'] = os.getenv('YOLO_MODEL_PATH')
    if os.getenv('YOLO_CONF_THRESHOLD'):
        env_config.setdefault('detection', {})['conf_threshold'] = float(os.getenv('YOLO_CONF_THRESHOLD'))
    if os.getenv('YOLO_IOU_THRESHOLD'):
        env_config.setdefault('detection', {})['iou_threshold'] = float(os.getenv('YOLO_IOU_THRESHOLD'))
    
    # Tracking config
    if os.getenv('TRACK_THRESHOLD'):
        env_config.setdefault('tracking', {})['track_threshold'] = float(os.getenv('TRACK_THRESHOLD'))
    if os.getenv('TRACK_BUFFER'):
        env_config.setdefault('tracking', {})['track_buffer'] = int(os.getenv('TRACK_BUFFER'))
    if os.getenv('MATCH_THRESHOLD'):
        env_config.setdefault('tracking', {})['match_threshold'] = float(os.getenv('MATCH_THRESHOLD'))
    
    # Quality config
    if os.getenv('FQA_MODEL_PATH'):
        env_config.setdefault('quality', {})['model_path'] = os.getenv('FQA_MODEL_PATH')
    if os.getenv('MIN_QUALITY_THRESHOLD'):
        env_config.setdefault('quality', {})['min_quality_threshold'] = float(os.getenv('MIN_QUALITY_THRESHOLD'))
    
    # Paths
    if os.getenv('WEIGHTS_DIR'):
        env_config['weights_dir'] = os.getenv('WEIGHTS_DIR')
    if os.getenv('OUTPUT_DIR'):
        env_config['output_dir'] = os.getenv('OUTPUT_DIR')
    
    return env_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged