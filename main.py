#!/usr/bin/env python3
"""
Main entry point for YOLOv8 Face Detection and Tracking.
Provides a unified interface for all functionality.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_default_config, load_config_from_file
from utils import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Face Detection and Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect faces in image
  python main.py detect --image images/test.jpg
  
  # Track faces in video
  python main.py track --input video.mp4 --output tracked_video.mp4
  
  # Assess face quality
  python main.py quality --input images/ --top-k 10
  
  # Align faces
  python main.py align --input images/ --output-dir output/aligned
  
  # Run with custom config
  python main.py detect --image test.jpg --config config.yaml
        """
    )
    
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Detection command
    detect_parser = subparsers.add_parser("detect", help="Face detection on image")
    detect_parser.add_argument("--image", type=str, required=True, help="Input image path")
    detect_parser.add_argument("--model", type=str, help="Model path (overrides config)")
    detect_parser.add_argument("--conf", type=float, help="Confidence threshold (overrides config)")
    detect_parser.add_argument("--output", type=str, help="Output image path")
    detect_parser.add_argument("--show", action="store_true", help="Show result window")
    
    # Tracking command
    track_parser = subparsers.add_parser("track", help="Face tracking on video")
    track_parser.add_argument("--input", type=str, required=True, help="Input video path or camera index")
    track_parser.add_argument("--output", type=str, help="Output video path")
    track_parser.add_argument("--model", type=str, help="Model path (overrides config)")
    track_parser.add_argument("--conf", type=float, help="Detection confidence (overrides config)")
    track_parser.add_argument("--track-thresh", type=float, help="Tracking threshold (overrides config)")
    track_parser.add_argument("--show", action="store_true", help="Show video window")
    track_parser.add_argument("--save-faces", action="store_true", help="Save face crops")
    
    # Quality assessment command
    quality_parser = subparsers.add_parser("quality", help="Face quality assessment")
    quality_parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
    quality_parser.add_argument("--detection-model", type=str, help="Detection model (overrides config)")
    quality_parser.add_argument("--quality-model", type=str, help="Quality model (overrides config)")
    quality_parser.add_argument("--align", action="store_true", help="Use face alignment")
    quality_parser.add_argument("--show", action="store_true", help="Show results")
    quality_parser.add_argument("--top-k", type=int, default=10, help="Show top K results")
    quality_parser.add_argument("--output", type=str, help="Output CSV file")
    
    # Alignment command
    align_parser = subparsers.add_parser("align", help="Face alignment and cropping")
    align_parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
    align_parser.add_argument("--output-dir", type=str, help="Output directory (overrides config)")
    align_parser.add_argument("--model", type=str, help="Detection model (overrides config)")
    align_parser.add_argument("--template-mode", type=str, choices=["arcface", "default"],
                             help="Alignment template mode (overrides config)")
    align_parser.add_argument("--output-size", type=int, help="Output face size (overrides config)")
    align_parser.add_argument("--show", action="store_true", help="Show results")
    align_parser.add_argument("--create-grid", action="store_true", help="Create alignment grid")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    try:
        # Load configuration
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config_from_file(args.config)
        else:
            logger.info("Using default configuration")
            config = get_default_config()
        
        # Validate paths
        config.validate_paths()
        
        # Execute command
        if args.command == "detect":
            from examples.detect_image import main as detect_main
            
            # Build arguments for detect script
            detect_args = [
                "--image", args.image,
                "--model", args.model or config.detection.model_path,
                "--conf", str(args.conf or config.detection.conf_threshold),
                "--iou", str(config.detection.iou_threshold)
            ]
            
            if args.output:
                detect_args.extend(["--output", args.output])
            if args.show:
                detect_args.append("--show")
            
            # Override sys.argv for the detect script
            original_argv = sys.argv
            sys.argv = ["detect_image.py"] + detect_args
            
            try:
                return detect_main()
            finally:
                sys.argv = original_argv
        
        elif args.command == "track":
            from examples.track_video import main as track_main
            
            # Build arguments for track script
            track_args = [
                "--input", args.input,
                "--model", args.model or config.detection.model_path,
                "--conf", str(args.conf or config.detection.conf_threshold),
                "--track-thresh", str(args.track_thresh or config.tracking.track_threshold),
                "--track-buffer", str(config.tracking.track_buffer),
                "--match-thresh", str(config.tracking.match_threshold),
                "--output-dir", config.output_dir
            ]
            
            if args.output:
                track_args.extend(["--output", args.output])
            if args.show:
                track_args.append("--show")
            if args.save_faces:
                track_args.append("--save-faces")
            
            # Override sys.argv for the track script
            original_argv = sys.argv
            sys.argv = ["track_video.py"] + track_args
            
            try:
                return track_main()
            finally:
                sys.argv = original_argv
        
        elif args.command == "quality":
            from examples.quality_assessment import main as quality_main
            
            # Build arguments for quality script
            quality_args = [
                "--input", args.input,
                "--detection-model", args.detection_model or config.detection.model_path,
                "--quality-model", args.quality_model or config.quality.model_path,
                "--conf", str(config.detection.conf_threshold),
                "--top-k", str(args.top_k)
            ]
            
            if args.align:
                quality_args.append("--align")
            if args.show:
                quality_args.append("--show")
            if args.output:
                quality_args.extend(["--output", args.output])
            
            # Override sys.argv for the quality script
            original_argv = sys.argv
            sys.argv = ["quality_assessment.py"] + quality_args
            
            try:
                return quality_main()
            finally:
                sys.argv = original_argv
        
        elif args.command == "align":
            from examples.align_faces import main as align_main
            
            # Build arguments for align script
            align_args = [
                "--input", args.input,
                "--output-dir", args.output_dir or config.output_dir + "/aligned",
                "--model", args.model or config.detection.model_path,
                "--conf", str(config.detection.conf_threshold),
                "--template-mode", args.template_mode or config.alignment.template_mode,
                "--output-size", str(args.output_size or config.alignment.output_size)
            ]
            
            if config.alignment.allow_upscale:
                align_args.append("--allow-upscale")
            if config.alignment.template_scale:
                align_args.extend(["--template-scale", str(config.alignment.template_scale)])
            if args.show:
                align_args.append("--show")
            if args.create_grid:
                align_args.append("--create-grid")
            
            # Override sys.argv for the align script
            original_argv = sys.argv
            sys.argv = ["align_faces.py"] + align_args
            
            try:
                return align_main()
            finally:
                sys.argv = original_argv
        
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
