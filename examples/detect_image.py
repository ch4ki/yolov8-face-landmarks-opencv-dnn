#!/usr/bin/env python3
"""
Example: Face detection on a single image.
"""

import argparse
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection import YOLOv8Face
from config import get_default_config
from utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Face detection on image")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--model", type=str, default="weights/yolov8n-face.onnx", help="Model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output", type=str, help="Output image path (optional)")
    parser.add_argument("--show", action="store_true", help="Show result window")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    
    try:
        # Initialize detector
        logger.info(f"Loading model: {args.model}")
        detector = YOLOv8Face(args.model, conf_thres=args.conf, iou_thres=args.iou)
        
        # Load image
        logger.info(f"Loading image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            raise ValueError(f"Could not load image: {args.image}")
        
        logger.info(f"Image shape: {image.shape}")
        
        # Detect faces
        logger.info("Running face detection...")
        boxes, scores, class_ids, landmarks = detector.detect(image)
        
        logger.info(f"Found {len(boxes)} faces")
        
        # Draw results
        result_image = detector.draw_detections(image, boxes, scores, landmarks)
        
        # Print detection results
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x, y, w, h = box.astype(int)
            logger.info(f"Face {i+1}: bbox=({x}, {y}, {w}, {h}), confidence={score:.3f}")
        
        # Save or show result
        if args.output:
            cv2.imwrite(args.output, result_image)
            logger.info(f"Result saved to: {args.output}")
        
        if args.show:
            cv2.namedWindow("Face Detection Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Face Detection Result", result_image)
            logger.info("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        if not args.output and not args.show:
            # Default: save to same directory as input
            output_path = Path(args.image).with_suffix("_result.jpg")
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"Result saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())