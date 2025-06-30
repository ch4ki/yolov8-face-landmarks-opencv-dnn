#!/usr/bin/env python3
"""
Example: Face quality assessment on images or video.
"""

import argparse
import cv2
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection import YOLOv8Face
from quality import FaceQualityAssessment
from alignment import FaceAligner
from utils import setup_logging, load_image_paths


def assess_single_image(image_path: str, detector: YOLOv8Face, fqa: FaceQualityAssessment, 
                       aligner: FaceAligner = None, show_results: bool = False):
    """Assess quality of faces in a single image."""
    logger = setup_logging()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return []
    
    # Detect faces
    boxes, scores, class_ids, landmarks = detector.detect(image)
    
    if len(boxes) == 0:
        logger.info(f"No faces detected in {image_path}")
        return []
    
    results = []
    result_image = image.copy()
    
    for i, (box, score, landmark) in enumerate(zip(boxes, scores, landmarks)):
        x, y, w, h = box.astype(int)
        
        # Extract face crop
        face_crop = image[y:y+h, x:x+w]
        
        # Align face if aligner is provided
        if aligner is not None and len(landmark) >= 10:
            try:
                bbox_xyxy = [x, y, x+w, y+h]
                aligned_face, _, _ = aligner.align_face(image, bbox_xyxy, landmark)
                face_crop = aligned_face
            except Exception as e:
                logger.warning(f"Face alignment failed for face {i}: {e}")
        
        # Assess quality
        try:
            quality_score = fqa.get_quality_score(face_crop)
            results.append({
                'face_id': i,
                'bbox': box,
                'detection_score': score,
                'quality_score': quality_score,
                'landmarks': landmark
            })
            
            # Draw results on image
            color = (0, 255, 0) if quality_score > 0.5 else (0, 0, 255)
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
            
            label = f"Face {i}: Det={score:.2f}, Qual={quality_score:.2f}"
            cv2.putText(result_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            logger.info(f"Face {i}: detection={score:.3f}, quality={quality_score:.3f}")
            
        except Exception as e:
            logger.error(f"Quality assessment failed for face {i}: {e}")
    
    if show_results and len(results) > 0:
        cv2.namedWindow("Face Quality Assessment", cv2.WINDOW_NORMAL)
        cv2.imshow("Face Quality Assessment", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def assess_folder(folder_path: str, detector: YOLOv8Face, fqa: FaceQualityAssessment, 
                 aligner: FaceAligner = None, top_k: int = 10):
    """Assess quality of faces in all images in a folder."""
    logger = setup_logging()
    
    try:
        image_paths = load_image_paths(folder_path)
        logger.info(f"Found {len(image_paths)} images in {folder_path}")
    except FileNotFoundError:
        logger.error(f"Folder not found: {folder_path}")
        return []
    
    all_results = []
    
    for image_path in image_paths:
        logger.info(f"Processing: {image_path}")
        results = assess_single_image(image_path, detector, fqa, aligner)
        
        for result in results:
            result['image_path'] = image_path
            all_results.append(result)
    
    # Sort by quality score
    all_results.sort(key=lambda x: x['quality_score'], reverse=True)
    
    logger.info(f"\nTop {min(top_k, len(all_results))} highest quality faces:")
    for i, result in enumerate(all_results[:top_k]):
        logger.info(f"{i+1}. {Path(result['image_path']).name} - Face {result['face_id']}: "
                   f"Quality={result['quality_score']:.3f}, Detection={result['detection_score']:.3f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Face quality assessment")
    parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
    parser.add_argument("--detection-model", type=str, default="weights/yolov8n-face.onnx", 
                       help="Face detection model path")
    parser.add_argument("--quality-model", type=str, default="weights/face-quality-assessment.onnx", 
                       help="Face quality assessment model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--align", action="store_true", help="Use face alignment")
    parser.add_argument("--show", action="store_true", help="Show results (for single image)")
    parser.add_argument("--top-k", type=int, default=10, help="Show top K results (for folder)")
    parser.add_argument("--output", type=str, help="Output CSV file path")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    
    try:
        # Initialize models
        logger.info(f"Loading detection model: {args.detection_model}")
        detector = YOLOv8Face(args.detection_model, conf_thres=args.conf)
        
        logger.info(f"Loading quality assessment model: {args.quality_model}")
        fqa = FaceQualityAssessment(args.quality_model)
        
        # Initialize aligner if requested
        aligner = None
        if args.align:
            logger.info("Initializing face aligner")
            aligner = FaceAligner()
        
        # Check if input is file or folder
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            logger.info(f"Processing single image: {args.input}")
            results = assess_single_image(str(input_path), detector, fqa, aligner, args.show)
            
        elif input_path.is_dir():
            # Folder of images
            logger.info(f"Processing folder: {args.input}")
            results = assess_folder(str(input_path), detector, fqa, aligner, args.top_k)
            
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
        
        # Save results to CSV if requested
        if args.output and results:
            import csv
            
            fieldnames = ['image_path', 'face_id', 'detection_score', 'quality_score', 
                         'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
            
            with open(args.output, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    row = {
                        'image_path': result.get('image_path', args.input),
                        'face_id': result['face_id'],
                        'detection_score': result['detection_score'],
                        'quality_score': result['quality_score'],
                        'bbox_x': result['bbox'][0],
                        'bbox_y': result['bbox'][1],
                        'bbox_w': result['bbox'][2],
                        'bbox_h': result['bbox'][3],
                    }
                    writer.writerow(row)
            
            logger.info(f"Results saved to: {args.output}")
        
        logger.info(f"Processing complete. Total faces processed: {len(results)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())