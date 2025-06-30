#!/usr/bin/env python3
"""
Example: Face alignment and cropping.
"""

import argparse
import cv2
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection import YOLOv8Face
from alignment import FaceAligner
from utils import setup_logging, load_image_paths, create_output_dirs


def align_faces_in_image(image_path: str, detector: YOLOv8Face, aligner: FaceAligner, 
                        output_dir: str, template_mode: str = "arcface", 
                        output_size: int = 112, show_results: bool = False):
    """Align faces in a single image."""
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
    
    logger.info(f"Found {len(boxes)} faces in {Path(image_path).name}")
    
    aligned_faces = []
    result_image = image.copy()
    
    for i, (box, score, landmark) in enumerate(zip(boxes, scores, landmarks)):
        x, y, w, h = box.astype(int)
        
        try:
            # Convert bbox to xyxy format
            bbox_xyxy = [x, y, x+w, y+h]
            
            # Align face
            aligned_face, aligned_bbox, aligned_landmarks = aligner.align_face(
                image, bbox_xyxy, landmark,
                template_mode=template_mode,
                image_size=output_size
            )
            
            # Save aligned face
            image_name = Path(image_path).stem
            output_filename = f"{image_name}_face_{i}_aligned.jpg"
            output_path = Path(output_dir) / output_filename
            cv2.imwrite(str(output_path), aligned_face)
            
            aligned_faces.append({
                'face_id': i,
                'original_bbox': box,
                'detection_score': score,
                'aligned_face': aligned_face,
                'aligned_landmarks': aligned_landmarks,
                'output_path': str(output_path)
            })
            
            # Draw original detection
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, f"Face {i}: {score:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw landmarks
            if len(landmark) >= 10:
                for j in range(0, len(landmark), 3):
                    if j+1 < len(landmark):
                        cv2.circle(result_image, (int(landmark[j]), int(landmark[j+1])), 
                                 3, (0, 0, 255), -1)
            
            logger.info(f"Face {i} aligned and saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to align face {i}: {e}")
            continue
    
    # Save result image with detections
    if aligned_faces:
        result_filename = f"{Path(image_path).stem}_detections.jpg"
        result_path = Path(output_dir) / result_filename
        cv2.imwrite(str(result_path), result_image)
        logger.info(f"Detection result saved to: {result_path}")
    
    if show_results and aligned_faces:
        # Show original with detections
        cv2.namedWindow("Original with Detections", cv2.WINDOW_NORMAL)
        cv2.imshow("Original with Detections", result_image)
        
        # Show aligned faces
        for i, face_data in enumerate(aligned_faces):
            window_name = f"Aligned Face {i}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, face_data['aligned_face'])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return aligned_faces


def batch_align_folder(folder_path: str, detector: YOLOv8Face, aligner: FaceAligner, 
                      output_dir: str, **kwargs):
    """Align faces in all images in a folder."""
    logger = setup_logging()
    
    try:
        image_paths = load_image_paths(folder_path)
        logger.info(f"Found {len(image_paths)} images in {folder_path}")
    except FileNotFoundError:
        logger.error(f"Folder not found: {folder_path}")
        return []
    
    all_aligned_faces = []
    
    for image_path in image_paths:
        logger.info(f"Processing: {Path(image_path).name}")
        aligned_faces = align_faces_in_image(image_path, detector, aligner, output_dir, **kwargs)
        all_aligned_faces.extend(aligned_faces)
    
    logger.info(f"Batch processing complete. Total aligned faces: {len(all_aligned_faces)}")
    return all_aligned_faces


def create_alignment_grid(aligned_faces: list, output_path: str, grid_size: tuple = None):
    """Create a grid visualization of aligned faces."""
    if not aligned_faces:
        return
    
    logger = setup_logging()
    
    # Determine grid size
    num_faces = len(aligned_faces)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_faces)))
        rows = int(np.ceil(num_faces / cols))
    else:
        rows, cols = grid_size
    
    # Get face size from first aligned face
    face_size = aligned_faces[0]['aligned_face'].shape[:2]
    
    # Create grid image
    grid_height = rows * face_size[0]
    grid_width = cols * face_size[1]
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    for i, face_data in enumerate(aligned_faces[:rows*cols]):
        row = i // cols
        col = i % cols
        
        y_start = row * face_size[0]
        y_end = y_start + face_size[0]
        x_start = col * face_size[1]
        x_end = x_start + face_size[1]
        
        grid_image[y_start:y_end, x_start:x_end] = face_data['aligned_face']
    
    cv2.imwrite(output_path, grid_image)
    logger.info(f"Alignment grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Face alignment and cropping")
    parser.add_argument("--input", type=str, required=True, help="Input image or folder path")
    parser.add_argument("--output-dir", type=str, default="output/aligned", help="Output directory")
    parser.add_argument("--model", type=str, default="weights/yolov8n-face.onnx", help="Detection model path")
    parser.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--template-mode", type=str, default="arcface", choices=["arcface", "default"],
                       help="Alignment template mode")
    parser.add_argument("--output-size", type=int, default=112, help="Output face size")
    parser.add_argument("--template-scale", type=float, help="Template scale factor (0.8-1.2)")
    parser.add_argument("--allow-upscale", action="store_true", help="Allow upscaling small faces")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--create-grid", action="store_true", help="Create alignment grid visualization")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    
    try:
        # Create output directory
        output_dirs = create_output_dirs(args.output_dir)
        logger.info(f"Output directory: {args.output_dir}")
        
        # Initialize models
        logger.info(f"Loading detection model: {args.model}")
        detector = YOLOv8Face(args.model, conf_thres=args.conf)
        
        logger.info("Initializing face aligner")
        aligner = FaceAligner()
        
        # Alignment parameters
        align_params = {
            'template_mode': args.template_mode,
            'output_size': args.output_size,
            'show_results': args.show
        }
        
        if args.template_scale:
            align_params['template_scale'] = args.template_scale
        if args.allow_upscale:
            align_params['allow_upscale'] = True
        
        # Check if input is file or folder
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Single image
            logger.info(f"Processing single image: {args.input}")
            aligned_faces = align_faces_in_image(str(input_path), detector, aligner, 
                                               args.output_dir, **align_params)
            
        elif input_path.is_dir():
            # Folder of images
            logger.info(f"Processing folder: {args.input}")
            aligned_faces = batch_align_folder(str(input_path), detector, aligner, 
                                             args.output_dir, **align_params)
            
        else:
            raise ValueError(f"Input path does not exist: {args.input}")
        
        # Create alignment grid if requested
        if args.create_grid and aligned_faces:
            grid_path = Path(args.output_dir) / "alignment_grid.jpg"
            create_alignment_grid(aligned_faces, str(grid_path))
        
        logger.info(f"Processing complete. Total aligned faces: {len(aligned_faces)}")
        
        # Print summary
        if aligned_faces:
            logger.info("Alignment summary:")
            for face_data in aligned_faces:
                logger.info(f"  Face {face_data['face_id']}: {face_data['output_path']}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())