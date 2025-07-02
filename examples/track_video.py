#!/usr/bin/env python3
"""
Example: Face tracking on video.
"""

import argparse
import cv2
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from tracking import FaceTracker
from utils import setup_logging, FPSCounter, VideoWriter, create_output_dirs
from face_crop_saver import FaceCropSaver
from alignment import FaceAligner
from quality import FaceQualityAssessment


def main():
    parser = argparse.ArgumentParser(description="Face tracking on video")
    parser.add_argument("--input", type=str, default="/home/ubuntu/Projects/sample_videos/metro_genclik_h265_45sec.mp4", help="Input video path or camera index")
    parser.add_argument("--model", type=str, default="weights/yolov8n-face.onnx", help="Model path")
    parser.add_argument("--output", type=str, default="/home/ubuntu/Projects/yolov8-face-landmarks-opencv-dnn/data/videos/tracker_video.mp4",help="Output video path")
    parser.add_argument("--conf", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--track-thresh", type=float, default=0.5, help="Tracking confidence threshold")
    parser.add_argument("--track-buffer", type=int, default=30, help="Track buffer frames")
    parser.add_argument("--match-thresh", type=float, default=0.8, help="Track matching threshold")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--save-faces", action="store_false", help="Save face crops")
    parser.add_argument("--output-dir", type=str, default="/home/ubuntu/Projects/yolov8-face-landmarks-opencv-dnn/data", help="Output directory")
    parser.add_argument("--align", action="store_false", help="Align face crops")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    
    # Create output directories
    if args.save_faces or args.output:
        output_dirs = create_output_dirs(args.output_dir)
        logger.info(f"Output directories created: {output_dirs}")

    # Initialize tracker
    logger.info(f"Loading model: {args.model}")
    tracker = FaceTracker(
        model_path=args.model,
        conf_thres=args.conf,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh
    )

    # Initialize aligner and quality model (if needed)
    aligner = FaceAligner() if args.align else None
    quality_model = FaceQualityAssessment('weights/face-quality-assessment.onnx')

    # Initialize face crop saver
    face_crop_saver = FaceCropSaver(
        output_dir=output_dirs['faces'],
        aligner=aligner,
        quality_model=quality_model,
        align=args.align,
        calc_quality=True  # Set True to enable quality
    )

    # Open video source
    try:
        # Try as camera index first
        video_source = int(args.input)
        logger.info(f"Opening camera: {video_source}")
    except ValueError:
        # Use as file path
        video_source = args.input
        logger.info(f"Opening video file: {video_source}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {args.input}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")

    # Initialize video writer if output specified
    video_writer = None
    if args.output:
        video_writer = VideoWriter(args.output, fps, (width, height))
        logger.info(f"Output video: {args.output}")

    # Initialize FPS counter
    fps_counter = FPSCounter()

    frame_count = 0
    logger.info("Starting face tracking...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Track faces
            tracked_faces = tracker.detect_and_track(frame)
            
            # Draw tracking results
            result_frame = tracker.draw_tracks(frame, tracked_faces)
            
            # Add frame info
            current_fps = fps_counter.update()
            info_text = f"Frame: {frame_count}, Tracks: {len(tracked_faces)}, FPS: {current_fps:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save face crops if requested
            if args.save_faces and len(tracked_faces) > 0:
                bboxes = tracked_faces[:, :4]  # Extract bounding boxes
                track_ids = tracked_faces[:, 4].astype(int)  # Extract track IDs
                landmarks = tracked_faces[:, 8:] if tracked_faces.shape[1] > 8 else None
                face_crop_saver.save(frame, bboxes, track_ids, frame_count, landmarks=landmarks)
            
            # Write to output video
            if video_writer:
                video_writer.write(result_frame)
            
            # Show video window
            if args.show:
                cv2.imshow("Face Tracking", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    logger.info("User requested exit")
                    break
            
            # Progress logging
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, active tracks: {len(tracked_faces)}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()
    
    logger.info(f"Processing complete. Total frames: {frame_count}")
    
        # Print tracking statistics
    active_tracks = tracker.get_track_count()
    track_ids = tracker.get_track_ids()
    logger.info(f"Final statistics: {active_tracks} active tracks, IDs: {track_ids}")
        
    # except Exception as e:
    #     logger.error(f"Error: {e}")
    #     return 1
    
    return 0


if __name__ == "__main__":
    main()