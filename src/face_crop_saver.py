import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional

class FaceCropSaver:
    def __init__(self, output_dir: str, aligner=None, quality_model=None, align: bool = False, calc_quality: bool = False):
        """
        Args:
            output_dir: Directory to save face crops
            aligner: Optional face aligner instance
            quality_model: Optional quality assessment model instance
            align: Whether to align faces before saving
            calc_quality: Whether to calculate and save quality score
        """
        self.output_dir = Path(output_dir)
        self.aligner = aligner
        self.quality_model = quality_model
        self.align = align
        self.calc_quality = calc_quality
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image: np.ndarray, bboxes: np.ndarray, track_ids: List[int], frame_id: int, landmarks=None) -> List[str]:
        saved_paths = []
        quality_scores = []
        def make_filename(frame_id, i, score=None):
            if score is not None:
                return f"frame_{frame_id:06d}_face_{i}_q{score:.3f}.jpg"
            else:
                return f"frame_{frame_id:06d}_face_{i}.jpg"

        if self.align and self.aligner is not None and landmarks is not None:
            if len(landmarks) != len(bboxes):
                logging.warning("Number of landmarks does not match number of bboxes. Skipping alignment.")
                landmarks = None
            else:
                zipped = zip(bboxes, track_ids, landmarks)
        if self.align and self.aligner is not None and landmarks is not None:
            for i, (bbox, track_id, landmark) in enumerate(zipped):
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    w = x2 - x1
                    h = y2 - y1
                    ratio = w / h if h > 0 else 0
                    if w >= 90 and h >= 90 and 0.8 <= ratio <= 1.25:
                        face_crop = image[y1:y2, x1:x2]
                        track_dir = self.output_dir / f"track_{track_id}"
                        track_dir.mkdir(exist_ok=True)
                        try:
                            aligned_face, _, _ = self.aligner.align_face(image, bbox, landmark, allow_upscale=True)
                            face_crop = aligned_face
                        except Exception as e:
                            logging.warning(f"Alignment failed for track {track_id}: {e}")
                            continue
                        score = None
                        if self.calc_quality and self.quality_model is not None:
                            score = self.quality_model.get_quality_score(face_crop)
                            filename = make_filename(frame_id, i, score)
                            quality_scores.append((str(track_dir / filename), score))
                        else:
                            filename = make_filename(frame_id, i)
                        file_path = track_dir / filename
                        cv2.imwrite(str(file_path), face_crop)
                        saved_paths.append(str(file_path))
        else:
            for i, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
                x1, y1, x2, y2 = bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    w = x2 - x1
                    h = y2 - y1
                    ratio = w / h if h > 0 else 0
                    if w >= 90 and h >= 90 and 0.8 <= ratio <= 1.25:
                        face_crop = image[y1:y2, x1:x2]
                        track_dir = self.output_dir / f"track_{track_id}"
                        track_dir.mkdir(exist_ok=True)
                        score = None
                        if self.calc_quality and self.quality_model is not None:
                            score = self.quality_model.get_quality_score(face_crop)
                            filename = make_filename(frame_id, i, score)
                            quality_scores.append((str(track_dir / filename), score))
                        else:
                            filename = make_filename(frame_id, i)
                        file_path = track_dir / filename
                        cv2.imwrite(str(file_path), face_crop)
                        saved_paths.append(str(file_path))
        if self.calc_quality:
            return quality_scores
        return saved_paths
