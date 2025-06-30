"""Face Alignment Module."""

from typing import Literal, Optional, Tuple, Union, List
import cv2
import numpy as np
from skimage import transform as trans


class FaceAligner:
    """Face alignment and cropping using facial landmarks."""

    # Reference templates for different face orientations (112x112 image size)
    _SRC1 = np.array([
        [51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
        [51.157, 89.050], [57.025, 89.702]
    ], dtype=np.float32)

    _SRC2 = np.array([
        [45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
        [45.177, 86.190], [64.246, 86.758]
    ], dtype=np.float32)

    _SRC3 = np.array([
        [39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
        [42.463, 87.010], [69.537, 87.010]
    ], dtype=np.float32)

    _SRC4 = np.array([
        [46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
        [48.167, 86.758], [67.236, 86.190]
    ], dtype=np.float32)

    _SRC5 = np.array([
        [54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
        [55.388, 89.702], [61.257, 89.050]
    ], dtype=np.float32)

    _SRC = np.array([_SRC1, _SRC2, _SRC3, _SRC4, _SRC5])
    _SRC_MAP = {112: _SRC, 224: _SRC * 2}

    # ArcFace 5-point reference landmarks
    _ARCFACE_TEMPLATE = np.array([
        [38.2946, 56.0],  # left eye
        [73.5318, 56.0],  # right eye
        [56.0252, 76.0],  # nose
        [41.5493, 96.0],  # left mouth
        [70.7299, 96.0],  # right mouth
    ], dtype=np.float32)

    _ARCFACE_TEMPLATE_224 = (_ARCFACE_TEMPLATE - 56.0) * 2.0 + 112.0
    _ARCFACE_TEMPLATE_MAP = {112: _ARCFACE_TEMPLATE, 224: _ARCFACE_TEMPLATE_224}

    TEMPLATE_MODES = ["arcface", "default"]

    def __init__(self):
        """Initialize the Face Aligner."""
        super().__init__()

    def align_face(self, image: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray,
                   template_scale: Optional[float] = None, template_mode: str = "arcface",
                   image_size: int = 112, allow_upscale: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop and align a facial image using detected landmarks.

        Args:
            image: Input image as numpy array (H, W, C)
            bbox: Detection bounding box as [x_min, y_min, x_max, y_max]
            landmarks: Facial landmarks coordinates (10 values for 5 points)
            template_scale: Optional scale factor for the template (0.8-1.2 recommended)
            template_mode: Template mode to use ('arcface' or 'default')
            image_size: Output image size (default: 112)
            allow_upscale: Whether to allow upscaling images

        Returns:
            Tuple containing:
                - Cropped and aligned facial image
                - Detection bounding box
                - Transformed facial landmarks in the aligned image
        """
        # Validate inputs
        self._validate_inputs(image, bbox, landmarks, template_scale, template_mode)

        # Check if upscaling would occur and adjust if not allowed
        if not allow_upscale:
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_size = max(face_width, face_height)

            if face_size < int(image_size * 0.85):
                raise ValueError(
                    f"Insufficient size of detected face: {face_size}. "
                    f"Consider setting allow_upscale=True"
                )

        # Convert detection coordinates to integers
        x_min, y_min, x_max, y_max = (int(coord) for coord in bbox)
        x_min, y_min = np.clip(x_min, 0, x_max - 1), np.clip(y_min, 0, y_max - 1)

        # Reshape landmarks to expected format (5, 2)
        landmarks = np.array(landmarks, dtype=np.float32).reshape((5, 2))

        # Apply alignment
        aligned_img, result_bbox, result_landmarks = self._align_using_landmarks(
            image, landmarks, (x_min, y_min, x_max, y_max),
            template_scale, template_mode, image_size
        )
        
        return aligned_img, result_bbox, result_landmarks

    def _validate_inputs(self, image: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray,
                        template_scale: Optional[float] = None, template_mode: str = "arcface") -> None:
        """Validate input parameters."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")

        if len(bbox) != 4:
            raise ValueError(f"Expected detection box with 4 values, got {len(bbox)}")

        landmarks_flat = np.array(landmarks).flatten()
        if len(landmarks_flat) != 10:
            raise ValueError(f"Expected 5 landmarks (10 values), got {len(landmarks_flat)}")

        if template_scale is not None and (template_scale <= 0 or template_scale > 2):
            raise ValueError(
                f"Invalid template scale: {template_scale}. "
                f"Value should be positive and preferably between 0.8 and 1.2."
            )

        if template_mode not in self.TEMPLATE_MODES:
            raise ValueError(
                f"Invalid template mode: {template_mode}. Choose from {self.TEMPLATE_MODES}."
            )

    def _align_using_landmarks(self, image: np.ndarray, landmarks: np.ndarray,
                              bbox: Tuple[int, int, int, int], template_scale: Optional[float] = None,
                              template_mode: str = "arcface", image_size: int = 112) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align face using all 5 facial landmarks."""
        try:
            # Get transformation matrix
            transform_matrix, _ = self._estimate_transform(
                landmarks, image_size, template_mode, template_scale
            )

            # Apply transformation to align the face
            aligned_img = self._apply_transform(
                image, landmarks, image_size, template_mode, template_scale
            )

            # Transform landmarks using the same transformation matrix
            transformed_landmarks = self._transform_landmarks(landmarks, transform_matrix)

            return aligned_img, np.array(bbox), transformed_landmarks
        except Exception as e:
            raise RuntimeError(f"Landmark alignment failed: {str(e)}") from e

    def _transform_landmarks(self, landmarks: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
        """Apply a transformation matrix to landmarks."""
        # Add homogeneous coordinate to landmarks
        homogeneous_landmarks = np.insert(landmarks, 2, values=np.ones(5), axis=1)
        
        # Apply transformation
        transformed = np.dot(transformation_matrix, homogeneous_landmarks.T).T
        
        return transformed.astype(np.int64)

    def _apply_transform(self, image: np.ndarray, landmarks: np.ndarray, image_size: int = 112,
                        mode: str = "arcface", template_scale: Optional[float] = None) -> np.ndarray:
        """Apply transformation to crop and align face."""
        transform_matrix, _ = self._estimate_transform(landmarks, image_size, mode, template_scale)
        warped = cv2.warpAffine(
            image, transform_matrix, (image_size, image_size), 
            borderMode=cv2.BORDER_REPLICATE
        )
        return warped

    def _estimate_transform(self, landmarks: np.ndarray, image_size: int, mode: str = "arcface",
                           template_scale: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Estimate the transformation matrix for face normalization."""
        assert landmarks.shape == (5, 2), f"Expected 5x2 landmarks, got {landmarks.shape}"

        # Create transformation estimator
        tform = trans.SimilarityTransform()

        # Add homogeneous coordinate
        lmk_tran = np.insert(landmarks, 2, values=np.ones(5), axis=1)

        # Initialize variables to track best transformation
        min_transform = []
        min_index = []
        min_error = float("inf")

        # Select appropriate template based on mode
        if mode == "arcface":
            src = self._get_arcface_template(image_size, template_scale)
            templates = [src]
        else:  # default mode
            src = self._get_default_templates(image_size, template_scale)
            templates = src

        # Find the best matching template
        for i, template in enumerate(templates):
            tform.estimate(landmarks, template)
            transform_matrix = tform.params[0:2, :]
            results = np.dot(transform_matrix, lmk_tran.T).T
            error = np.sum(np.sqrt(np.sum((results - template) ** 2, axis=1)))

            if error < min_error:
                min_error = error
                min_transform = transform_matrix
                min_index = i

        return min_transform, min_index

    def _get_arcface_template(self, image_size: int, template_scale: Optional[float] = None) -> np.ndarray:
        """Get ArcFace template for given image size."""
        # Try to get pre-calculated template from map
        src = self._ARCFACE_TEMPLATE_MAP.get(image_size)

        # If not found in map, scale the 112 template dynamically
        if src is None:
            base_src = self._ARCFACE_TEMPLATE_MAP[112].copy()
            scale_factor = image_size / 112.0
            center = (56, 56)  # Center of the 112x112 template
            src = self._scale_template(base_src, scale_factor, center)

            # Adjust center point for the new image size
            new_center = (image_size // 2, image_size // 2)
            src = src - np.array([56, 56]) + np.array([new_center[0], new_center[1]])
        else:
            src = src.copy()

        # Apply additional user-requested scaling if provided
        if template_scale is not None:
            center = (image_size // 2, image_size // 2)
            src = self._scale_template(src, template_scale, center)

        return src

    def _get_default_templates(self, image_size: int, template_scale: Optional[float] = None) -> np.ndarray:
        """Get default templates for given image size."""
        # Check if we have a template for this image size
        src = self._SRC_MAP.get(image_size)

        # If no template exists for this size, scale from 112
        if src is None:
            base_src = self._SRC_MAP[112].copy()
            scale_factor = image_size / 112.0

            # Scale each template in the set
            scaled_src = []
            for template in base_src:
                center = (56, 56)  # Center of the 112x112 template
                scaled_template = self._scale_template(template, scale_factor, center)

                # Adjust center point for the new image size
                new_center = (image_size // 2, image_size // 2)
                scaled_template = (
                    scaled_template - np.array([56, 56]) + 
                    np.array([new_center[0], new_center[1]])
                )
                scaled_src.append(scaled_template)

            src = np.array(scaled_src)

        # Apply additional user-requested scaling if provided
        if template_scale is not None:
            scaled_src = []
            center = (image_size // 2, image_size // 2)
            for template in src:
                scaled_template = self._scale_template(template, template_scale, center)
                scaled_src.append(scaled_template)
            src = np.array(scaled_src)

        return src

    @staticmethod
    def _scale_template(template: np.ndarray, scale: float = 0.9, 
                       center: Tuple[int, int] = (56, 56)) -> np.ndarray:
        """Scale landmark template around the center."""
        template = np.array(template, dtype=np.float32)
        cx, cy = center
        scaled = (template - [cx, cy]) * scale + [cx, cy]
        return scaled

    def batch_align_faces(self, image: np.ndarray, bboxes: List[np.ndarray], 
                         landmarks_list: List[np.ndarray], **kwargs) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Align multiple faces in a single image.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            landmarks_list: List of landmarks for each face
            **kwargs: Additional arguments for align_face
            
        Returns:
            List of (aligned_face, bbox, landmarks) tuples
        """
        results = []
        
        for bbox, landmarks in zip(bboxes, landmarks_list):
            try:
                aligned_face, result_bbox, result_landmarks = self.align_face(
                    image, bbox, landmarks, **kwargs
                )
                results.append((aligned_face, result_bbox, result_landmarks))
            except Exception as e:
                print(f"Failed to align face: {e}")
                continue
                
        return results