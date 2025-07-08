"""Face Quality Assessment Module."""

import cv2
import numpy as np
from typing import Union, List
import pathlib
import onnxruntime
from pathlib import Path

class FaceQualityLightQnet:
    """Face Quality Assessment using LightQnet model."""
    
    def __init__(self, model_path: str):
        """
        Initialize LightQnet model.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.net = onnxruntime.InferenceSession(model_path)
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
        self.input_shape = self.net.get_inputs()[0].shape
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
    def assess_quality(self, face_image: np.ndarray) -> np.ndarray:
        """
        Assess quality of a face image using ONNX LightQnet model.
        Args:
            face_image: Input face image (BGR format)
        Returns:
            Quality scores array (np.ndarray)
        """
        # Resize to model input
        img = cv2.resize(face_image, (self.input_width, self.input_height))
        # Convert to RGB if needed (model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize as in sample code: (img - 128.) / 128.
        img = (img.astype(np.float32) - 128.) / 128.
        # Add batch dimension
        images = np.expand_dims(img, axis=0)
        # If model expects grayscale, convert
        if self.input_shape[1] == 1:
            images = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=(0, -1))
        # Run ONNX inference
        qscore = self.net.run([self.output_name], {self.input_name: images})[0]
        return qscore[0] if qscore.shape[0] == 1 else qscore

    def get_quality_score(self, face_image: np.ndarray) -> float:
        """
        Get mean quality score for a face image.
        Args:
            face_image: Input face image
        Returns:
            Mean quality score (float)
        """
        scores = self.assess_quality(face_image)
        return float(np.mean(scores))
    
class FaceQualityAssessment:
    """Face Quality Assessment using deep learning model."""
    
    def __init__(self, model_path: str):
        """
        Initialize Face Quality Assessment model.
        
        Args:
            model_path: Path to ONNX model file
        """
        self.net = cv2.dnn.readNet(model_path)
        self.input_height = 112
        self.input_width = 112

    def assess_quality(self, face_image: np.ndarray) -> np.ndarray:
        """
        Assess quality of a face image.
        
        Args:
            face_image: Input face image (BGR format)
            
        Returns:
            Quality scores array
        """
        input_img = cv2.resize(
            cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB), 
            (self.input_width, self.input_height)
        )
        input_img = (input_img.astype(np.float32) / 255.0 - 0.5) / 0.5

        blob = cv2.dnn.blobFromImage(input_img.astype(np.float32))
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return outputs[0].reshape(-1)
    
    def get_quality_score(self, face_image: np.ndarray) -> float:
        """
        Get mean quality score for a face image.
        
        Args:
            face_image: Input face image
            
        Returns:
            Mean quality score
        """
        scores = self.assess_quality(face_image)
        return float(np.mean(scores))
    
    def batch_assess_folder(self, folder_path: str, top_k: int = 10) -> List[tuple]:
        """
        Assess quality of all images in a folder and return top K.
        
        Args:
            folder_path: Path to folder containing face images
            top_k: Number of top quality images to return
            
        Returns:
            List of (image_path, quality_score) tuples sorted by quality
        """
        folder = pathlib.Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder {folder_path} does not exist")
        
        results = []
        
        for image_path in folder.iterdir():
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                try:
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        quality_score = self.get_quality_score(image)
                        results.append((str(image_path), quality_score))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        # Sort by quality score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def filter_by_quality(self, face_crops: List[np.ndarray], 
                         min_quality: float = 0.5) -> List[tuple]:
        """
        Filter face crops by minimum quality threshold.
        
        Args:
            face_crops: List of face crop images
            min_quality: Minimum quality threshold
            
        Returns:
            List of (image, quality_score) tuples that meet threshold
        """
        filtered_faces = []
        
        for i, face_crop in enumerate(face_crops):
            try:
                quality_score = self.get_quality_score(face_crop)
                if quality_score >= min_quality:
                    filtered_faces.append((face_crop, quality_score))
            except Exception as e:
                print(f"Error assessing quality for face {i}: {e}")
                continue
                
        return filtered_faces