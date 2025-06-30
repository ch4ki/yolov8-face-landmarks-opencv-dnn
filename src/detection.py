"""YOLOv8 Face Detection Module."""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional


class YOLOv8Face:
    """YOLOv8 Face Detection with landmarks."""
    
    def __init__(self, model_path: str, conf_thres: float = 0.2, iou_thres: float = 0.5):
        """
        Initialize YOLOv8 Face detector.
        
        Args:
            model_path: Path to ONNX model file
            conf_thres: Confidence threshold for detection
            iou_thres: IoU threshold for NMS
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.net = cv2.dnn.readNet(model_path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [
            (math.ceil(self.input_height / self.strides[i]), 
             math.ceil(self.input_width / self.strides[i]))
            for i in range(len(self.strides))
        ]
        self.anchors = self._make_anchors(self.feats_hw)

    def _make_anchors(self, feats_hw: List[Tuple[int, int]], grid_cell_offset: float = 0.5) -> dict:
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h, w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset
            y = np.arange(0, h) + grid_cell_offset
            sx, sy = np.meshgrid(x, y)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def _softmax(self, x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Apply softmax function."""
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        return x_exp / x_sum
    
    def _resize_image(self, srcimg: np.ndarray, keep_ratio: bool = True) -> Tuple[np.ndarray, int, int, int, int]:
        """Resize image with optional aspect ratio preservation."""
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img, 0, 0, left, self.input_width - neww - left, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img, top, self.input_height - newh - top, 0, 0, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        
        return img, newh, neww, top, left

    def detect(self, srcimg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect faces in image.
        
        Args:
            srcimg: Input image (BGR format)
            
        Returns:
            Tuple of (bboxes, confidences, class_ids, landmarks)
        """
        input_img, newh, neww, padh, padw = self._resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        return self._post_process(outputs, scale_h, scale_w, padh, padw)

    def _post_process(self, preds: List[np.ndarray], scale_h: float, scale_w: float, 
                     padh: int, padw: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Post-process model predictions."""
        bboxes, scores, landmarks = [], [], []
        
        for i, pred in enumerate(preds):
            stride = int(self.input_height / pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1, 1))
            kpts = pred[..., -15:].reshape((-1, 15))  # x1,y1,score1, ..., x5,y5,score5

            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self._softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1, 4))

            bbox = self._distance2bbox(
                self.anchors[stride], bbox_pred, 
                max_shape=(self.input_height, self.input_width)
            ) * stride
            
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1, 1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1 + np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1, 15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1, 15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  # Convert to xywh
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > self.conf_threshold
        bboxes_wh = bboxes_wh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        landmarks = landmarks[mask]
        
        if len(bboxes_wh) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        indices = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(), confidences.tolist(), 
            self.conf_threshold, self.iou_threshold
        )
        
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
            
        if len(indices) > 0:
            return (bboxes_wh[indices], confidences[indices], 
                   class_ids[indices], landmarks[indices])
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def _distance2bbox(self, points: np.ndarray, distance: np.ndarray, 
                      max_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Convert distance predictions to bounding boxes."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
            
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image: np.ndarray, boxes: np.ndarray, 
                       scores: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw detection results on image."""
        result_img = image.copy()
        
        for box, score, kp in zip(boxes, scores, landmarks):
            x, y, w, h = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(
                result_img, f"face:{score:.2f}", (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2
            )
            
            # Draw landmarks
            for i in range(5):
                cv2.circle(
                    result_img, (int(kp[i * 3]), int(kp[i * 3 + 1])), 
                    4, (0, 255, 0), thickness=-1
                )
                
        return result_img