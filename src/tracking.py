"""Face Tracking Module with ByteTracker implementation."""

import cv2
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Any, List, Optional, Tuple
import scipy
import scipy.linalg
from .detection import YOLOv8Face

# Try to import optional dependencies
try:
    import lap
    HAS_LAP = True
except ImportError:
    HAS_LAP = False


class TrackState:
    """Enumeration for track states."""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Base class for object tracking."""
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()
        self.features = []
        self.curr_feature = None
        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        BaseTrack._count = 0


class KalmanFilterXYAH:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3],
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov


def linear_assignment(cost_matrix, thresh):
    """Perform linear assignment using scipy or lap."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.ndim == 0:
        cost_matrix = cost_matrix.reshape(1, 1)
    elif cost_matrix.ndim == 1:
        cost_matrix = cost_matrix.reshape(1, -1)

    try:
        if HAS_LAP:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
        else:
            raise ImportError("Using scipy fallback")
    except (ImportError, TypeError, ValueError):
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])
        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(frozenset(np.arange(cost_matrix.shape[0])) - frozenset(matches[:, 0]))
            unmatched_b = list(frozenset(np.arange(cost_matrix.shape[1])) - frozenset(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def batch_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)

    boxes1 = np.expand_dims(boxes1, axis=1)
    boxes2 = np.expand_dims(boxes2, axis=0)

    inter_x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)

    result = iou.squeeze()
    if result.ndim == 0:
        result = result.reshape(1, 1)
    return result


def iou_distance(atracks, btracks):
    """Compute cost based on IoU between tracks."""
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    if hasattr(atracks[0], 'xyxy'):
        atlbrs = [track.xyxy for track in atracks]
    else:
        atlbrs = atracks

    if hasattr(btracks[0], 'xyxy'):
        btlbrs = [track.xyxy for track in btracks]
    else:
        btlbrs = btracks

    ious = batch_iou(atlbrs, btlbrs)
    return 1 - ious


class STrack(BaseTrack):
    """Single object tracking representation using Kalman filtering."""
    
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh, score, cls):
        super().__init__()
        if len(xywh) == 4:
            x, y, w, h = xywh
            self._tlwh = np.asarray([x, y, w, h], dtype=np.float32)
        else:
            self._tlwh = np.asarray(xywh[:4], dtype=np.float32)
            
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def result(self):
        coords = self.xyxy
        return coords.tolist() + [self.track_id, self.score, self.cls, self.idx]


class BYTETracker:
    """ByteTracker implementation for multi-object tracking."""

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = KalmanFilterXYAH()
        self.reset_id()

    def update(self, detections):
        boxes, scores, classes, landmarks = detections
        
        if len(boxes) == 0:
            self.frame_id += 1
            for track in self.tracked_stracks:
                track.mark_lost()
                self.lost_stracks.append(track)
            self.tracked_stracks = []
            return np.array([])

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        high_conf_mask = scores >= self.track_thresh
        low_conf_mask = (scores > 0.1) & (scores < self.track_thresh)
        
        dets_high = boxes[high_conf_mask]
        scores_high = scores[high_conf_mask]
        classes_high = classes[high_conf_mask] if len(classes) > 0 else np.zeros(len(dets_high))
        
        dets_low = boxes[low_conf_mask]
        scores_low = scores[low_conf_mask]
        classes_low = classes[low_conf_mask] if len(classes) > 0 else np.zeros(len(dets_low))

        detections_high = self.init_track(dets_high, scores_high, classes_high)
        
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        dists = self.get_dists(strack_pool, detections_high)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if len(dets_low) > 0:
            detections_low = self.init_track(dets_low, scores_low, classes_low)
            r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
            
            dists = iou_distance(r_tracked_stracks, detections_low)
            matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_low[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections_remain = [detections_high[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections_remain)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections_remain[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def init_track(self, dets, scores, cls):
        return [STrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)] if len(dets) else []

    def get_dists(self, tracks, detections):
        return iou_distance(tracks, detections)

    def multi_predict(self, tracks):
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        STrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYAH()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista, tlistb):
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista, tlistb):
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb


class FaceTracker:
    """Complete face detection and tracking system."""
    
    def __init__(self, model_path: str, conf_thres: float = 0.2, iou_thres: float = 0.5, 
                 track_thresh: float = 0.5, track_buffer: int = 30, match_thresh: float = 0.8, 
                 frame_rate: int = 30):
        """
        Initialize Face Tracker.
        
        Args:
            model_path: Path to YOLOv8 face detection model
            conf_thres: Detection confidence threshold
            iou_thres: IoU threshold for NMS
            track_thresh: Tracking confidence threshold
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Track-detection matching threshold
            frame_rate: Video frame rate
        """
        # Initialize face detector
        self.detector = YOLOv8Face(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
        
        # Initialize tracker
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
        
        # Track history for visualization
        self.track_history = defaultdict(lambda: [])
        
    def detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and update tracking.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (tracked_faces, landmarks)
        """
        boxes, scores, classes, landmarks = self.detector.detect(frame)
        tracked_faces = self.tracker.update((boxes, scores, classes, landmarks))
        return tracked_faces, landmarks
    
    def draw_tracks(self, frame: np.ndarray, tracked_faces: np.ndarray, 
                   landmarks: Optional[np.ndarray] = None, draw_history: bool = True) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracked_faces: Tracked face results
            landmarks: Optional landmarks
            draw_history: Whether to draw track history
            
        Returns:
            Frame with tracking visualization
        """
        result_frame = frame.copy()
        
        for track in tracked_faces:
            if len(track) < 5:
                continue
                
            x1, y1, x2, y2, track_id, score = track[:6]
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID and confidence
            label = f"ID:{track_id} {score:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Update and draw track history
            if draw_history:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_hist = self.track_history[track_id]
                track_hist.append(center)
                if len(track_hist) > 30:
                    track_hist.pop(0)
                
                # Draw track trail
                if len(track_hist) > 1:
                    points = np.array(track_hist, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(result_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        
        return result_frame
    
    def reset(self):
        """Reset the tracker state."""
        self.tracker.reset()
        self.track_history.clear()
        
    def get_track_count(self) -> int:
        """Get current number of active tracks."""
        return len(self.tracker.tracked_stracks)
    
    def get_track_ids(self) -> List[int]:
        """Get list of current active track IDs."""
        return [track.track_id for track in self.tracker.tracked_stracks if track.is_activated]