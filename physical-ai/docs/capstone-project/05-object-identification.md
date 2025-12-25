---
sidebar_label: "Object Identification Using Computer Vision"
---

# Object Identification Using Computer Vision

## Introduction

Object identification is a critical capability for the Autonomous Humanoid system, enabling the robot to perceive, recognize, and interact with objects in its environment. This chapter explores the implementation of computer vision techniques for object detection, recognition, and tracking, with a focus on real-time performance and robustness in dynamic environments.

## Computer Vision Architecture

### Perception Pipeline Overview

The object identification system follows a multi-stage pipeline:

```
Raw Image → Preprocessing → Feature Extraction → Object Detection → Recognition → Tracking → Semantic Understanding
```

Each stage builds upon the previous one to provide increasingly sophisticated understanding of the environment.

### System Requirements

For humanoid robot applications, the system must meet:

- **Real-time Performance**: Process images at 15-30 FPS for smooth interaction
- **Robust Detection**: Handle varying lighting, occlusions, and viewpoints
- **Low Latency**: Fast response for dynamic interaction scenarios
- **Multi-object Handling**: Track and identify multiple objects simultaneously
- **3D Understanding**: Estimate object poses and spatial relationships

## Deep Learning-Based Object Detection

### YOLO Integration for Real-time Detection

YOLO (You Only Look Once) provides an excellent balance of speed and accuracy for real-time applications:

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center coordinates
    area: float  # bounding box area

class YOLODetector:
    def __init__(self, model_path: str = "yolov5s.pt"):
        # Load pre-trained YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.eval()

        # Set to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # COCO dataset class names for humanoid-relevant objects
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Detect objects in an image using YOLO"""
        # Convert image to RGB if it's BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Run YOLO inference
        results = self.model(image_rgb)

        # Parse results
        detections = []
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)

                detection = Detection(
                    class_name=self.class_names[int(cls)],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(center_x, center_y),
                    area=area
                )
                detections.append(detection)

        return detections

    def detect_and_annotate(self, image: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """Detect objects and annotate the image"""
        detections = self.detect_objects(image)
        annotated_image = image.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            label = f"{detection.class_name}: {detection.confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                annotated_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return annotated_image, detections
```

## Custom Object Recognition

### Fine-tuning for Domain-Specific Objects

For humanoid applications, custom object recognition models may be needed:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

class CustomObjectDetector(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(CustomObjectDetector, self).__init__()

        # Use ResNet as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final classification layer

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Add bounding box regression head
        self.bbox_regressor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # x, y, width, height
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        bbox_deltas = self.bbox_regressor(features)

        return class_logits, bbox_deltas

class ObjectRecognitionPipeline:
    def __init__(self):
        # Initialize both general and custom detectors
        self.general_detector = YOLODetector()
        self.custom_detector = self._load_custom_model()
        self.object_database = self._load_object_database()

    def _load_custom_model(self) -> CustomObjectDetector:
        """Load custom-trained model for specific objects"""
        model = CustomObjectDetector(num_classes=20)  # Example: 20 custom classes

        # Load trained weights
        try:
            model.load_state_dict(torch.load("custom_objects_model.pth"))
            model.eval()
            return model
        except:
            print("Custom model not found, using general detector")
            return None

    def _load_object_database(self) -> Dict:
        """Load database of known objects with properties"""
        return {
            "water_bottle": {
                "size_range": (0.1, 0.3),  # meters
                "color": ["blue", "red", "clear"],
                "grasp_points": ["center", "top"],
                "weight": 0.5  # kg
            },
            "book": {
                "size_range": (0.15, 0.3),  # meters
                "color": ["various"],
                "grasp_points": ["spine", "center"],
                "weight": 0.8  # kg
            },
            "cup": {
                "size_range": (0.05, 0.1),  # meters
                "color": ["white", "black", "colored"],
                "grasp_points": ["handle", "rim"],
                "weight": 0.2  # kg
            }
        }

    def identify_objects(self, image: np.ndarray, depth_map: np.ndarray = None) -> Dict:
        """Comprehensive object identification with 3D information"""
        # Run general detection
        general_detections = self.general_detector.detect_objects(image)

        # Run custom detection if available
        custom_detections = []
        if self.custom_detector:
            custom_detections = self._run_custom_detection(image)

        # Combine and enrich detections
        all_detections = general_detections + custom_detections
        enriched_detections = self._enrich_detections(all_detections, image, depth_map)

        return {
            "detections": enriched_detections,
            "object_count": len(enriched_detections),
            "classes_found": list(set(d.class_name for d in enriched_detections))
        }

    def _run_custom_detection(self, image: np.ndarray) -> List[Detection]:
        """Run custom object detection model"""
        # Preprocess image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.custom_detector = self.custom_detector.cuda()

        with torch.no_grad():
            class_logits, bbox_deltas = self.custom_detector(input_tensor)

        # Process results (simplified - would need proper NMS and decoding)
        return []  # Placeholder

    def _enrich_detections(self, detections: List[Detection],
                          image: np.ndarray, depth_map: np.ndarray) -> List[Detection]:
        """Add 3D and semantic information to detections"""
        enriched_detections = []

        for detection in detections:
            # Calculate 3D position if depth map is available
            if depth_map is not None:
                x_center, y_center = detection.center
                depth = depth_map[y_center, x_center] if (0 <= y_center < depth_map.shape[0] and
                                                         0 <= x_center < depth_map.shape[1]) else None
                detection.depth = depth

            # Add semantic properties
            if detection.class_name in self.object_database:
                obj_props = self.object_database[detection.class_name]
                detection.properties = obj_props

            enriched_detections.append(detection)

        return enriched_detections
```

## 3D Object Pose Estimation

### Pose Estimation for Manipulation

Accurate 3D pose estimation is crucial for manipulation tasks:

```python
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PoseEstimator:
    def __init__(self):
        # Predefined 3D models for common objects
        self.object_models = self._load_object_models()

    def _load_object_models(self) -> Dict:
        """Load 3D models for pose estimation"""
        models = {}

        # Example: simple geometric models for common objects
        models['bottle'] = self._create_cylinder_model(0.05, 0.2)  # radius, height
        models['book'] = self._create_box_model(0.2, 0.15, 0.02)  # width, height, depth
        models['cup'] = self._create_cylinder_model(0.04, 0.1)  # radius, height

        return models

    def _create_cylinder_model(self, radius: float, height: float) -> o3d.geometry.TriangleMesh:
        """Create a cylinder model"""
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        # Move to center at origin
        center = cylinder.get_center()
        cylinder.translate(-center)
        return cylinder

    def _create_box_model(self, width: float, height: float, depth: float) -> o3d.geometry.TriangleMesh:
        """Create a box model"""
        box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
        # Move to center at origin
        center = box.get_center()
        box.translate(-center)
        return box

    def estimate_pose(self, detection: Detection, rgb_image: np.ndarray,
                     depth_image: np.ndarray, camera_intrinsics: Dict) -> Dict:
        """Estimate 3D pose of detected object"""
        # Extract region of interest from depth image
        x1, y1, x2, y2 = detection.bbox
        roi_depth = depth_image[y1:y2, x1:x2]
        roi_rgb = rgb_image[y1:y2, x1:x2]

        # Create point cloud from ROI
        points = self._depth_to_pointcloud(roi_depth, camera_intrinsics, (x1, y1))

        if len(points) < 100:  # Not enough points for reliable pose estimation
            return None

        # Match with object model using ICP or similar
        if detection.class_name in self.object_models:
            model = self.object_models[detection.class_name]
            pose = self._match_model_to_pointcloud(model, points)
            return pose

        return None

    def _depth_to_pointcloud(self, depth_image: np.ndarray,
                           camera_intrinsics: Dict, offset: Tuple[int, int]) -> np.ndarray:
        """Convert depth image to 3D point cloud"""
        height, width = depth_image.shape
        cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D coordinates
        z_coords = depth_image
        x_coords_3d = (x_coords - cx) * z_coords / fx
        y_coords_3d = (y_coords - cy) * z_coords / fy

        # Stack coordinates
        points = np.stack([x_coords_3d, y_coords_3d, z_coords], axis=-1)

        # Reshape to (N, 3) and remove invalid points
        points = points.reshape(-1, 3)
        valid_points = points[~np.isnan(points).any(axis=1) & (points[:, 2] > 0)]

        # Apply offset to get world coordinates
        offset_x, offset_y = offset
        valid_points[:, 0] += offset_x * z_coords.mean() / fx
        valid_points[:, 1] += offset_y * z_coords.mean() / fy

        return valid_points

    def _match_model_to_pointcloud(self, model: o3d.geometry.TriangleMesh,
                                 points: np.ndarray) -> Dict:
        """Match 3D model to point cloud to estimate pose"""
        # Convert points to PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate pose using ICP or other method
        # This is a simplified version - real implementation would be more complex
        if len(points) > 0:
            # Calculate centroid as position
            centroid = np.mean(points, axis=0)

            # For rotation, we'd typically use more sophisticated methods
            # For now, assume upright orientation
            rotation_matrix = np.eye(3)

            return {
                'position': centroid.tolist(),
                'rotation': rotation_matrix.tolist(),
                'confidence': 0.8  # Placeholder confidence
            }

        return None
```

## Object Tracking and Association

### Multi-Object Tracking System

Maintaining consistent object identities across frames:

```python
from scipy.optimize import linear_sum_assignment
import uuid

@dataclass
class TrackedObject:
    id: str
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    last_seen: float
    trajectory: List[Tuple[int, int]]
    velocity: Tuple[float, float]

class MultiObjectTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: int = 50):
        self.next_object_id = 0
        self.objects = {}  # ID -> TrackedObject
        self.disappeared = {}  # ID -> frames since last detection
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, detections: List[Detection], timestamp: float) -> List[TrackedObject]:
        """Update object tracking with new detections"""
        if len(detections) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return list(self.objects.values())

        # Calculate centroids for new detections
        input_centroids = np.array([det.center for det in detections])

        if len(self.objects) == 0:
            # If no existing objects, register all detections as new objects
            for i, detection in enumerate(detections):
                self._register_object(detection, timestamp)
        else:
            # Calculate distance between existing objects and new detections
            object_centroids = np.array([obj.center for obj in self.objects.values()])
            D = self._calculate_distance_matrix(object_centroids, input_centroids)

            # Find optimal assignment using Hungarian algorithm
            rows, cols = linear_sum_assignment(D)

            used_row_indices = set()
            used_col_indices = set()

            # Assign detections to existing objects
            for (row, col) in zip(rows, cols):
                if D[row, col] <= self.max_distance:
                    object_id = list(self.objects.keys())[row]
                    detection = detections[col]

                    self._update_object(object_id, detection, timestamp)

                    used_row_indices.add(row)
                    used_col_indices.add(col)

            # Handle unassigned existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            for row in unused_row_indices:
                object_id = list(self.objects.keys())[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]

            # Handle unassigned detections (new objects)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            for col in unused_col_indices:
                detection = detections[col]
                self._register_object(detection, timestamp)

        return list(self.objects.values())

    def _calculate_distance_matrix(self, centroids1: np.ndarray,
                                 centroids2: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between two sets of centroids"""
        if len(centroids1) == 0 or len(centroids2) == 0:
            return np.zeros((len(centroids1), len(centroids2)))

        # Calculate Euclidean distances
        distances = np.linalg.norm(centroids1[:, np.newaxis] - centroids2, axis=2)
        return distances

    def _register_object(self, detection: Detection, timestamp: float):
        """Register a new object"""
        object_id = f"obj_{self.next_object_id}"
        self.next_object_id += 1

        tracked_obj = TrackedObject(
            id=object_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            center=detection.center,
            last_seen=timestamp,
            trajectory=[detection.center],
            velocity=(0.0, 0.0)
        )

        self.objects[object_id] = tracked_obj
        self.disappeared[object_id] = 0

    def _update_object(self, object_id: str, detection: Detection, timestamp: float):
        """Update an existing object with new detection"""
        obj = self.objects[object_id]

        # Update trajectory and calculate velocity
        if obj.trajectory:
            prev_center = obj.trajectory[-1]
            dt = timestamp - obj.last_seen
            if dt > 0:
                velocity_x = (detection.center[0] - prev_center[0]) / dt
                velocity_y = (detection.center[1] - prev_center[1]) / dt
                obj.velocity = (velocity_x, velocity_y)

        obj.bbox = detection.bbox
        obj.center = detection.center
        obj.last_seen = timestamp
        obj.trajectory.append(detection.center)

        # Keep trajectory length reasonable
        if len(obj.trajectory) > 100:
            obj.trajectory = obj.trajectory[-50:]  # Keep last 50 points

        # Reset disappeared counter
        if object_id in self.disappeared:
            self.disappeared[object_id] = 0
```

## Integration with ROS 2

### Object Detection Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import message_filters

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize computer vision components
        self.detector = ObjectRecognitionPipeline()
        self.tracker = MultiObjectTracker()
        self.pose_estimator = PoseEstimator()
        self.cv_bridge = CvBridge()

        # Synchronize image and camera info
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/rgb/image_raw')
        self.info_sub = message_filters.Subscriber(self, CameraInfo, '/camera/rgb/camera_info')

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.image_info_callback)

        # Publishers
        self.detection_pub = self.create_publisher(
            ObjectDetectionArray,  # Custom message type
            '/object_detections',
            10
        )

        self.tracked_objects_pub = self.create_publisher(
            TrackedObjectArray,  # Custom message type
            '/tracked_objects',
            10
        )

        # Timer for periodic processing
        self.timer = self.create_timer(0.1, self.process_timer)  # 10 Hz

        self.get_logger().info('Object Detection Node Initialized')

    def image_info_callback(self, image_msg: Image, info_msg: CameraInfo):
        """Process synchronized image and camera info"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Extract camera intrinsics
            camera_intrinsics = {
                'fx': info_msg.k[0],  # [0, 0]
                'fy': info_msg.k[4],  # [1, 1]
                'cx': info_msg.k[2],  # [0, 2]
                'cy': info_msg.k[5],  # [1, 2]
            }

            # Store for processing
            self.last_image = cv_image
            self.last_camera_info = camera_intrinsics
            self.last_timestamp = image_msg.header.stamp

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def process_timer(self):
        """Process stored image with object detection"""
        if not hasattr(self, 'last_image'):
            return

        try:
            # Perform object detection
            detection_results = self.detector.identify_objects(self.last_image)

            # Update object tracking
            detections = detection_results['detections']
            tracked_objects = self.tracker.update(
                detections,
                self.last_timestamp.sec + self.last_timestamp.nanosec / 1e9
            )

            # Estimate poses for important objects
            for obj in tracked_objects:
                if obj.class_name in ['bottle', 'cup', 'book']:  # Manipulable objects
                    pose = self.pose_estimator.estimate_pose(
                        obj, self.last_image, None, self.last_camera_info
                    )
                    if pose:
                        obj.pose = pose

            # Publish results
            self.publish_detections(detection_results)
            self.publish_tracked_objects(tracked_objects)

        except Exception as e:
            self.get_logger().error(f'Error in object detection processing: {e}')

    def publish_detections(self, detection_results: Dict):
        """Publish object detection results"""
        msg = ObjectDetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_rgb_optical_frame'

        for detection in detection_results['detections']:
            detection_msg = ObjectDetection()
            detection_msg.class_name = detection.class_name
            detection_msg.confidence = detection.confidence
            detection_msg.bbox.x_offset = detection.bbox[0]
            detection_msg.bbox.y_offset = detection.bbox[1]
            detection_msg.bbox.width = detection.bbox[2] - detection.bbox[0]
            detection_msg.bbox.height = detection.bbox[3] - detection.bbox[1]

            msg.detections.append(detection_msg)

        self.detection_pub.publish(msg)

    def publish_tracked_objects(self, tracked_objects: List[TrackedObject]):
        """Publish tracked object information"""
        msg = TrackedObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        for obj in tracked_objects:
            obj_msg = TrackedObjectMsg()
            obj_msg.id = obj.id
            obj_msg.class_name = obj.class_name
            obj_msg.position.x = float(obj.center[0])
            obj_msg.position.y = float(obj.center[1])
            obj_msg.velocity.x = obj.velocity[0]
            obj_msg.velocity.y = obj.velocity[1]

            msg.objects.append(obj_msg)

        self.tracked_objects_pub.publish(msg)
```

## Semantic Scene Understanding

### Object Relations and Context

Understanding object relationships and scene context:

```python
class SceneUnderstanding:
    def __init__(self):
        # Define common object relationships
        self.spatial_relations = {
            'on': ['cup', 'book', 'plate'],  # Objects that can be on surfaces
            'in': ['food', 'drink', 'utensils'],  # Objects that can be in containers
            'near': ['person', 'furniture'],  # Objects that are typically near each other
            'holding': ['person']  # Objects that can hold other objects
        }

        # Define functional relationships
        self.functional_relations = {
            'cutlery': ['fork', 'knife', 'spoon'],
            'beverage_containers': ['cup', 'bottle', 'glass'],
            'reading_materials': ['book', 'magazine', 'newspaper'],
            'seating': ['chair', 'couch', 'stool']
        }

    def analyze_scene_context(self, detected_objects: List[Detection],
                            tracked_objects: List[TrackedObject]) -> Dict:
        """Analyze scene context and object relationships"""
        scene_analysis = {
            'object_groups': self._find_object_groups(detected_objects),
            'spatial_relations': self._analyze_spatial_relations(tracked_objects),
            'functional_context': self._determine_functional_context(detected_objects),
            'action_potential': self._identify_action_potential(detected_objects)
        }

        return scene_analysis

    def _find_object_groups(self, objects: List[Detection]) -> List[List[Detection]]:
        """Group related objects together"""
        groups = []
        used_indices = set()

        for i, obj1 in enumerate(objects):
            if i in used_indices:
                continue

            group = [obj1]
            used_indices.add(i)

            for j, obj2 in enumerate(objects[i+1:], i+1):
                if j in used_indices:
                    continue

                # Check if objects are close in image space
                center_dist = self._calculate_center_distance(obj1, obj2)
                if center_dist < 100:  # Threshold in pixels
                    group.append(obj2)
                    used_indices.add(j)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _analyze_spatial_relations(self, objects: List[TrackedObject]) -> List[Dict]:
        """Analyze spatial relationships between objects"""
        relations = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                # Calculate spatial relationship
                dx = obj2.center[0] - obj1.center[0]
                dy = obj2.center[1] - obj1.center[1]
                distance = (dx**2 + dy**2)**0.5

                # Determine relationship based on object types and distance
                relation = self._determine_spatial_relationship(obj1, obj2, distance)
                if relation:
                    relations.append(relation)

        return relations

    def _determine_spatial_relationship(self, obj1: TrackedObject, obj2: TrackedObject,
                                      distance: float) -> Dict:
        """Determine spatial relationship between two objects"""
        if obj1.class_name in self.spatial_relations.get('on', []) and distance < 50:
            return {
                'subject': obj1.id,
                'relation': 'on',
                'object': obj2.id,
                'confidence': 0.8
            }
        elif obj1.class_name in self.spatial_relations.get('near', []) and distance < 100:
            return {
                'subject': obj1.id,
                'relation': 'near',
                'object': obj2.id,
                'confidence': 0.7
            }

        return None

    def _determine_functional_context(self, objects: List[Detection]) -> str:
        """Determine functional context of the scene"""
        object_classes = [obj.class_name for obj in objects]

        # Check for dining scene
        dining_objects = set(self.functional_relations['beverage_containers'] +
                           self.functional_relations['cutlery'])
        if any(obj in dining_objects for obj in object_classes):
            return 'dining_area'

        # Check for reading scene
        reading_objects = set(self.functional_relations['reading_materials'])
        if any(obj in reading_objects for obj in object_classes):
            return 'reading_area'

        # Check for general living area
        seating_objects = set(self.functional_relations['seating'])
        if any(obj in seating_objects for obj in object_classes):
            return 'living_area'

        return 'unknown'

    def _identify_action_potential(self, objects: List[Detection]) -> List[Dict]:
        """Identify potential actions based on detected objects"""
        actions = []

        for obj in objects:
            if obj.class_name in ['bottle', 'cup', 'glass']:
                actions.append({
                    'action': 'grasp',
                    'target': obj.class_name,
                    'confidence': 0.9,
                    'description': f'Grasp the {obj.class_name} for drinking'
                })
            elif obj.class_name in ['book', 'magazine']:
                actions.append({
                    'action': 'pick_up',
                    'target': obj.class_name,
                    'confidence': 0.8,
                    'description': f'Pick up the {obj.class_name} for reading'
                })
            elif obj.class_name == 'person':
                actions.append({
                    'action': 'greet',
                    'target': 'person',
                    'confidence': 0.95,
                    'description': 'Approach and greet the person'
                })

        return actions
```

## Performance Optimization

### Efficient Processing Pipelines

Optimizing computer vision for real-time performance:

```python
import threading
import queue
from collections import deque

class OptimizedObjectDetection:
    def __init__(self):
        self.detection_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Use smaller models for distant objects
        self.model_selection = {
            'close_range': YOLODetector(model_path="yolov5s.pt"),  # High accuracy
            'medium_range': YOLODetector(model_path="yolov5n.pt"),  # Faster
            'far_range': YOLODetector(model_path="yolov5n.pt")     # Fastest
        }

        # ROI-based processing
        self.roi_enabled = True
        self.focus_regions = []

    def _process_loop(self):
        """Background processing loop"""
        while True:
            try:
                # Get image from queue
                image, metadata = self.detection_queue.get(timeout=1.0)

                # Perform detection
                results = self._perform_detection(image, metadata)

                # Put results in output queue
                self.result_queue.put(results)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def _perform_detection(self, image: np.ndarray, metadata: Dict) -> Dict:
        """Perform optimized detection based on scene characteristics"""
        # Select appropriate model based on distance/size requirements
        distance_estimate = metadata.get('distance_estimate', 'medium_range')
        detector = self.model_selection[distance_estimate]

        # Apply ROI processing if enabled
        if self.roi_enabled and self.focus_regions:
            detections = []
            for roi in self.focus_regions:
                x1, y1, x2, y2 = roi
                roi_image = image[y1:y2, x1:x2]
                roi_detections = detector.detect_objects(roi_image)

                # Adjust coordinates back to full image frame
                for det in roi_detections:
                    det.bbox = (det.bbox[0] + x1, det.bbox[1] + y1,
                               det.bbox[2] + x1, det.bbox[3] + y1)
                    det.center = (det.center[0] + x1, det.center[1] + y1)

                detections.extend(roi_detections)
        else:
            detections = detector.detect_objects(image)

        return {
            'detections': detections,
            'timestamp': metadata.get('timestamp'),
            'processing_time': metadata.get('processing_time')
        }

    def detect_async(self, image: np.ndarray, metadata: Dict = None) -> bool:
        """Submit image for async processing"""
        try:
            self.detection_queue.put_nowait((image, metadata or {}))
            return True
        except queue.Full:
            return False

    def get_results(self) -> Dict:
        """Get latest detection results"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
```

## Learning Objectives

After completing this chapter, you should be able to:
- Implement real-time object detection using deep learning models
- Estimate 3D object poses for manipulation tasks
- Track multiple objects across frames with consistent IDs
- Analyze scene context and object relationships
- Optimize computer vision pipelines for real-time performance
- Integrate object detection with ROS 2 systems

## Key Takeaways

- Object detection enables environmental awareness for autonomous robots
- 3D pose estimation is crucial for manipulation tasks
- Object tracking maintains consistent identities across frames
- Scene understanding enables contextual decision-making
- Performance optimization is essential for real-time applications