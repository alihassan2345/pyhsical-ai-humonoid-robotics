---
sidebar_label: "Sensor Integration and Data Fusion"
---

# Sensor Integration and Data Fusion

## Introduction

Sensor integration is fundamental to humanoid robotics, providing the robot with awareness of its environment, body state, and interaction capabilities. This chapter explores the integration of diverse sensor types, the challenges of sensor fusion, and the implementation of robust perception systems for humanoid robots operating in dynamic environments.

## Sensor Categories for Humanoid Robots

### Proprioceptive Sensors

Sensors that provide information about the robot's own state:

**Joint Position Sensors:**
- **Technology**: Encoders (optical, magnetic, Hall effect)
- **Accuracy**: &lt;0.1° for precise control
- **Update Rate**: 100Hz+ for real-time control
- **Integration**: Direct connection to motor controllers

**Inertial Measurement Units (IMUs):**
- **Components**: 3-axis accelerometer, gyroscope, magnetometer
- **Purpose**: Balance control, orientation estimation, motion detection
- **Placement**: Center of mass, feet, torso, head
- **Update Rate**: 200Hz+ for balance control

**Force/Torque Sensors:**
- **Location**: Joints, feet, hands
- **Purpose**: Contact detection, grasp control, balance
- **Resolution**: Millinewton level for fine control
- **Update Rate**: 1000Hz+ for impact detection

### Exteroceptive Sensors

Sensors that perceive the external environment:

**Vision Systems:**
- **RGB Cameras**: Color perception, object recognition
- **Depth Cameras**: 3D scene understanding
- **Stereo Cameras**: Depth estimation, obstacle detection
- **Event Cameras**: High-speed motion, low-latency perception

**Range Sensors:**
- **LiDAR**: 360° environment mapping, navigation
- **Ultrasonic**: Close-range obstacle detection
- **Infrared**: Night vision, proximity detection

**Tactile Sensors:**
- **Gripper Sensors**: Object contact, slip detection
- **Skin Sensors**: Whole-body contact awareness
- **Pressure Sensors**: Grasp force, surface interaction

## Sensor Fusion Architecture

### Data Fusion Framework

```python
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time

class SensorType(Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    JOINT = "joint"
    FORCE = "force"
    TACTILE = "tactile"

@dataclass
class SensorData:
    sensor_type: SensorType
    data: Any
    timestamp: float
    frame_id: str
    covariance: Optional[np.ndarray] = None

class SensorFusionNode:
    def __init__(self):
        self.sensors = {}
        self.data_buffers = {}
        self.fusion_results = {}
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.running = True

        # Initialize sensor data queues
        for sensor_type in SensorType:
            self.data_buffers[sensor_type] = queue.Queue(maxsize=100)

        # Start fusion thread
        self.fusion_thread.start()

    def register_sensor(self, sensor_id: str, sensor_type: SensorType,
                       topic: str, callback_func):
        """Register a sensor with the fusion system"""
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'topic': topic,
            'callback': callback_func,
            'last_update': 0.0
        }

    def sensor_callback(self, sensor_id: str, sensor_data: SensorData):
        """Callback for incoming sensor data"""
        sensor_info = self.sensors.get(sensor_id)
        if not sensor_info:
            return

        # Add to appropriate buffer
        buffer = self.data_buffers[sensor_info['type']]
        try:
            buffer.put_nowait(sensor_data)
        except queue.Full:
            # Remove oldest if full
            try:
                buffer.get_nowait()
                buffer.put_nowait(sensor_data)
            except queue.Empty:
                pass  # Should not happen

        # Update last update time
        sensor_info['last_update'] = time.time()

    def _fusion_loop(self):
        """Main fusion processing loop"""
        while self.running:
            try:
                # Get synchronized data from all sensors
                synchronized_data = self._get_synchronized_data()

                if synchronized_data:
                    # Perform fusion
                    fusion_result = self._perform_fusion(synchronized_data)

                    # Store result
                    self.fusion_results[time.time()] = fusion_result

                    # Publish results
                    self._publish_fusion_result(fusion_result)

                time.sleep(0.01)  # 100Hz fusion rate

            except Exception as e:
                print(f"Fusion loop error: {e}")

    def _get_synchronized_data(self) -> Dict[SensorType, List[SensorData]]:
        """Get synchronized sensor data within time window"""
        time_window = 0.05  # 50ms synchronization window
        current_time = time.time()

        synchronized_data = {}

        for sensor_type, buffer in self.data_buffers.items():
            # Get all data in buffer
            temp_data = []
            while not buffer.empty():
                try:
                    data = buffer.get_nowait()
                    if current_time - data.timestamp <= time_window:
                        temp_data.append(data)
                except queue.Empty:
                    break

            if temp_data:
                # Sort by timestamp and take most recent
                temp_data.sort(key=lambda x: x.timestamp)
                synchronized_data[sensor_type] = temp_data[-1:]  # Take most recent

        return synchronized_data

    def _perform_fusion(self, synchronized_data: Dict[SensorType, List[SensorData]]) -> Dict:
        """Perform sensor fusion"""
        result = {
            'timestamp': time.time(),
            'environment_map': self._fuse_environment(synchronized_data),
            'robot_state': self._fuse_robot_state(synchronized_data),
            'object_detections': self._fuse_object_detections(synchronized_data),
            'contact_state': self._fuse_contact_state(synchronized_data)
        }

        return result

    def _fuse_environment(self, data: Dict[SensorType, List[SensorData]]) -> Dict:
        """Fuse environmental perception data"""
        environment = {
            'occupancy_grid': None,
            'obstacles': [],
            'free_space': [],
            'confidence_map': None
        }

        # Process LiDAR data for occupancy mapping
        if SensorType.LIDAR in data:
            lidar_data = data[SensorType.LIDAR][0].data
            environment['occupancy_grid'] = self._process_lidar_to_grid(lidar_data)

        # Process camera data for object detection
        if SensorType.CAMERA in data:
            camera_data = data[SensorType.CAMERA][0].data
            camera_objects = self._process_camera_objects(camera_data)
            environment['obstacles'].extend(camera_objects)

        return environment

    def _fuse_robot_state(self, data: Dict[SensorType, List[SensorData]]) -> Dict:
        """Fuse robot state information"""
        state = {
            'position': np.array([0.0, 0.0, 0.0]),
            'orientation': np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion
            'velocity': np.array([0.0, 0.0, 0.0]),
            'angular_velocity': np.array([0.0, 0.0, 0.0]),
            'joint_positions': {},
            'balance_state': 'stable'
        }

        # Process IMU data for orientation
        if SensorType.IMU in data:
            imu_data = data[SensorType.IMU][0].data
            state['orientation'] = self._process_imu_orientation(imu_data)
            state['angular_velocity'] = np.array(imu_data['angular_velocity'])

        # Process joint data for position
        if SensorType.JOINT in data:
            joint_data = data[SensorType.JOINT][0].data
            state['joint_positions'] = joint_data['positions']

        # Determine balance state
        if SensorType.FORCE in data:
            force_data = data[SensorType.FORCE][0].data
            state['balance_state'] = self._determine_balance_state(force_data)

        return state

    def _fuse_object_detections(self, data: Dict[SensorType, List[SensorData]]) -> List[Dict]:
        """Fuse object detection from multiple sensors"""
        objects = []

        # Process camera detections
        if SensorType.CAMERA in data:
            camera_detections = data[SensorType.CAMERA][0].data.get('detections', [])
            for detection in camera_detections:
                objects.append({
                    'id': detection['id'],
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'position_3d': self._camera_to_world(detection),
                    'sensor': 'camera'
                })

        # Process LiDAR detections
        if SensorType.LIDAR in data:
            lidar_clusters = self._cluster_lidar_points(data[SensorType.LIDAR][0].data)
            for cluster in lidar_clusters:
                objects.append({
                    'id': f"lidar_{len(objects)}",
                    'class': 'unknown',
                    'confidence': 0.8,
                    'position_3d': cluster['centroid'],
                    'sensor': 'lidar'
                })

        # Associate and fuse detections
        fused_objects = self._associate_detections(objects)

        return fused_objects

    def _fuse_contact_state(self, data: Dict[SensorType, List[SensorData]]) -> Dict:
        """Fuse contact state information"""
        contact_state = {
            'contact_points': [],
            'grasp_status': 'unknown',
            'contact_force': 0.0,
            'slip_detected': False
        }

        # Process force/torque sensors
        if SensorType.FORCE in data:
            force_data = data[SensorType.FORCE][0].data
            contact_state['contact_points'] = force_data.get('contact_points', [])
            contact_state['contact_force'] = force_data.get('total_force', 0.0)

        # Process tactile sensors
        if SensorType.TACTILE in data:
            tactile_data = data[SensorType.TACTILE][0].data
            contact_state['slip_detected'] = tactile_data.get('slip_detected', False)
            contact_state['grasp_status'] = tactile_data.get('grasp_status', 'unknown')

        return contact_state

    def _publish_fusion_result(self, fusion_result: Dict):
        """Publish fusion results to ROS 2 topics"""
        # This would publish to ROS 2 topics in a real implementation
        print(f"Fusion result: {fusion_result}")

    def _process_lidar_to_grid(self, lidar_data) -> np.ndarray:
        """Convert LiDAR data to occupancy grid"""
        # Implementation would convert point cloud to occupancy grid
        return np.zeros((100, 100))  # Placeholder

    def _process_camera_objects(self, camera_data) -> List:
        """Process camera object detections"""
        # Implementation would process camera detections
        return []  # Placeholder

    def _process_imu_orientation(self, imu_data) -> np.ndarray:
        """Process IMU data to get orientation quaternion"""
        # Convert angular velocities and accelerations to orientation
        # This is a simplified implementation
        return np.array([0.0, 0.0, 0.0, 1.0])  # Placeholder

    def _determine_balance_state(self, force_data) -> str:
        """Determine robot balance state from force sensors"""
        # Analyze ZMP (Zero Moment Point) and CoM (Center of Mass)
        left_foot_force = force_data.get('left_foot', 0.0)
        right_foot_force = force_data.get('right_foot', 0.0)
        total_force = left_foot_force + right_foot_force

        if total_force < 10.0:  # Robot is not standing
            return 'airborne'
        elif abs(left_foot_force - right_foot_force) > 50.0:  # Unbalanced
            return 'unstable'
        else:
            return 'stable'

    def _camera_to_world(self, detection) -> np.ndarray:
        """Convert camera detection to world coordinates"""
        # Implementation would use camera calibration and robot pose
        return np.array([0.0, 0.0, 0.0])  # Placeholder

    def _cluster_lidar_points(self, point_cloud) -> List[Dict]:
        """Cluster LiDAR points into objects"""
        # Implementation would use clustering algorithm (DBSCAN, etc.)
        return []  # Placeholder

    def _associate_detections(self, objects) -> List[Dict]:
        """Associate detections from different sensors"""
        # Implementation would use data association algorithms
        return objects  # Placeholder
```

## Time Synchronization

### Sensor Synchronization Strategies

Ensuring accurate timing across sensors:

```python
import time
from collections import defaultdict, deque

class TimeSynchronizer:
    def __init__(self, time_tolerance: float = 0.01):  # 10ms tolerance
        self.time_tolerance = time_tolerance
        self.sensor_buffers = defaultdict(lambda: deque(maxlen=100))
        self.global_clock_offset = 0.0

    def add_sensor_data(self, sensor_id: str, data: Any, timestamp: float):
        """Add sensor data with its timestamp"""
        # Adjust timestamp using global offset
        adjusted_timestamp = timestamp + self.global_clock_offset

        self.sensor_buffers[sensor_id].append({
            'data': data,
            'timestamp': adjusted_timestamp,
            'received_time': time.time()
        })

    def get_synchronized_data(self, sensor_ids: List[str],
                            target_time: float) -> Dict[str, Any]:
        """Get synchronized data for given sensor IDs at target time"""
        synchronized_data = {}

        for sensor_id in sensor_ids:
            # Find closest data to target time
            buffer = self.sensor_buffers[sensor_id]

            if not buffer:
                continue

            # Find data within tolerance
            closest_data = None
            min_diff = float('inf')

            for item in buffer:
                diff = abs(item['timestamp'] - target_time)
                if diff < min_diff and diff <= self.time_tolerance:
                    min_diff = diff
                    closest_data = item

            if closest_data:
                synchronized_data[sensor_id] = closest_data['data']

        return synchronized_data

    def calibrate_clocks(self):
        """Calibrate sensor clocks relative to master clock"""
        # Implementation would use PTP or other synchronization protocols
        pass
```

## Kalman Filtering for Sensor Fusion

### Extended Kalman Filter Implementation

```python
import numpy as np
from scipy.linalg import block_diag

class ExtendedKalmanFilter:
    def __init__(self, state_dim: int, measurement_dim: int):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        self.state = np.zeros(state_dim)
        self.covariance = np.eye(state_dim) * 1000.0  # Initial uncertainty

        # Process noise
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise
        self.R = np.eye(measurement_dim) * 1.0

    def predict(self, dt: float, control_input: np.ndarray = None):
        """Prediction step of EKF"""
        # State transition model (simplified for position/velocity)
        F = np.eye(self.state_dim)

        # Position update: x_new = x + v*dt
        for i in range(3):  # x, y, z positions
            F[i, i+3] = dt  # Add velocity contribution

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurement: np.ndarray, measurement_model_func):
        """Update step of EKF"""
        # Linearize measurement model around current state
        H = self._linearize_measurement_model(measurement_model_func)

        # Innovation
        expected_measurement = measurement_model_func(self.state)
        innovation = measurement - expected_measurement

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance

    def _linearize_measurement_model(self, measurement_model_func) -> np.ndarray:
        """Linearize measurement model using Jacobian"""
        # Numerical differentiation to compute Jacobian
        epsilon = 1e-8
        H = np.zeros((self.measurement_dim, self.state_dim))

        base_measurement = measurement_model_func(self.state)

        for i in range(self.state_dim):
            # Perturb state
            perturbed_state = self.state.copy()
            perturbed_state[i] += epsilon

            # Get perturbed measurement
            perturbed_measurement = measurement_model_func(perturbed_state)

            # Compute partial derivative
            H[:, i] = (perturbed_measurement - base_measurement) / epsilon

        return H

class HumanoidStateEstimator:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, qw, qx, qy, qz, roll, pitch, yaw]
        self.ekf = ExtendedKalmanFilter(state_dim=13, measurement_dim=6)  # pos + orient
        self.last_update_time = time.time()

    def update_with_sensors(self, imu_data: Dict, joint_data: Dict,
                          vision_data: Dict = None):
        """Update state estimate with sensor data"""
        current_time = time.time()
        dt = current_time - self.last_update_time

        # Prediction step
        self.ekf.predict(dt)

        # Prepare measurement vector [pos, orientation]
        measurement = self._prepare_measurement(imu_data, joint_data, vision_data)

        # Update step
        self.ekf.update(measurement, self._measurement_model)

        self.last_update_time = current_time

    def _prepare_measurement(self, imu_data: Dict, joint_data: Dict,
                           vision_data: Dict) -> np.ndarray:
        """Prepare measurement vector from sensor data"""
        measurement = np.zeros(6)  # [x, y, z, qw, qx, qy, qz]

        # Position from vision or kinematics
        if vision_data:
            measurement[:3] = vision_data['position']
        else:
            # Use forward kinematics from joint data
            measurement[:3] = self._forward_kinematics(joint_data)

        # Orientation from IMU
        measurement[3:] = imu_data['orientation_quaternion']

        return measurement

    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        """Measurement model function"""
        # Return expected measurement given state
        # In this case, it's just the position and orientation part of state
        return state[:6]  # [x, y, z, qw, qx, qy, qz]

    def _forward_kinematics(self, joint_data: Dict) -> np.ndarray:
        """Compute forward kinematics for position estimate"""
        # Simplified implementation
        # In practice, would use DH parameters or other kinematic model
        return np.array([0.0, 0.0, 0.0])  # Placeholder
```

## Sensor Calibration

### Multi-Sensor Calibration

Calibrating sensors relative to each other:

```python
import cv2
import numpy as np
from scipy.optimize import minimize

class MultiSensorCalibrator:
    def __init__(self):
        self.calibration_data = []
        self.transformation_matrices = {}

    def add_calibration_pair(self, sensor1_data: np.ndarray,
                           sensor2_data: np.ndarray,
                           sensor1_type: str, sensor2_type: str):
        """Add a calibration data pair"""
        self.calibration_data.append({
            'sensor1_data': sensor1_data,
            'sensor2_data': sensor2_data,
            'sensor1_type': sensor1_type,
            'sensor2_type': sensor2_type
        })

    def calibrate_transform(self, sensor1_type: str, sensor2_type: str) -> np.ndarray:
        """Calibrate transformation between two sensors"""
        # Extract corresponding points
        points1 = []
        points2 = []

        for data_pair in self.calibration_data:
            if (data_pair['sensor1_type'] == sensor1_type and
                data_pair['sensor2_type'] == sensor2_type):
                points1.append(data_pair['sensor1_data'])
                points2.append(data_pair['sensor2_data'])

        if len(points1) < 3:
            raise ValueError("Need at least 3 point correspondences")

        # Compute transformation using SVD
        points1 = np.array(points1)
        points2 = np.array(points2)

        # Center the points
        centroid1 = np.mean(points1, axis=0)
        centroid2 = np.mean(points2, axis=0)

        points1_centered = points1 - centroid1
        points2_centered = points2 - centroid2

        # Compute cross-covariance matrix
        H = points1_centered.T @ points2_centered

        # SVD
        U, _, Vt = np.linalg.svd(H)

        # Compute rotation matrix
        R = Vt.T @ U.T

        # Ensure rotation matrix is proper (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid2 - R @ centroid1

        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t

        self.transformation_matrices[f"{sensor1_type}_to_{sensor2_type}"] = transform
        return transform

    def transform_point(self, point: np.ndarray, from_sensor: str, to_sensor: str) -> np.ndarray:
        """Transform a point from one sensor frame to another"""
        transform_key = f"{from_sensor}_to_{to_sensor}"
        if transform_key not in self.transformation_matrices:
            # Try inverse transform
            inv_key = f"{to_sensor}_to_{from_sensor}"
            if inv_key in self.transformation_matrices:
                inv_transform = self.transformation_matrices[inv_key]
                # Invert the transformation
                R_inv = inv_transform[:3, :3].T
                t_inv = -R_inv @ inv_transform[:3, 3]
                full_inv = np.eye(4)
                full_inv[:3, :3] = R_inv
                full_inv[:3, 3] = t_inv
                self.transformation_matrices[transform_key] = full_inv
            else:
                raise ValueError(f"No transformation found for {transform_key}")

        # Apply transformation
        point_homogeneous = np.ones(4)
        point_homogeneous[:3] = point
        transformed = self.transformation_matrices[transform_key] @ point_homogeneous
        return transformed[:3]
```

## Fault Detection and Tolerance

### Sensor Health Monitoring

Monitoring sensor health and handling failures:

```python
import statistics
from datetime import datetime

class SensorHealthMonitor:
    def __init__(self, sensor_names: List[str]):
        self.sensor_health = {name: {
            'status': 'healthy',  # healthy, degraded, failed
            'last_update': None,
            'data_history': [],
            'expected_rate': 0,
            'actual_rate': 0,
            'outlier_count': 0,
            'bias_drift': 0.0
        } for name in sensor_names}

        self.health_thresholds = {
            'rate_tolerance': 0.1,  # 10% tolerance
            'outlier_threshold': 5,  # max outliers per minute
            'bias_threshold': 0.1   # max bias drift
        }

    def update_sensor_data(self, sensor_name: str, data: Any, timestamp: float):
        """Update sensor health with new data"""
        health_info = self.sensor_health[sensor_name]

        # Update timing information
        if health_info['last_update']:
            time_diff = timestamp - health_info['last_update']
            if time_diff > 0:
                current_rate = 1.0 / time_diff
                health_info['actual_rate'] = current_rate

        health_info['last_update'] = timestamp

        # Store data for analysis
        health_info['data_history'].append({
            'data': data,
            'timestamp': timestamp,
            'value': self._extract_scalar_value(data)
        })

        # Keep only recent history (last 100 samples)
        if len(health_info['data_history']) > 100:
            health_info['data_history'] = health_info['data_history'][-50:]

        # Analyze data for anomalies
        self._analyze_sensor_data(sensor_name)

    def _extract_scalar_value(self, data: Any) -> float:
        """Extract a scalar value from sensor data for analysis"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, (list, tuple, np.ndarray)):
            # Use magnitude for vector data
            return np.linalg.norm(np.array(data))
        else:
            # For complex data structures, use a representative value
            return 0.0

    def _analyze_sensor_data(self, sensor_name: str):
        """Analyze sensor data for anomalies"""
        health_info = self.sensor_health[sensor_name]

        if len(health_info['data_history']) < 10:
            return  # Need more data for analysis

        # Calculate statistics
        values = [d['value'] for d in health_info['data_history']]

        # Check for outliers using IQR method
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        health_info['outlier_count'] = len(outliers)

        # Check for bias drift
        if len(values) >= 20:
            recent_mean = np.mean(values[-10:])
            historical_mean = np.mean(values[:-10])
            health_info['bias_drift'] = abs(recent_mean - historical_mean)

        # Update health status
        self._update_health_status(sensor_name)

    def _update_health_status(self, sensor_name: str):
        """Update the health status of a sensor"""
        health_info = self.sensor_health[sensor_name]

        status = 'healthy'

        # Check data rate
        if (health_info['expected_rate'] > 0 and
            health_info['actual_rate'] < health_info['expected_rate'] * (1 - self.health_thresholds['rate_tolerance'])):
            status = 'degraded'

        # Check outlier count
        if health_info['outlier_count'] > self.health_thresholds['outlier_threshold']:
            status = 'degraded' if status == 'healthy' else 'failed'

        # Check bias drift
        if health_info['bias_drift'] > self.health_thresholds['bias_threshold']:
            status = 'degraded' if status == 'healthy' else 'failed'

        health_info['status'] = status

    def get_healthy_sensors(self) -> List[str]:
        """Get list of healthy sensors"""
        return [name for name, info in self.sensor_health.items()
                if info['status'] == 'healthy']

    def get_sensor_status(self, sensor_name: str) -> str:
        """Get status of a specific sensor"""
        return self.sensor_health[sensor_name]['status']

    def handle_sensor_failure(self, failed_sensor: str) -> Dict[str, Any]:
        """Handle sensor failure by providing alternatives"""
        failure_info = {
            'failed_sensor': failed_sensor,
            'recommended_alternatives': [],
            'fallback_behavior': 'continue_with_defaults'
        }

        # Determine alternatives based on sensor type
        if 'camera' in failed_sensor:
            # Use LiDAR or other cameras if available
            failure_info['recommended_alternatives'] = [
                name for name in self.sensor_health.keys()
                if 'lidar' in name or ('camera' in name and name != failed_sensor)
            ]
            failure_info['fallback_behavior'] = 'use_lidar_for_navigation'

        elif 'imu' in failed_sensor:
            # Use encoder-based estimation
            failure_info['recommended_alternatives'] = [
                name for name in self.sensor_health.keys()
                if 'encoder' in name or 'joint' in name
            ]
            failure_info['fallback_behavior'] = 'estimate_orientation_from_kinematics'

        elif 'lidar' in failed_sensor:
            # Use cameras for obstacle detection
            failure_info['recommended_alternatives'] = [
                name for name in self.sensor_health.keys()
                if 'camera' in name
            ]
            failure_info['fallback_behavior'] = 'use_vision_for_obstacle_detection'

        return failure_info
```

## Integration with ROS 2

### Sensor Fusion ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, PointCloud2, Image
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

class HumanoidSensorFusionNode(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_fusion')

        # Initialize fusion components
        self.fusion_engine = SensorFusionNode()
        self.state_estimator = HumanoidStateEstimator()
        self.calibrator = MultiSensorCalibrator()
        self.health_monitor = SensorHealthMonitor([
            'imu_base', 'camera_left', 'camera_right',
            'lidar_front', 'force_left_foot', 'force_right_foot'
        ])

        # Subscribers
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        self.lidar_sub = self.create_subscription(
            PointCloud2, '/laser_scan', self.lidar_callback, 10)

        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)

        # Publishers
        self.fused_state_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_state', 10)

        self.environment_pub = self.create_publisher(
            PointCloud2, '/fused_environment', 10)

        self.health_status_pub = self.create_publisher(
            String, '/sensor_health_status', 10)

        # Timer for periodic fusion
        self.fusion_timer = self.create_timer(0.01, self.fusion_callback)  # 100Hz

        self.cv_bridge = CvBridge()
        self.get_logger().info('Humanoid Sensor Fusion Node Initialized')

    def imu_callback(self, msg):
        """Handle IMU data"""
        # Convert ROS IMU message to our format
        imu_data = {
            'linear_acceleration': [msg.linear_acceleration.x,
                                  msg.linear_acceleration.y,
                                  msg.linear_acceleration.z],
            'angular_velocity': [msg.angular_velocity.x,
                               msg.angular_velocity.y,
                               msg.angular_velocity.z],
            'orientation': [msg.orientation.x,
                          msg.orientation.y,
                          msg.orientation.z,
                          msg.orientation.w]
        }

        # Update health monitor
        self.health_monitor.update_sensor_data(
            'imu_base', imu_data, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

        # Add to fusion engine
        sensor_data = SensorData(
            sensor_type=SensorType.IMU,
            data=imu_data,
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            frame_id=msg.header.frame_id
        )
        self.fusion_engine.sensor_callback('imu_0', sensor_data)

    def joint_callback(self, msg):
        """Handle joint state data"""
        joint_data = {
            'positions': dict(zip(msg.name, msg.position)),
            'velocities': dict(zip(msg.name, msg.velocity)),
            'effort': dict(zip(msg.name, msg.effort))
        }

        # Update health monitor
        self.health_monitor.update_sensor_data(
            'joint_states', joint_data, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

        # Add to fusion engine
        sensor_data = SensorData(
            sensor_type=SensorType.JOINT,
            data=joint_data,
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            frame_id=msg.header.frame_id
        )
        self.fusion_engine.sensor_callback('joint_0', sensor_data)

    def lidar_callback(self, msg):
        """Handle LiDAR data"""
        # Convert PointCloud2 to numpy array
        lidar_data = self._convert_pointcloud2(msg)

        # Update health monitor
        self.health_monitor.update_sensor_data(
            'lidar_front', lidar_data, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)

        # Add to fusion engine
        sensor_data = SensorData(
            sensor_type=SensorType.LIDAR,
            data=lidar_data,
            timestamp=msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9,
            frame_id=msg.header.frame_id
        )
        self.fusion_engine.sensor_callback('lidar_0', sensor_data)

    def camera_callback(self, msg):
        """Handle camera data"""
        try:
            # Convert to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform object detection (simplified)
            detections = self._detect_objects(cv_image)

            camera_data = {
                'image': cv_image,
                'detections': detections,
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            }

            # Update health monitor
            self.health_monitor.update_sensor_data(
                'camera_left', camera_data, camera_data['timestamp'])

            # Add to fusion engine
            sensor_data = SensorData(
                sensor_type=SensorType.CAMERA,
                data=camera_data,
                timestamp=camera_data['timestamp'],
                frame_id=msg.header.frame_id
            )
            self.fusion_engine.sensor_callback('camera_0', sensor_data)

        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def fusion_callback(self):
        """Periodic fusion callback"""
        # Update state estimator
        # Note: In practice, this would use the fused data from the fusion engine
        current_time = self.get_clock().now().to_msg()
        self.state_estimator.update_with_sensors({}, {})  # Simplified

        # Publish health status
        healthy_sensors = self.health_monitor.get_healthy_sensors()
        health_msg = String()
        health_msg.data = f"Healthy sensors: {', '.join(healthy_sensors)}"
        self.health_status_pub.publish(health_msg)

    def _convert_pointcloud2(self, msg):
        """Convert PointCloud2 message to numpy array"""
        # Implementation would use sensor_msgs.point_cloud2.read_points
        # This is a simplified placeholder
        return np.array([])  # Placeholder

    def _detect_objects(self, image):
        """Detect objects in image"""
        # Placeholder for object detection
        return []  # Placeholder

    def destroy_node(self):
        """Clean up resources"""
        self.fusion_engine.running = False
        super().destroy_node()
```

## Learning Objectives

After completing this chapter, you should be able to:
- Integrate multiple sensor types in a humanoid robot system
- Implement sensor fusion algorithms for state estimation
- Perform sensor calibration and time synchronization
- Monitor sensor health and handle failures gracefully
- Design robust perception systems for dynamic environments

## Key Takeaways

- Sensor fusion combines data from multiple sensors for better perception
- Time synchronization is critical for accurate fusion
- Kalman filtering provides optimal state estimation
- Sensor calibration ensures accurate spatial relationships
- Health monitoring enables robust operation despite sensor failures