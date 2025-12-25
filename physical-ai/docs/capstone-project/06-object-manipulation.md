---
sidebar_label: "Object Manipulation Using Robotic Control"
---

# Object Manipulation Using Robotic Control

## Introduction

Object manipulation represents one of the most challenging aspects of humanoid robotics, requiring precise coordination of perception, planning, and control systems. This chapter explores the implementation of robotic manipulation capabilities that enable the Autonomous Humanoid to interact with objects in its environment through grasping, transporting, and placing operations.

## Manipulation System Architecture

### Overview of Manipulation Pipeline

The manipulation system follows a coordinated pipeline:

```
Perception → Grasp Planning → Motion Planning → Execution → Feedback → Success Verification
```

Each component must work seamlessly to achieve reliable manipulation in real-world scenarios.

### System Components

1. **Perception Integration**: Object detection, pose estimation, and property analysis
2. **Grasp Planning**: Determining optimal grasp points and configurations
3. **Motion Planning**: Path planning for arm and hand movements
4. **Control Systems**: Low-level control for precise movements
5. **Force Control**: Managing contact forces during manipulation
6. **Feedback Systems**: Monitoring grasp success and object state

## Grasp Planning and Analysis

### Grasp Point Selection

Determining optimal grasp points requires analysis of object properties:

```python
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class GraspType(Enum):
    PINCH = "pinch"
    PALM = "palm"
    LATERAL = "lateral"
    SUCTION = "suction"  # For specialized grippers

@dataclass
class GraspCandidate:
    position: Tuple[float, float, float]  # World coordinates
    orientation: Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    grasp_type: GraspType
    quality_score: float  # 0.0 to 1.0
    approach_direction: Tuple[float, float, float]  # Direction to approach object
    grasp_width: float  # Required gripper width (meters)

@dataclass
class ObjectProperties:
    dimensions: Tuple[float, float, float]  # width, height, depth in meters
    mass: float  # in kg
    center_of_mass: Tuple[float, float, float]  # relative to object origin
    surface_texture: str  # smooth, rough, etc.
    material: str  # plastic, metal, fabric, etc.
    stability: float  # 0.0 (unstable) to 1.0 (very stable)

class GraspPlanner:
    def __init__(self):
        self.robot_hand_params = {
            'max_aperture': 0.08,  # meters
            'max_force': 50.0,     # Newtons
            'finger_tips': True,   # Whether to use finger tips
            'suction_available': False  # Whether suction gripper is available
        }

    def generate_grasp_candidates(self, object_pose: Dict, object_props: ObjectProperties) -> List[GraspCandidate]:
        """Generate potential grasp points for an object"""
        candidates = []

        # Get object dimensions
        width, height, depth = object_props.dimensions

        # Generate grasp points based on object type and dimensions
        if width > 0.1 and depth > 0.1:  # Can be grasped from multiple sides
            # Side grasps (lateral)
            candidates.extend(self._generate_lateral_grasps(object_pose, object_props))

        if height > 0.05:  # Has sufficient height for top-down grasp
            # Top-down grasps (for handles, edges)
            candidates.extend(self._generate_top_down_grasps(object_pose, object_props))

        if width > 0.03 and depth > 0.03:  # Suitable for pinch grasp
            # Pinch grasps (for edges, handles)
            candidates.extend(self._generate_pinch_grasps(object_pose, object_props))

        # Filter and score candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._score_grasp_candidate(candidate, object_props)
            if score > 0.1:  # Minimum quality threshold
                candidate.quality_score = score
                scored_candidates.append(candidate)

        # Sort by quality score
        scored_candidates.sort(key=lambda x: x.quality_score, reverse=True)

        return scored_candidates

    def _generate_lateral_grasps(self, object_pose: Dict, object_props: ObjectProperties) -> List[GraspCandidate]:
        """Generate lateral grasp points along the object"""
        candidates = []
        width, height, depth = object_props.dimensions

        # Generate grasps along the width
        for i in range(3):  # Three grasp points along width
            x_offset = (i - 1) * width * 0.3  # -30%, center, +30%
            grasp_pos = (
                object_pose['position']['x'] + x_offset,
                object_pose['position']['y'],
                object_pose['position']['z'] + height / 2  # At half height
            )

            # Orientation for lateral grasp (perpendicular to object)
            grasp_orientation = self._calculate_lateral_orientation(object_pose)

            # Check if grasp width is feasible
            required_width = min(width, depth) * 0.8  # 80% of smallest dimension
            if required_width <= self.robot_hand_params['max_aperture']:
                candidate = GraspCandidate(
                    position=grasp_pos,
                    orientation=grasp_orientation,
                    grasp_type=GraspType.LATERAL,
                    quality_score=0.0,
                    approach_direction=(0, -1, 0),  # Approach from front
                    grasp_width=required_width
                )
                candidates.append(candidate)

        return candidates

    def _generate_top_down_grasps(self, object_pose: Dict, object_props: ObjectProperties) -> List[GraspCandidate]:
        """Generate top-down grasp points"""
        candidates = []
        width, height, depth = object_props.dimensions

        # Center top-down grasp
        grasp_pos = (
            object_pose['position']['x'],
            object_pose['position']['y'],
            object_pose['position']['z'] + height  # At top of object
        )

        # Orientation for top-down grasp (gripper pointing down)
        grasp_orientation = (0, 0, 0, 1)  # No rotation, gripper pointing down

        candidate = GraspCandidate(
            position=grasp_pos,
            orientation=grasp_orientation,
            grasp_type=GraspType.PALM,
            quality_score=0.0,
            approach_direction=(0, 0, -1),  # Approach from above
            grasp_width=min(width, depth) * 0.7
        )

        if candidate.grasp_width <= self.robot_hand_params['max_aperture']:
            candidates.append(candidate)

        return candidates

    def _generate_pinch_grasps(self, object_pose: Dict, object_props: ObjectProperties) -> List[GraspCandidate]:
        """Generate pinch grasp points for edges or handles"""
        candidates = []
        width, height, depth = object_props.dimensions

        # Generate pinch grasps at corners or edges
        corner_offsets = [
            (width/2, depth/2, height/2),   # Top-right corner
            (-width/2, depth/2, height/2),  # Top-left corner
            (width/2, -depth/2, height/2),  # Bottom-right corner
            (-width/2, -depth/2, height/2)  # Bottom-left corner
        ]

        for offset in corner_offsets:
            grasp_pos = (
                object_pose['position']['x'] + offset[0],
                object_pose['position']['y'] + offset[1],
                object_pose['position']['z'] + offset[2]
            )

            # Orientation for pinch grasp (fingers perpendicular to surface)
            grasp_orientation = (0, 0.707, 0, 0.707)  # 90-degree rotation

            candidate = GraspCandidate(
                position=grasp_pos,
                orientation=grasp_orientation,
                grasp_type=GraspType.PINCH,
                quality_score=0.0,
                approach_direction=(0, 0, -1),  # Approach from above
                grasp_width=0.02  # Small pinch grasp
            )

            if candidate.grasp_width <= self.robot_hand_params['max_aperture']:
                candidates.append(candidate)

        return candidates

    def _calculate_lateral_orientation(self, object_pose: Dict) -> Tuple[float, float, float, float]:
        """Calculate orientation for lateral grasp"""
        # For simplicity, assume object orientation is the grasp orientation
        # In practice, this would depend on object shape and desired grasp
        return (
            object_pose['orientation']['x'],
            object_pose['orientation']['y'],
            object_pose['orientation']['z'],
            object_pose['orientation']['w']
        )

    def _score_grasp_candidate(self, candidate: GraspCandidate, object_props: ObjectProperties) -> float:
        """Score a grasp candidate based on multiple factors"""
        score = 0.0

        # Stability factor (based on grasp type and object properties)
        if candidate.grasp_type == GraspType.PALM:
            stability_factor = 0.9
        elif candidate.grasp_type == GraspType.PINCH:
            stability_factor = 0.7
        else:  # LATERAL
            stability_factor = 0.8

        # Mass factor (heavier objects need more secure grasps)
        mass_factor = max(0.5, 1.0 - (object_props.mass / 5.0))  # Heavier = lower score

        # Grasp width factor (optimal grasp width)
        optimal_width = min(object_props.dimensions) * 0.6
        width_diff = abs(candidate.grasp_width - optimal_width)
        width_factor = max(0.1, 1.0 - width_diff)

        # Approach direction factor (some approaches are better than others)
        approach_factor = self._evaluate_approach_direction(candidate)

        # Combine factors
        score = (stability_factor * 0.3 +
                mass_factor * 0.2 +
                width_factor * 0.2 +
                approach_factor * 0.3)

        return min(1.0, score)  # Clamp to [0, 1]

    def _evaluate_approach_direction(self, candidate: GraspCandidate) -> float:
        """Evaluate how good the approach direction is"""
        # For now, assume front approach is good (0.8), others are ok (0.5)
        approach_dir = np.array(candidate.approach_direction)
        front_approach = np.array([0, -1, 0])  # Front direction

        # Calculate cosine similarity
        cos_sim = np.dot(approach_dir, front_approach) / (np.linalg.norm(approach_dir) * np.linalg.norm(front_approach))
        return max(0.5, (cos_sim + 1) / 2)  # Normalize to [0.5, 1]
```

## Motion Planning for Manipulation

### Arm Trajectory Planning

Planning smooth, collision-free trajectories for manipulation:

```python
import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Tuple
import open3d as o3d

class ManipulationMotionPlanner:
    def __init__(self):
        self.workspace_bounds = {
            'x': (-1.0, 1.0),
            'y': (-1.0, 1.0),
            'z': (0.0, 1.5)
        }
        self.collision_threshold = 0.05  # meters

    def plan_grasp_trajectory(self, object_pose: Dict, grasp_candidate: GraspCandidate,
                             robot_pose: Dict, obstacles: List) -> Optional[List[Tuple[float, float, float]]]:
        """Plan trajectory to reach and grasp an object"""
        # Define key waypoints for the grasp trajectory
        waypoints = []

        # 1. Pre-grasp position (above the object)
        pre_grasp_pos = (
            grasp_candidate.position[0],
            grasp_candidate.position[1],
            grasp_candidate.position[2] + 0.1  # 10cm above object
        )
        waypoints.append(pre_grasp_pos)

        # 2. Approach position (just above grasp point)
        approach_pos = (
            grasp_candidate.position[0],
            grasp_candidate.position[1],
            grasp_candidate.position[2] + 0.02  # 2cm above grasp point
        )
        waypoints.append(approach_pos)

        # 3. Grasp position
        waypoints.append(grasp_candidate.position)

        # Verify trajectory is collision-free
        if self._is_trajectory_collision_free(waypoints, robot_pose, obstacles):
            return self._smooth_trajectory(waypoints)
        else:
            # Try alternative trajectory
            alternative_waypoints = self._generate_alternative_trajectory(
                object_pose, grasp_candidate, robot_pose, obstacles)
            if alternative_waypoints and self._is_trajectory_collision_free(
                alternative_waypoints, robot_pose, obstacles):
                return self._smooth_trajectory(alternative_waypoints)

        return None

    def plan_place_trajectory(self, target_pose: Dict, current_object_pose: Dict,
                             robot_pose: Dict, obstacles: List) -> Optional[List[Tuple[float, float, float]]]:
        """Plan trajectory to place object at target location"""
        waypoints = []

        # 1. Lift position (object slightly raised from current position)
        lift_pos = (
            current_object_pose['position']['x'],
            current_object_pose['position']['y'],
            current_object_pose['position']['z'] + 0.05  # 5cm lift
        )
        waypoints.append(lift_pos)

        # 2. Transport position (intermediate safe position)
        transport_pos = self._calculate_transport_position(
            current_object_pose, target_pose, robot_pose)
        waypoints.append(transport_pos)

        # 3. Approach position (above target)
        approach_pos = (
            target_pose['position']['x'],
            target_pose['position']['y'],
            target_pose['position']['z'] + 0.1  # 10cm above target
        )
        waypoints.append(approach_pos)

        # 4. Place position
        place_pos = (
            target_pose['position']['x'],
            target_pose['position']['y'],
            target_pose['position']['z'] + target_pose['dimensions'][1]/2  # Half height of target
        )
        waypoints.append(place_pos)

        if self._is_trajectory_collision_free(waypoints, robot_pose, obstacles):
            return self._smooth_trajectory(waypoints)

        return None

    def _is_trajectory_collision_free(self, waypoints: List[Tuple[float, float, float]],
                                    robot_pose: Dict, obstacles: List) -> bool:
        """Check if trajectory is collision-free"""
        # For simplicity, check each waypoint against obstacles
        # In practice, would check the full arm configuration
        for waypoint in waypoints:
            if not self._is_position_safe(waypoint, obstacles):
                return False

        return True

    def _is_position_safe(self, position: Tuple[float, float, float], obstacles: List) -> bool:
        """Check if a position is safe from obstacles"""
        x, y, z = position

        # Check workspace bounds
        if (not (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1]) or
            not (self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1]) or
            not (self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1])):
            return False

        # Check collision with obstacles
        for obstacle in obstacles:
            obs_pos = (obstacle['x'], obstacle['y'], obstacle['z'])
            obs_size = obstacle.get('size', 0.1)  # Default size

            distance = np.sqrt((x - obs_pos[0])**2 + (y - obs_pos[1])**2 + (z - obs_pos[2])**2)
            if distance < (obs_size + self.collision_threshold):
                return False

        return True

    def _smooth_trajectory(self, waypoints: List[Tuple[float, float, float]],
                          num_points: int = 20) -> List[Tuple[float, float, float]]:
        """Smooth trajectory using cubic spline interpolation"""
        if len(waypoints) < 2:
            return waypoints

        # Convert to numpy arrays
        points = np.array(waypoints)

        # Create parameterization based on cumulative distance
        distances = [0]
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            distances.append(distances[-1] + dist)

        # Create spline
        tck, u = np.linspace(0, 1, len(points)), distances
        cs = CubicSpline(u, points, bc_type='natural')

        # Generate smooth path
        smooth_points = []
        t_new = np.linspace(u[0], u[-1], num_points)
        for t in t_new:
            point = cs(t)
            smooth_points.append(tuple(point))

        return smooth_points

    def _generate_alternative_trajectory(self, object_pose: Dict, grasp_candidate: GraspCandidate,
                                       robot_pose: Dict, obstacles: List) -> Optional[List]:
        """Generate alternative trajectory if primary fails"""
        # Try approach from different side
        approach_offset = 0.15  # 15cm offset
        alt_approach_pos = (
            grasp_candidate.position[0] + approach_offset,
            grasp_candidate.position[1],
            grasp_candidate.position[2] + 0.05
        )

        waypoints = [alt_approach_pos, grasp_candidate.position]
        return waypoints if self._is_trajectory_collision_free(waypoints, robot_pose, obstacles) else None

    def _calculate_transport_position(self, current_pose: Dict, target_pose: Dict,
                                    robot_pose: Dict) -> Tuple[float, float, float]:
        """Calculate safe transport position between current and target"""
        # Choose a position that's reachable and safe
        # For now, use a position at robot shoulder height in between
        mid_x = (current_pose['position']['x'] + target_pose['position']['x']) / 2
        mid_y = (current_pose['position']['y'] + target_pose['position']['y']) / 2

        return (mid_x, mid_y, 0.8)  # Shoulder height
```

## Force Control and Grasp Management

### Impedance Control for Safe Interaction

Implementing force control for safe manipulation:

```python
class ForceController:
    def __init__(self):
        # Impedance control parameters
        self.stiffness = {
            'position': 1000.0,  # N/m
            'orientation': 100.0  # N.m/rad
        }
        self.damping_ratio = 1.0  # Critical damping
        self.force_limits = {
            'gripper': 40.0,  # Maximum gripper force in N
            'wrist': 20.0    # Maximum wrist force in N
        }

        # Force thresholds for different operations
        self.force_thresholds = {
            'contact_detection': 5.0,    # N
            'grasp_confirmation': 15.0,  # N
            'slip_detection': 8.0,       # N
            'release_threshold': 2.0     # N
        }

    def impedance_control_step(self, desired_pose: Dict, current_pose: Dict,
                              external_forces: Dict, dt: float) -> Dict:
        """Perform one step of impedance control"""
        # Calculate position error
        pos_error = {
            'x': desired_pose['position']['x'] - current_pose['position']['x'],
            'y': desired_pose['position']['y'] - current_pose['position']['y'],
            'z': desired_pose['position']['z'] - current_pose['position']['z']
        }

        # Calculate orientation error (simplified)
        orientation_error = self._calculate_orientation_error(
            desired_pose['orientation'], current_pose['orientation'])

        # Calculate impedance force
        impedance_force = {
            'linear': {
                'x': self.stiffness['position'] * pos_error['x'],
                'y': self.stiffness['position'] * pos_error['y'],
                'z': self.stiffness['position'] * pos_error['z']
            },
            'angular': {
                'x': self.stiffness['orientation'] * orientation_error['x'],
                'y': self.stiffness['orientation'] * orientation_error['y'],
                'z': self.stiffness['orientation'] * orientation_error['z']
            }
        }

        # Apply external force compensation
        commanded_force = self._apply_force_compensation(impedance_force, external_forces)

        # Limit forces
        limited_force = self._limit_forces(commanded_force)

        return limited_force

    def _calculate_orientation_error(self, desired_quat: Dict, current_quat: Dict) -> Dict:
        """Calculate orientation error in angular velocity space"""
        # Convert quaternions to rotation matrices and calculate error
        # Simplified implementation
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Placeholder

    def _apply_force_compensation(self, impedance_force: Dict, external_forces: Dict) -> Dict:
        """Apply external force compensation"""
        compensated_force = {
            'linear': {
                'x': impedance_force['linear']['x'] - external_forces.get('linear', {}).get('x', 0),
                'y': impedance_force['linear']['y'] - external_forces.get('linear', {}).get('y', 0),
                'z': impedance_force['linear']['z'] - external_forces.get('linear', {}).get('z', 0)
            },
            'angular': {
                'x': impedance_force['angular']['x'] - external_forces.get('angular', {}).get('x', 0),
                'y': impedance_force['angular']['y'] - external_forces.get('angular', {}).get('y', 0),
                'z': impedance_force['angular']['z'] - external_forces.get('angular', {}).get('z', 0)
            }
        }

        return compensated_force

    def _limit_forces(self, forces: Dict) -> Dict:
        """Apply force limits"""
        limited = {
            'linear': {},
            'angular': {}
        }

        for axis in ['x', 'y', 'z']:
            limited['linear'][axis] = np.clip(
                forces['linear'][axis], -self.force_limits['wrist'], self.force_limits['wrist'])
            limited['angular'][axis] = np.clip(
                forces['angular'][axis], -self.force_limits['wrist']*0.1, self.force_limits['wrist']*0.1)

        return limited

    def adaptive_grasp_control(self, object_properties: ObjectProperties,
                              current_force: float, desired_force: float) -> Dict:
        """Adaptive control for grasp force based on object properties"""
        # Adjust desired force based on object properties
        adjusted_force = desired_force

        # Increase force for heavier objects
        if object_properties.mass > 1.0:
            adjusted_force *= 1.0 + (object_properties.mass - 1.0) * 0.2

        # Increase force for smooth objects
        if object_properties.surface_texture in ['smooth', 'slippery']:
            adjusted_force *= 1.3

        # Increase force for unstable objects
        if object_properties.stability < 0.5:
            adjusted_force *= 1.2

        # Ensure within limits
        adjusted_force = min(adjusted_force, self.force_limits['gripper'])

        # Calculate control output
        force_error = adjusted_force - current_force
        control_output = self._pid_control(force_error)

        return {
            'commanded_force': min(adjusted_force, self.force_limits['gripper']),
            'control_signal': control_output,
            'success_probability': self._estimate_grasp_success(current_force, object_properties)
        }

    def _pid_control(self, error: float, kp: float = 0.5, ki: float = 0.1, kd: float = 0.05) -> float:
        """Simple PID controller for force control"""
        # In a real implementation, you'd track integral and derivative terms
        return kp * error  # Simplified proportional control

    def _estimate_grasp_success(self, applied_force: float, object_props: ObjectProperties) -> float:
        """Estimate probability of grasp success"""
        # Heuristic-based estimation
        min_force_for_success = max(5.0, object_props.mass * 2.0)  # At least 2x weight
        max_force_for_success = min(40.0, object_props.mass * 10.0)  # Don't exceed 10x weight

        if min_force_for_success <= applied_force <= max_force_for_success:
            # Within optimal range
            success_prob = 0.9
        elif applied_force < min_force_for_success:
            # Too little force
            success_prob = 0.3
        else:
            # Too much force (risk of damage)
            success_prob = 0.7

        # Adjust for object properties
        if object_props.surface_texture in ['rough', 'textured']:
            success_prob *= 1.1  # Better for rough surfaces
        elif object_props.surface_texture in ['smooth', 'slippery']:
            success_prob *= 0.8  # Worse for smooth surfaces

        return min(1.0, success_prob)
```

## Manipulation Execution and Control

### ROS 2 Manipulation Interface

Implementing the manipulation system with ROS 2:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration

from ..interfaces.action import GraspObject, PlaceObject  # Custom action definitions

class ManipulationController(Node):
    def __init__(self):
        super().__init__('manipulation_controller')

        # Initialize manipulation components
        self.grasp_planner = GraspPlanner()
        self.motion_planner = ManipulationMotionPlanner()
        self.force_controller = ForceController()

        # Action servers
        self.grasp_server = ActionServer(
            self,
            GraspObject,
            'grasp_object',
            self.execute_grasp_object
        )

        self.place_server = ActionServer(
            self,
            PlaceObject,
            'place_object',
            self.execute_place_object
        )

        # Publishers and subscribers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.gripper_command_pub = self.create_publisher(
            GripperCommand, '/gripper_controller/command', 10)

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/wrist_wrench', self.wrench_callback, 10)

        # Robot state
        self.current_joint_states = None
        self.current_wrench = None
        self.robot_pose = None

        self.get_logger().info('Manipulation Controller Initialized')

    def execute_grasp_object(self, goal_handle):
        """Execute grasp object action"""
        self.get_logger().info(f'Attempting to grasp {goal_handle.request.object_name}')

        try:
            # Get object information
            object_info = self._get_object_info(goal_handle.request.object_name)
            if not object_info:
                self.get_logger().error(f'Object {goal_handle.request.object_name} not found')
                goal_handle.abort()
                return GraspObject.Result(success=False, message="Object not found")

            # Plan grasp
            grasp_candidates = self.grasp_planner.generate_grasp_candidates(
                object_info['pose'], object_info['properties'])

            if not grasp_candidates:
                self.get_logger().error(f'No valid grasp candidates for {goal_handle.request.object_name}')
                goal_handle.abort()
                return GraspObject.Result(success=False, message="No valid grasp candidates")

            # Select best grasp candidate
            best_grasp = grasp_candidates[0]
            self.get_logger().info(f'Selected grasp with quality: {best_grasp.quality_score:.2f}')

            # Plan trajectory to grasp
            trajectory = self.motion_planner.plan_grasp_trajectory(
                object_info['pose'], best_grasp, self.robot_pose, object_info.get('obstacles', []))

            if not trajectory:
                self.get_logger().error('Could not plan valid trajectory to grasp')
                goal_handle.abort()
                return GraspObject.Result(success=False, message="No valid trajectory")

            # Execute approach trajectory
            approach_success = self._execute_trajectory(trajectory[:len(trajectory)//2])
            if not approach_success:
                goal_handle.abort()
                return GraspObject.Result(success=False, message="Approach trajectory failed")

            # Execute grasp
            grasp_success = self._execute_grasp(best_grasp, object_info['properties'])
            if not grasp_success:
                goal_handle.abort()
                return GraspObject.Result(success=False, message="Grasp execution failed")

            # Lift object
            lift_success = self._execute_lift(best_grasp.position)
            if not lift_success:
                goal_handle.abort()
                return GraspObject.Result(success=False, message="Lift operation failed")

            # Report success
            goal_handle.succeed()
            return GraspObject.Result(success=True, message="Object successfully grasped")

        except Exception as e:
            self.get_logger().error(f'Grasp execution error: {e}')
            goal_handle.abort()
            return GraspObject.Result(success=False, message=f"Execution error: {str(e)}")

    def execute_place_object(self, goal_handle):
        """Execute place object action"""
        self.get_logger().info(f'Attempting to place object at {goal_handle.request.target_pose}')

        try:
            # Plan placement trajectory
            trajectory = self.motion_planner.plan_place_trajectory(
                goal_handle.request.target_pose,
                self._get_held_object_pose(),  # Current pose of held object
                self.robot_pose,
                goal_handle.request.obstacles
            )

            if not trajectory:
                self.get_logger().error('Could not plan valid placement trajectory')
                goal_handle.abort()
                return PlaceObject.Result(success=False, message="No valid trajectory")

            # Execute trajectory to placement location
            move_success = self._execute_trajectory(trajectory[:-1])  # All but last point
            if not move_success:
                goal_handle.abort()
                return PlaceObject.Result(success=False, message="Movement to placement failed")

            # Release object
            release_success = self._execute_release()
            if not release_success:
                goal_handle.abort()
                return PlaceObject.Result(success=False, message="Object release failed")

            # Retract gripper
            retract_success = self._execute_retract()
            if not retract_success:
                self.get_logger().warn('Retraction failed, but object was released')

            # Report success
            goal_handle.succeed()
            return PlaceObject.Result(success=True, message="Object successfully placed")

        except Exception as e:
            self.get_logger().error(f'Place execution error: {e}')
            goal_handle.abort()
            return PlaceObject.Result(success=False, message=f"Execution error: {str(e)}")

    def _get_object_info(self, object_name: str) -> Dict:
        """Get information about a detected object"""
        # This would interface with the object detection system
        # For now, return a mock object
        return {
            'pose': {
                'position': {'x': 0.5, 'y': 0.2, 'z': 0.1},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            },
            'properties': ObjectProperties(
                dimensions=(0.05, 0.15, 0.05),
                mass=0.3,
                center_of_mass=(0.0, 0.0, 0.0),
                surface_texture='smooth',
                material='plastic',
                stability=0.8
            ),
            'obstacles': []  # List of nearby obstacles
        }

    def _execute_trajectory(self, waypoints: List[Tuple[float, float, float]]) -> bool:
        """Execute a trajectory defined by waypoints"""
        # Convert waypoints to joint trajectory
        joint_trajectory = self._waypoints_to_joint_trajectory(waypoints)

        # Publish trajectory
        self.joint_trajectory_pub.publish(joint_trajectory)

        # Wait for execution (in practice, monitor feedback)
        import time
        time.sleep(2)  # Simulate execution time

        # Check if successful (in practice, use feedback)
        return True

    def _waypoints_to_joint_trajectory(self, waypoints: List) -> JointTrajectory:
        """Convert Cartesian waypoints to joint trajectory"""
        # This would use inverse kinematics to convert Cartesian to joint space
        # For now, return a simple trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

        # Create trajectory points
        for i, waypoint in enumerate(waypoints):
            point = JointTrajectoryPoint()
            # Inverse kinematics would go here
            point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder
            point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            point.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            point.time_from_start = Duration(sec=i, nanosec=0)
            trajectory.points.append(point)

        return trajectory

    def _execute_grasp(self, grasp_candidate: GraspCandidate, object_props: ObjectProperties) -> bool:
        """Execute the grasp motion"""
        # Calculate appropriate grasp force
        grasp_params = self.force_controller.adaptive_grasp_control(
            object_props, current_force=0.0, desired_force=15.0)

        # Command gripper
        gripper_cmd = GripperCommand()
        gripper_cmd.position = grasp_candidate.grasp_width / 2  # Half for symmetric gripper
        gripper_cmd.max_effort = grasp_params['commanded_force']
        self.gripper_command_pub.publish(gripper_cmd)

        # Wait for grasp completion
        import time
        time.sleep(1)

        # Verify grasp success through force sensing
        if self.current_wrench:
            applied_force = (self.current_wrench.wrench.force.x**2 +
                           self.current_wrench.wrench.force.y**2 +
                           self.current_wrench.wrench.force.z**2)**0.5
            return applied_force > 5.0  # Minimum force indicates grasp

        return True  # Assume success if no force feedback

    def _execute_lift(self, grasp_position: Tuple[float, float, float]) -> bool:
        """Lift the grasped object"""
        # Move up by 5cm
        lift_position = (
            grasp_position[0],
            grasp_position[1],
            grasp_position[2] + 0.05
        )

        # Plan and execute lift trajectory
        waypoints = [grasp_position, lift_position]
        return self._execute_trajectory(waypoints)

    def _execute_release(self) -> bool:
        """Release the grasped object"""
        # Open gripper fully
        gripper_cmd = GripperCommand()
        gripper_cmd.position = 0.04  # Fully open (max aperture is 0.08, so half for each finger)
        gripper_cmd.max_effort = 0.0  # Minimal effort for release
        self.gripper_command_pub.publish(gripper_cmd)

        import time
        time.sleep(0.5)  # Wait for release

        return True

    def _execute_retract(self) -> bool:
        """Retract gripper after release"""
        # Move gripper up and away from placed object
        current_pos = self.current_joint_states.position if self.current_joint_states else [0]*7
        # Plan simple retraction motion
        return True  # Simplified for example

    def _get_held_object_pose(self) -> Dict:
        """Get the current pose of the held object"""
        # This would calculate the object pose based on end-effector pose and grasp offset
        return {
            'position': {'x': 0.6, 'y': 0.0, 'z': 0.5},
            'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
        }

    def joint_state_callback(self, msg: JointState):
        """Update joint state"""
        self.current_joint_states = msg

    def wrench_callback(self, msg: WrenchStamped):
        """Update force/torque information"""
        self.current_wrench = msg
```

## Humanoid-Specific Manipulation Considerations

### Bipedal Balance During Manipulation

Maintaining balance during manipulation tasks:

```python
class BalanceAwareManipulation:
    def __init__(self):
        self.balance_margin = 0.05  # meters from support polygon edge
        self.com_velocity_limit = 0.1  # m/s
        self.zmp_stability_threshold = 0.03  # meters

    def plan_balance_aware_trajectory(self, manipulation_task: Dict,
                                    robot_state: Dict) -> Optional[List]:
        """Plan manipulation trajectory considering balance constraints"""
        # Calculate center of mass trajectory
        com_trajectory = self._calculate_com_trajectory(manipulation_task, robot_state)

        # Verify balance constraints
        if self._is_balance_safe(com_trajectory, robot_state):
            return self._generate_manipulation_trajectory(manipulation_task, robot_state)
        else:
            # Plan compensatory motion
            compensatory_plan = self._plan_balance_compensation(manipulation_task, robot_state)
            if compensatory_plan:
                return compensatory_plan

        return None

    def _calculate_com_trajectory(self, task: Dict, robot_state: Dict) -> List:
        """Calculate expected center of mass trajectory"""
        # Simplified calculation - in practice would use full dynamics model
        return []  # Placeholder

    def _is_balance_safe(self, com_trajectory: List, robot_state: Dict) -> bool:
        """Check if planned motion maintains balance"""
        # Check if center of mass stays within support polygon
        # Check if ZMP (Zero Moment Point) stays within stable region
        return True  # Placeholder

    def _plan_balance_compensation(self, task: Dict, robot_state: Dict) -> Optional[List]:
        """Plan compensatory motion to maintain balance"""
        # Strategies: adjust stance, move COM, use arm for counterbalance
        return []  # Placeholder
```

## Learning Objectives

After completing this chapter, you should be able to:
- Plan optimal grasp points based on object properties and geometry
- Implement motion planning for manipulation tasks with collision avoidance
- Apply force control techniques for safe object interaction
- Integrate manipulation systems with ROS 2 action servers
- Consider balance constraints in humanoid manipulation
- Implement feedback systems for grasp verification

## Key Takeaways

- Grasp planning requires analysis of object properties and geometry
- Motion planning must consider collision avoidance and kinematic constraints
- Force control is essential for safe and successful manipulation
- Balance maintenance is critical for humanoid robots during manipulation
- Feedback systems enable adaptive and robust manipulation
- Integration with perception systems enables autonomous manipulation