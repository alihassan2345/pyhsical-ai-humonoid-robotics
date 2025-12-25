---
sidebar_label: "Navigation and Obstacle Avoidance"
---

# Navigation and Obstacle Avoidance

## Introduction

Navigation and obstacle avoidance form the foundation of mobile robot capabilities in the Autonomous Humanoid system. This chapter explores the integration of path planning, localization, mapping, and dynamic obstacle avoidance to enable safe and efficient movement in complex environments. The system must navigate to user-specified destinations while avoiding both static obstacles and dynamic hazards.

## Navigation System Architecture

### Overview of Navigation Stack

The navigation system follows the standard ROS 2 navigation stack architecture:

```
Goal → Global Planner → Local Planner → Controller → Robot Motion
                     ↓
                Costmap Updates (Static & Dynamic)
```

### Key Components

1. **Global Planner**: Computes optimal path from start to goal
2. **Local Planner**: Executes path while avoiding local obstacles
3. **Costmaps**: Represent obstacles and navigation costs
4. **Transform System**: Maintains coordinate frame relationships
5. **Sensors**: Provide environment perception for navigation

## Global Path Planning

### A* Path Planning Algorithm

The global planner uses A* algorithm for optimal pathfinding:

```python
import heapq
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Node:
    x: int
    y: int
    cost: float
    heuristic: float
    parent: Optional['Node'] = None

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class GlobalPlanner:
    def __init__(self, resolution: float = 0.05, inflation_radius: float = 0.5):
        self.resolution = resolution
        self.inflation_radius = inflation_radius
        self.costmap = None

    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float],
                  costmap: np.ndarray) -> List[Tuple[float, float]]:
        """Plan path using A* algorithm"""
        self.costmap = costmap

        # Convert world coordinates to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)

        # Run A* algorithm
        path_grid = self._a_star(start_grid, goal_grid)

        # Convert back to world coordinates
        path_world = [self._grid_to_world(pos) for pos in path_grid]

        return path_world

    def _a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* path planning implementation"""
        open_set = []
        closed_set = set()

        start_node = Node(start[0], start[1], 0, self._heuristic(start, goal))
        heapq.heappush(open_set, start_node)

        g_costs = {start: 0}
        f_costs = {start: self._heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)

            if (current.x, current.y) == goal:
                return self._reconstruct_path(current)

            closed_set.add((current.x, current.y))

            # Explore neighbors
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                neighbor_pos = (current.x + dx, current.y + dy)

                if not self._is_valid_position(neighbor_pos):
                    continue

                if neighbor_pos in closed_set:
                    continue

                # Calculate movement cost (considering diagonal vs orthogonal)
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1.0
                tentative_g = g_costs[(current.x, current.y)] + move_cost

                # Check if this path is better
                if neighbor_pos not in g_costs or tentative_g < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = tentative_g
                    h_cost = self._heuristic(neighbor_pos, goal)
                    f_costs[neighbor_pos] = tentative_g + h_cost

                    neighbor_node = Node(
                        neighbor_pos[0],
                        neighbor_pos[1],
                        tentative_g,
                        h_cost,
                        current
                    )

                    heapq.heappush(open_set, neighbor_node)

        return []  # No path found

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate heuristic (Manhattan distance with diagonal consideration)"""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        return min(dx, dy) * 1.414 + abs(dx - dy)

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not occupied)"""
        if self.costmap is None:
            return False

        x, y = pos
        if x < 0 or x >= self.costmap.shape[1] or y < 0 or y >= self.costmap.shape[0]:
            return False

        # Check if cell is occupied (cost > 50 on 0-100 scale)
        return self.costmap[y, x] < 50

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start"""
        path = []
        current = node
        while current:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]  # Reverse to get start-to-goal path

    def _world_to_grid(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y = world_pos
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return (grid_x, grid_y)

    def _grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        grid_x, grid_y = grid_pos
        world_x = grid_x * self.resolution
        world_y = grid_y * self.resolution
        return (world_x, world_y)
```

## Costmap Management

### Static and Dynamic Costmaps

The navigation system maintains multiple costmaps for different purposes:

```python
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
import open3d as o3d

class CostmapManager:
    def __init__(self, width: int, height: int, resolution: float):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Static costmap (for known obstacles)
        self.static_costmap = np.zeros((height, width), dtype=np.uint8)

        # Local costmap (for dynamic obstacles)
        self.local_costmap = np.zeros((height, width), dtype=np.uint8)

        # Combined costmap
        self.combined_costmap = np.zeros((height, width), dtype=np.uint8)

        # Inflation parameters
        self.inflation_radius = 0.5  # meters
        self.cost_scaling_factor = 3.0

    def update_static_costmap(self, map_msg: OccupancyGrid):
        """Update static costmap from map server"""
        self.width = map_msg.info.width
        self.height = map_msg.info.height
        self.resolution = map_msg.info.resolution
        self.origin_x = map_msg.info.origin.position.x
        self.origin_y = map_msg.info.origin.position.y

        # Convert occupancy grid to numpy array
        data = np.array(map_msg.data).reshape((self.height, self.width))
        # Convert from -1 (unknown), 0-100 (occupied percentage) to 0-255 cost
        self.static_costmap = np.clip(data, 0, 100).astype(np.uint8)

        # Inflate obstacles
        self._inflate_obstacles(self.static_costmap)

    def update_local_costmap(self, laser_scan: LaserScan, robot_pose):
        """Update local costmap with dynamic obstacles from laser scan"""
        # Reset local costmap
        self.local_costmap.fill(0)

        # Convert laser scan to obstacle points in robot frame
        obstacle_points = self._laser_scan_to_points(laser_scan)

        # Transform to map frame
        obstacle_points_map = self._transform_points_to_map(
            obstacle_points, robot_pose)

        # Add obstacles to local costmap
        self._add_obstacles_to_costmap(obstacle_points_map, self.local_costmap)

        # Inflate local obstacles
        self._inflate_obstacles(self.local_costmap)

    def _laser_scan_to_points(self, scan: LaserScan) -> List[Tuple[float, float]]:
        """Convert laser scan to point cloud"""
        points = []

        for i, range_val in enumerate(scan.ranges):
            if scan.range_min <= range_val <= scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                points.append((x, y))

        return points

    def _transform_points_to_map(self, points: List[Tuple[float, float]],
                                robot_pose) -> List[Tuple[int, int]]:
        """Transform points from robot frame to map frame"""
        transformed_points = []

        # Extract robot position and orientation
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y
        robot_yaw = self._quaternion_to_yaw(robot_pose.orientation)

        cos_yaw = np.cos(robot_yaw)
        sin_yaw = np.sin(robot_yaw)

        for x, y in points:
            # Rotate and translate
            map_x = x * cos_yaw - y * sin_yaw + robot_x
            map_y = x * sin_yaw + y * cos_yaw + robot_y

            # Convert to grid coordinates
            grid_x = int((map_x - self.origin_x) / self.resolution)
            grid_y = int((map_y - self.origin_y) / self.resolution)

            if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                transformed_points.append((grid_x, grid_y))

        return transformed_points

    def _add_obstacles_to_costmap(self, obstacle_points: List[Tuple[int, int]],
                                 costmap: np.ndarray):
        """Add obstacle points to costmap"""
        for x, y in obstacle_points:
            if 0 <= x < self.width and 0 <= y < self.height:
                costmap[y, x] = 254  # Mark as obstacle

    def _inflate_obstacles(self, costmap: np.ndarray):
        """Inflate obstacles by inflation radius"""
        inflation_cells = int(self.inflation_radius / self.resolution)

        height, width = costmap.shape
        inflated = costmap.copy()

        for y in range(height):
            for x in range(width):
                if costmap[y, x] > 50:  # If it's an obstacle
                    # Inflate around this obstacle
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        for dx in range(-inflation_cells, inflation_cells + 1):
                            ny, nx = y + dy, x + dx

                            if 0 <= ny < height and 0 <= nx < width:
                                # Calculate distance-based cost
                                distance = np.sqrt(dx**2 + dy**2) * self.resolution

                                if distance <= self.inflation_radius:
                                    cost = min(254, int(254 * (1 - distance / self.inflation_radius)))
                                    inflated[ny, nx] = max(inflated[ny, nx], cost)

        costmap[:] = inflated

    def get_combined_costmap(self) -> np.ndarray:
        """Get combined costmap (static + local)"""
        # Combine static and local costmaps
        self.combined_costmap = np.maximum(self.static_costmap, self.local_costmap)
        return self.combined_costmap

    def _quaternion_to_yaw(self, orientation) -> float:
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return np.arctan2(siny_cosp, cosy_cosp)
```

## Local Path Following and Obstacle Avoidance

### Dynamic Window Approach (DWA)

The local planner uses DWA for dynamic obstacle avoidance:

```python
from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class VelocitySample:
    linear: float
    angular: float
    score: float

class LocalPlanner:
    def __init__(self):
        # Robot parameters
        self.max_vel_x = 0.5  # m/s
        self.max_vel_theta = 1.0  # rad/s
        self.min_vel_x = 0.1  # m/s
        self.min_vel_theta = 0.1  # rad/s

        # DWA parameters
        self.sim_time = 1.5  # seconds to simulate
        self.sim_granularity = 0.05  # time step
        self.vx_samples = 10
        self.vtheta_samples = 20

        # Cost function weights
        self.heading_weight = 0.2
        self.vel_weight = 0.1
        self.clearance_weight = 0.2

        # Robot dimensions
        self.robot_radius = 0.3  # meters

    def calculate_velocity_commands(self, robot_pose, robot_vel,
                                   goal_pose, costmap) -> Tuple[float, float]:
        """Calculate optimal velocity commands using DWA"""
        # Generate velocity samples
        velocity_samples = self._generate_velocity_samples(robot_vel)

        # Evaluate each sample
        best_score = float('-inf')
        best_vel = (0.0, 0.0)

        for vel_sample in velocity_samples:
            score = self._evaluate_trajectory(
                robot_pose, vel_sample, goal_pose, costmap)

            if score > best_score:
                best_score = score
                best_vel = (vel_sample.linear, vel_sample.angular)

        return best_vel

    def _generate_velocity_samples(self, current_vel) -> List[VelocitySample]:
        """Generate velocity samples for DWA"""
        samples = []

        # Calculate velocity ranges based on current velocity and limits
        dvx = (self.max_vel_x - self.min_vel_x) / self.vx_samples
        dvtheta = (self.max_vel_theta - self.min_vel_theta) / self.vtheta_samples

        for i in range(self.vx_samples):
            for j in range(self.vtheta_samples):
                # Positive and negative angular velocities
                for sign in [1, -1]:
                    vx = self.min_vel_x + i * dvx
                    vtheta = sign * (self.min_vel_theta + j * dvtheta)

                    # Check if within acceleration limits
                    if self._is_valid_velocity(vx, vtheta, current_vel):
                        score = 0.0  # Will be calculated during evaluation
                        samples.append(VelocitySample(vx, vtheta, score))

        return samples

    def _is_valid_velocity(self, vx: float, vtheta: float, current_vel) -> bool:
        """Check if velocity is valid given acceleration limits"""
        # For simplicity, just check velocity limits
        return (abs(vx) <= self.max_vel_x and
                abs(vtheta) <= self.max_vel_theta)

    def _evaluate_trajectory(self, robot_pose, vel_sample: VelocitySample,
                           goal_pose, costmap) -> float:
        """Evaluate a trajectory sample"""
        # Simulate the trajectory
        trajectory = self._simulate_trajectory(robot_pose, vel_sample)

        # Calculate scores
        heading_score = self._calculate_heading_score(trajectory, goal_pose)
        clearance_score = self._calculate_clearance_score(trajectory, costmap)
        velocity_score = self._calculate_velocity_score(vel_sample)

        # Weighted combination
        total_score = (self.heading_weight * heading_score +
                      self.clearance_weight * clearance_score +
                      self.vel_weight * velocity_score)

        return total_score

    def _simulate_trajectory(self, start_pose, vel_sample: VelocitySample) -> List:
        """Simulate trajectory for given velocity commands"""
        trajectory = []
        dt = self.sim_granularity

        # Start from current pose
        x = start_pose.position.x
        y = start_pose.position.y
        theta = self._quaternion_to_yaw(start_pose.orientation)

        vx = vel_sample.linear
        vtheta = vel_sample.angular

        # Simulate for sim_time
        time = 0
        while time < self.sim_time:
            # Update position
            x += vx * math.cos(theta) * dt
            y += vx * math.sin(theta) * dt
            theta += vtheta * dt

            trajectory.append((x, y))
            time += dt

        return trajectory

    def _calculate_heading_score(self, trajectory: List, goal_pose) -> float:
        """Calculate score based on heading toward goal"""
        if not trajectory:
            return 0.0

        # Look at the end of the trajectory
        final_x, final_y = trajectory[-1]
        goal_x = goal_pose.position.x
        goal_y = goal_pose.position.y

        # Calculate distance to goal
        dist_to_goal = math.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)

        # Score is higher for trajectories that get closer to goal
        # Normalize by maximum possible distance
        max_score = 1.0
        score = max(0, max_score - dist_to_goal / 10.0)  # Assume 10m is far

        return score

    def _calculate_clearance_score(self, trajectory: List, costmap) -> float:
        """Calculate score based on obstacle clearance"""
        min_clearance = float('inf')

        for x, y in trajectory:
            # Convert to costmap coordinates
            costmap_x = int((x - costmap.origin_x) / costmap.resolution)
            costmap_y = int((y - costmap.origin_y) / costmap.resolution)

            if (0 <= costmap_x < costmap.width and
                0 <= costmap_y < costmap.height):
                cost = costmap.combined_costmap[costmap_y, costmap_x]
                if cost > 50:  # Significant cost
                    return 0.0  # Trajectory hits obstacle

        # Return score based on minimum clearance
        return min(1.0, len(trajectory) / 10.0)  # Simple scoring

    def _calculate_velocity_score(self, vel_sample: VelocitySample) -> float:
        """Calculate score based on velocity"""
        # Prefer higher velocities (but not too high)
        max_vel = self.max_vel_x
        vel_ratio = vel_sample.linear / max_vel
        return min(1.0, vel_ratio)

    def _quaternion_to_yaw(self, orientation) -> float:
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)
```

## Integration with Navigation Stack

### ROS 2 Navigation Interface

The navigation system integrates with ROS 2 Navigation2 stack:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import tf_transformations

class HumanoidNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_navigator')

        # Initialize navigation components
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()
        self.costmap_manager = CostmapManager(200, 200, 0.05)

        # Action server for navigation
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_navigate_to_pose
        )

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10
        )

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Robot state
        self.current_pose = None
        self.current_velocity = Twist()
        self.global_path = []
        self.is_navigating = False

        self.get_logger().info('Humanoid Navigator Initialized')

    def execute_navigate_to_pose(self, goal_handle):
        """Execute navigation to pose goal"""
        self.get_logger().info(f'Navigating to goal: {goal_handle.request.pose.pose.position}')

        # Get current robot pose
        robot_pose = self._get_current_pose()
        if robot_pose is None:
            self.get_logger().error('Cannot get current robot pose')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Get current costmap
        costmap = self.costmap_manager.get_combined_costmap()
        if costmap is None:
            self.get_logger().error('Cannot get current costmap')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Plan global path
        goal_pos = (
            goal_handle.request.pose.pose.position.x,
            goal_handle.request.pose.pose.position.y
        )
        robot_pos = (
            robot_pose.position.x,
            robot_pose.position.y
        )

        path = self.global_planner.plan_path(robot_pos, goal_pos, costmap)
        if not path:
            self.get_logger().error('Cannot find path to goal')
            goal_handle.abort()
            return NavigateToPose.Result()

        self.global_path = path
        self.is_navigating = True

        # Follow path until goal reached
        result = self._follow_path(goal_handle, goal_pos)

        if result:
            self.get_logger().info('Successfully reached goal')
            goal_handle.succeed()
        else:
            self.get_logger().error('Failed to reach goal')
            goal_handle.abort()

        self.is_navigating = False
        return result

    def _follow_path(self, goal_handle, goal_pos: tuple) -> NavigateToPose.Result:
        """Follow the planned path with obstacle avoidance"""
        path_index = 0

        while self.is_navigating and not goal_handle.is_cancel_requested:
            # Get current state
            current_pose = self._get_current_pose()
            if current_pose is None:
                continue

            # Update costmap with latest sensor data
            costmap = self.costmap_manager.get_combined_costmap()

            # Check if goal reached
            current_pos = (current_pose.position.x, current_pose.position.y)
            distance_to_goal = self._calculate_distance(current_pos, goal_pos)

            if distance_to_goal < 0.5:  # 50cm tolerance
                return NavigateToPose.Result()

            # Get next path segment to follow
            look_ahead_point = self._get_look_ahead_point(current_pos, path_index)

            if look_ahead_point:
                # Calculate velocity commands
                vel_cmd = self.local_planner.calculate_velocity_commands(
                    current_pose, self.current_velocity,
                    self._create_pose_stamped(look_ahead_point[0], look_ahead_point[1]),
                    costmap
                )

                # Publish velocity command
                cmd_msg = Twist()
                cmd_msg.linear.x = vel_cmd[0]
                cmd_msg.angular.z = vel_cmd[1]
                self.cmd_vel_pub.publish(cmd_msg)

                # Update path index if close to current look-ahead point
                if self._calculate_distance(current_pos, look_ahead_point) < 0.2:
                    path_index += 1

            # Rate limiting
            self._rate.sleep()

        return NavigateToPose.Result()

    def _get_current_pose(self):
        """Get current robot pose from TF"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            return transform.transform.translation
        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')
            return None

    def _get_look_ahead_point(self, current_pos: tuple, path_index: int) -> tuple:
        """Get point on path to look ahead to"""
        if path_index >= len(self.global_path):
            return None

        # Return point that's ~1 meter ahead on the path
        look_ahead_distance = 1.0
        total_distance = 0

        for i in range(path_index, len(self.global_path) - 1):
            p1 = self.global_path[i]
            p2 = self.global_path[i + 1]
            segment_distance = self._calculate_distance(p1, p2)

            if total_distance + segment_distance >= look_ahead_distance:
                # Interpolate point along this segment
                remaining_distance = look_ahead_distance - total_distance
                ratio = remaining_distance / segment_distance
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                return (x, y)

            total_distance += segment_distance

        # If path is shorter than look-ahead distance, return last point
        if self.global_path:
            return self.global_path[-1]

        return None

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def _create_pose_stamped(self, x: float, y: float) -> PoseStamped:
        """Create a PoseStamped message"""
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        return pose

    def laser_callback(self, msg: LaserScan):
        """Update local costmap with laser scan data"""
        if self.current_pose:
            self.costmap_manager.update_local_costmap(msg, self.current_pose)
```

## Humanoid-Specific Navigation Considerations

### Bipedal Locomotion Constraints

Humanoid robots have unique navigation constraints:

```python
class HumanoidSpecificNavigation:
    def __init__(self):
        # Bipedal-specific parameters
        self.step_size_limit = 0.3  # Maximum step size
        self.turn_radius_limit = 0.4  # Minimum turning radius
        self.balance_constraints = {
            'max_inclination': 15.0,  # degrees
            'zmp_limits': (-0.1, 0.1)  # Zero Moment Point limits
        }
        self.footprint = self._calculate_humanoid_footprint()

    def _calculate_humanoid_footprint(self) -> List[Tuple[float, float]]:
        """Calculate humanoid robot footprint for collision checking"""
        # Approximate humanoid as a rectangle with safety margin
        width = 0.4  # meters
        length = 0.6  # meters
        safety_margin = 0.1  # meters

        half_width = width/2 + safety_margin
        half_length = length/2 + safety_margin

        return [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, half_width),
            (half_length, -half_width)
        ]

    def adjust_path_for_bipedal_constraints(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Adjust path to respect bipedal locomotion constraints"""
        if len(path) < 2:
            return path

        adjusted_path = [path[0]]

        for i in range(1, len(path)):
            prev_point = adjusted_path[-1]
            current_point = path[i]

            # Calculate direction vector
            dx = current_point[0] - prev_point[0]
            dy = current_point[1] - prev_point[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if distance > self.step_size_limit:
                # Interpolate additional points
                steps_needed = math.ceil(distance / self.step_size_limit)
                for step in range(1, steps_needed + 1):
                    ratio = step / steps_needed
                    new_x = prev_point[0] + ratio * dx
                    new_y = prev_point[1] + ratio * dy
                    adjusted_path.append((new_x, new_y))
            else:
                adjusted_path.append(current_point)

        return adjusted_path

    def validate_path_for_balance(self, path: List[Tuple[float, float]], costmap) -> bool:
        """Validate path considering balance constraints"""
        for point in path:
            # Check if point is in a stable area
            if not self._is_stable_surface(point, costmap):
                return False

        return True

    def _is_stable_surface(self, point: Tuple[float, float], costmap) -> bool:
        """Check if surface at point is stable for bipedal walking"""
        # Check for slopes, holes, and unstable surfaces
        # This would integrate with perception systems to detect terrain properties
        return True  # Placeholder
```

## Safety and Human-Aware Navigation

### Socially-Aware Navigation

The system considers human presence and social conventions:

```python
class SociallyAwareNavigation:
    def __init__(self):
        self.personal_space_radius = 0.8  # meters
        self.social_navigation_rules = {
            'right_passing': True,
            'group_awareness': True,
            'eye_contact_zones': True
        }

    def modify_path_for_social_awareness(self, original_path: List[Tuple[float, float]],
                                       human_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Modify path to respect human personal space and social conventions"""
        if not human_positions:
            return original_path

        modified_path = []
        for point in original_path:
            # Check distance to humans
            min_distance = float('inf')
            for human_pos in human_positions:
                dist = self._calculate_distance(point, human_pos)
                min_distance = min(min_distance, dist)

            if min_distance < self.personal_space_radius:
                # Find alternative path around human
                detour_point = self._calculate_detour(point, human_positions)
                modified_path.append(detour_point)
            else:
                modified_path.append(point)

        return modified_path

    def _calculate_detour(self, original_point: Tuple[float, float],
                         human_positions: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate detour point to maintain personal space"""
        # Simple strategy: move perpendicular to the direction of the closest human
        closest_human = min(human_positions,
                           key=lambda h: self._calculate_distance(original_point, h))

        dx = original_point[0] - closest_human[0]
        dy = original_point[1] - closest_human[1]
        distance = math.sqrt(dx*dx + dy*dy)

        if distance > 0:
            # Normalize and move perpendicular
            nx, ny = dx/distance, dy/distance
            # Move to maintain personal space
            detour_x = closest_human[0] + nx * self.personal_space_radius
            detour_y = closest_human[1] + ny * self.personal_space_radius
            return (detour_x, detour_y)

        return original_point
```

## Performance Optimization

### Multi-Scale Path Planning

For efficiency, the system uses multi-scale planning:

```python
class MultiScaleNavigation:
    def __init__(self):
        self.global_scale_resolution = 0.5  # Coarse for global planning
        self.local_scale_resolution = 0.05  # Fine for local planning
        self.adaptive_threshold = 5.0  # Switch to fine planning when close

    def plan_path_multiscale(self, start: Tuple[float, float],
                           goal: Tuple[float, float],
                           costmap: np.ndarray) -> List[Tuple[float, float]]:
        """Plan path using multi-scale approach"""
        # First, plan on global scale
        global_path = self._plan_coarse_path(start, goal, costmap)

        # For segments close to goal, refine with fine resolution
        refined_path = []
        for i, point in enumerate(global_path):
            distance_to_goal = self._calculate_distance(point, goal)

            if distance_to_goal < self.adaptive_threshold:
                # Use fine-scale planning for the remainder
                refined_path.extend(self._plan_fine_path(point, goal, costmap))
                break
            else:
                refined_path.append(point)

        return refined_path

    def _plan_coarse_path(self, start: Tuple[float, float],
                         goal: Tuple[float, float],
                         costmap: np.ndarray) -> List[Tuple[float, float]]:
        """Plan path with coarse resolution"""
        # Downsample costmap
        coarse_costmap = self._downsample_costmap(costmap, self.global_scale_resolution)
        coarse_start = self._coarsen_point(start, self.global_scale_resolution)
        coarse_goal = self._coarsen_point(goal, self.global_scale_resolution)

        # Plan path
        coarse_path = GlobalPlanner().plan_path(coarse_start, coarse_goal, coarse_costmap)

        # Upsample back to fine resolution
        fine_path = [self._refine_point(p, self.global_scale_resolution) for p in coarse_path]

        return fine_path

    def _downsample_costmap(self, costmap: np.ndarray, factor: float) -> np.ndarray:
        """Downsample costmap by given factor"""
        # Implementation would use appropriate downsampling technique
        return costmap[::int(1/factor), ::int(1/factor)]  # Simple subsampling

    def _coarsen_point(self, point: Tuple[float, float], resolution: float) -> Tuple[float, float]:
        """Convert fine-resolution point to coarse resolution"""
        x, y = point
        return (x // resolution * resolution, y // resolution * resolution)
```

## Learning Objectives

After completing this chapter, you should be able to:
- Implement global and local path planning algorithms
- Manage static and dynamic costmaps for navigation
- Apply dynamic window approach for obstacle avoidance
- Integrate navigation with ROS 2 navigation stack
- Consider humanoid-specific constraints in navigation
- Implement socially-aware navigation for human environments

## Key Takeaways

- Navigation requires coordination between global planning and local obstacle avoidance
- Costmaps provide essential information for safe navigation
- Humanoid robots have unique locomotion constraints
- Socially-aware navigation improves human-robot interaction
- Multi-scale approaches optimize navigation performance