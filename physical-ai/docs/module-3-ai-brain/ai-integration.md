---
sidebar_position: 6
---

# AI Integration with Isaac Platform

## Combining Isaac Tools with AI Models

This module covers how to integrate Isaac tools with AI models for advanced robotics applications. It demonstrates how to combine Isaac's perception, navigation, and simulation capabilities with modern AI techniques.

### Isaac AI Architecture

Isaac provides several pathways for AI integration:

- **Isaac ROS**: Hardware-accelerated AI packages for perception and control
- **Isaac Sim**: High-fidelity simulation for AI training and testing
- **Isaac Apps**: Pre-built AI applications for common robotics tasks
- **Isaac Extensions**: Custom AI extensions for specific use cases

### Isaac ROS AI Integration

#### Hardware-Accelerated Perception

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import torch

class IsaacAIPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_ai_perception_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Isaac ROS AI perception components
        self.initialize_isaac_ai_components()

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Subscribe to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers for AI results
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/isaac_ai/detections',
            10
        )
        self.feature_pub = self.create_publisher(
            PointStamped,
            '/isaac_ai/features',
            10
        )

        # Store camera info
        self.camera_info = None

    def initialize_isaac_ai_components(self):
        """Initialize Isaac AI components"""
        try:
            # Initialize Isaac-accelerated perception models
            # This would typically use Isaac's optimized DNN packages
            from isaac_ros_messages.msg import IsaacROSImage

            # Initialize detection model
            self.detection_model = self.load_isaac_optimized_model('detection')

            # Initialize feature extraction model
            self.feature_model = self.load_isaac_optimized_model('features')

            # Initialize depth estimation model
            self.depth_model = self.load_isaac_optimized_model('depth')

            self.get_logger().info('Isaac AI components initialized successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to initialize Isaac AI components: {e}')

    def load_isaac_optimized_model(self, model_type):
        """Load Isaac-optimized AI model"""
        # This would load models optimized for Isaac's hardware acceleration
        # In a real implementation, this would interface with Isaac's DNN packages

        if model_type == 'detection':
            # Load object detection model
            model = {
                'type': 'detection',
                'input_shape': (3, 640, 640),
                'classes': ['person', 'cup', 'bottle', 'chair', 'table'],
                'confidence_threshold': 0.5
            }
        elif model_type == 'features':
            # Load feature extraction model
            model = {
                'type': 'features',
                'input_shape': (3, 224, 224),
                'output_dim': 512
            }
        elif model_type == 'depth':
            # Load depth estimation model
            model = {
                'type': 'depth',
                'input_shape': (3, 480, 640),
                'output_channels': 1
            }
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        return model

    def image_callback(self, msg):
        """Process image with Isaac AI models"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

            # Run Isaac-optimized detection
            detections = self.run_isaac_detection(cv_image)

            # Run feature extraction
            features = self.run_isaac_feature_extraction(cv_image)

            # Publish results
            if detections:
                self.publish_detections(detections, msg.header)

            if features:
                self.publish_features(features, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def run_isaac_detection(self, image):
        """Run Isaac-optimized object detection"""
        try:
            # Resize image to model input size
            input_height, input_width = self.detection_model['input_shape'][1:]
            resized_image = cv2.resize(image, (input_width, input_height))

            # Normalize image
            normalized_image = resized_image.astype(np.float32) / 255.0
            normalized_image = np.transpose(normalized_image, (2, 0, 1))  # CHW format
            normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

            # In a real implementation, this would run the Isaac-optimized model
            # For simulation, we'll create mock detections
            mock_detections = self.create_mock_detections(image.shape[0], image.shape[1])

            return mock_detections

        except Exception as e:
            self.get_logger().error(f'Error in Isaac detection: {e}')
            return []

    def create_mock_detections(self, img_height, img_width):
        """Create mock detections for demonstration"""
        import random

        # Simulate detection results
        num_detections = random.randint(0, 3)
        detections = []

        for i in range(num_detections):
            # Random class
            classes = self.detection_model['classes']
            class_idx = random.randint(0, len(classes) - 1)
            class_name = classes[class_idx]

            # Random bounding box
            x = random.randint(0, img_width - 100)
            y = random.randint(0, img_height - 100)
            w = random.randint(50, 150)
            h = random.randint(50, 150)

            # Ensure bounding box is within image bounds
            x = min(x, img_width - w)
            y = min(y, img_height - h)

            detection = {
                'class_name': class_name,
                'confidence': random.uniform(0.5, 0.99),
                'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
            }

            detections.append(detection)

        return detections

    def run_isaac_feature_extraction(self, image):
        """Run Isaac-optimized feature extraction"""
        try:
            # Resize image to model input size
            input_height, input_width = self.feature_model['input_shape'][1:]
            resized_image = cv2.resize(image, (input_width, input_height))

            # Normalize image
            normalized_image = resized_image.astype(np.float32) / 255.0
            normalized_image = np.transpose(normalized_image, (2, 0, 1))  # CHW format
            normalized_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

            # In a real implementation, this would run the Isaac-optimized model
            # For simulation, we'll create mock features
            mock_features = np.random.rand(self.feature_model['output_dim']).astype(np.float32)

            return mock_features

        except Exception as e:
            self.get_logger().error(f'Error in Isaac feature extraction: {e}')
            return None

    def publish_detections(self, detections, header):
        """Publish detection results"""
        try:
            detection_array = Detection2DArray()
            detection_array.header = header

            for det in detections:
                detection_2d = Detection2D()

                # Set class and confidence
                detection_2d.results = [ObjectHypothesisWithPose()]
                detection_2d.results[0].id = det['class_name']
                detection_2d.results[0].score = det['confidence']

                # Set bounding box
                detection_2d.bbox.center.x = det['bbox']['x'] + det['bbox']['width'] / 2
                detection_2d.bbox.center.y = det['bbox']['y'] + det['bbox']['height'] / 2
                detection_2d.bbox.size_x = det['bbox']['width']
                detection_2d.bbox.size_y = det['bbox']['height']

                detection_array.detections.append(detection_2d)

            self.detection_pub.publish(detection_array)

        except Exception as e:
            self.get_logger().error(f'Error publishing detections: {e}')

    def publish_features(self, features, header):
        """Publish extracted features"""
        try:
            # For demonstration, publish a point representing feature statistics
            feature_point = PointStamped()
            feature_point.header = header

            # Use mean of features as a representative point
            feature_point.point.x = float(np.mean(features[:170]))  # Split features across x,y,z
            feature_point.point.y = float(np.mean(features[170:340]))
            feature_point.point.z = float(np.mean(features[340:]))

            self.feature_pub.publish(feature_point)

        except Exception as e:
            self.get_logger().error(f'Error publishing features: {e}')


def main(args=None):
    rclpy.init(args=args)

    perception_node = IsaacAIPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Isaac AI Perception node stopped by user')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Isaac Sim AI Training Integration

#### Simulation for AI Training

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.sensors import Camera, ImuSensor
from omni.isaac.range_sensor import LidarRtx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

class IsaacAITrainingEnvironment:
    def __init__(self, config):
        self.config = config
        self.world = World(stage_units_in_meters=1.0)

        # Initialize AI components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_ai_models()

        # Setup simulation environment
        self.setup_simulation()

        # Initialize training parameters
        self.episode_count = 0
        self.total_steps = 0
        self.writer = SummaryWriter(log_dir=config.get('log_dir', './logs'))

    def setup_ai_models(self):
        """Setup AI models for training"""
        # Define policy network
        self.policy_network = self.create_policy_network().to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)

        # Define value network for actor-critic methods
        self.value_network = self.create_value_network().to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=3e-4)

    def create_policy_network(self):
        """Create policy network for robot control"""
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()

                # Convolutional layers for visual input
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU()
                )

                # Calculate conv output size
                conv_out_size = 64 * 7 * 7  # Assuming 84x84 input

                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(conv_out_size + state_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU()
                )

                # Output layers for action mean and std
                self.action_mean = nn.Linear(256, action_dim)
                self.action_std = nn.Linear(256, action_dim)

            def forward(self, visual_input, state_input):
                # Process visual input
                conv_out = self.conv_layers(visual_input)
                conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten

                # Concatenate with state input
                combined = torch.cat([conv_out, state_input], dim=1)

                # Process through FC layers
                features = self.fc_layers(combined)

                # Output action parameters
                mean = torch.tanh(self.action_mean(features))
                std = torch.sigmoid(self.action_std(features)) + 1e-5  # Ensure positive std

                return mean, std

        return PolicyNetwork(
            state_dim=self.config.get('state_dim', 24),
            action_dim=self.config.get('action_dim', 6)
        )

    def create_value_network(self):
        """Create value network for advantage estimation"""
        class ValueNetwork(nn.Module):
            def __init__(self, state_dim):
                super().__init__()

                # Convolutional layers for visual input
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU()
                )

                # Calculate conv output size
                conv_out_size = 64 * 7 * 7  # Assuming 84x84 input

                # Fully connected layers
                self.fc_layers = nn.Sequential(
                    nn.Linear(conv_out_size + state_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1)
                )

            def forward(self, visual_input, state_input):
                # Process visual input
                conv_out = self.conv_layers(visual_input)
                conv_out = conv_out.view(conv_out.size(0), -1)  # Flatten

                # Concatenate with state input
                combined = torch.cat([conv_out, state_input], dim=1)

                # Process through FC layers
                value = self.fc_layers(combined)

                return value.squeeze(-1)  # Remove last dimension

        return ValueNetwork(state_dim=self.config.get('state_dim', 24))

    def setup_simulation(self):
        """Setup Isaac Sim environment for training"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add robot to environment
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="training_robot",
                usd_path=self.config.get('robot_usd_path', "path/to/robot.usd"),
                position=np.array([0.0, 0.0, 0.1]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        )

        # Add sensors to robot
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Robot/base_link/chassis/camera",
                name="training_camera",
                position=np.array([0.1, 0.0, 0.1]),
                frequency=30,
                resolution=(84, 84)  # Smaller resolution for training speed
            )
        )

        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path="/World/Robot/base_link/chassis/lidar",
                name="training_lidar",
                translation=np.array([0.1, 0.0, 0.2]),
                config="Example_Rotary",
                depth_range=(0.1, 10.0),
                frequency=10
            )
        )

        # Add target objects
        self.targets = []
        for i in range(self.config.get('num_targets', 5)):
            target = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Target_{i}",
                    name=f"target_{i}",
                    position=np.array([
                        np.random.uniform(-3, 3),
                        np.random.uniform(-3, 3),
                        0.1
                    ]),
                    size=0.2,
                    color=np.array([0.0, 1.0, 0.0])  # Green targets
                )
            )
            self.targets.append(target)

    def get_observation(self):
        """Get current observation from simulation"""
        # Get robot state
        robot_position, robot_orientation = self.robot.get_world_pose()
        robot_lin_vel, robot_ang_vel = self.robot.get_linear_velocity(), self.robot.get_angular_velocity()

        # Get camera image
        camera_image = self.camera.get_rgb()
        if camera_image is not None:
            # Convert to tensor and normalize
            visual_obs = torch.FloatTensor(camera_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        else:
            # Fallback to zeros if no image available
            visual_obs = torch.zeros(1, 3, 84, 84)

        # Get LiDAR data
        lidar_data = self.lidar.get_linear_depth_data()
        if lidar_data is not None:
            lidar_obs = torch.FloatTensor(lidar_data).unsqueeze(0)
        else:
            lidar_obs = torch.zeros(1, 720)  # Assuming 720 beams

        # Combine state observations
        state_obs = torch.FloatTensor([
            robot_position[0], robot_position[1], robot_position[2],  # Position
            robot_orientation[0], robot_orientation[1], robot_orientation[2], robot_orientation[3],  # Orientation (quaternion)
            robot_lin_vel[0], robot_lin_vel[1], robot_lin_vel[2],  # Linear velocity
            robot_ang_vel[0], robot_ang_vel[1], robot_ang_vel[2],  # Angular velocity
        ]).unsqueeze(0)

        return visual_obs.to(self.device), lidar_obs.to(self.device), state_obs.to(self.device)

    def compute_reward(self, action):
        """Compute reward based on current state and action"""
        # Get current robot position
        robot_pos, _ = self.robot.get_world_pose()

        # Find closest target
        min_distance = float('inf')
        for target in self.targets:
            target_pos, _ = target.get_world_pose()
            distance = np.linalg.norm(robot_pos - target_pos)
            min_distance = min(min_distance, distance)

        # Reward components
        distance_reward = -min_distance  # Negative distance encourages getting closer
        time_penalty = -0.01  # Small penalty for each timestep
        action_penalty = -0.001 * torch.sum(action**2)  # Penalty for large actions

        # Bonus for getting very close to target
        bonus = 100.0 if min_distance < 0.3 else 0.0

        total_reward = distance_reward + time_penalty + action_penalty.item() + bonus

        return total_reward, {"distance": min_distance, "bonus": bonus}

    def reset_environment(self):
        """Reset environment to initial state"""
        # Reset robot position
        self.robot.set_world_pose(
            position=np.array([0.0, 0.0, 0.1]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Reset robot velocity
        self.robot.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.robot.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # Randomize target positions
        for target in self.targets:
            new_pos = np.array([
                np.random.uniform(-3, 3),
                np.random.uniform(-3, 3),
                0.1
            ])
            target.set_world_pose(position=new_pos)

        # Reset simulation
        self.world.reset()

    def train_step(self):
        """Perform one training step"""
        # Get initial observation
        visual_obs, lidar_obs, state_obs = self.get_observation()

        episode_reward = 0
        episode_steps = 0
        done = False

        while not done and episode_steps < self.config.get('max_episode_steps', 1000):
            # Get action from policy
            with torch.no_grad():
                action_mean, action_std = self.policy_network(visual_obs, torch.cat([lidar_obs, state_obs], dim=1))

                # Sample action from distribution
                action_dist = torch.distributions.Normal(action_mean, action_std)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=1)

            # Apply action to robot
            self.apply_action_to_robot(action.cpu().numpy())

            # Step simulation
            self.world.step(render=True)

            # Get next observation and reward
            next_visual_obs, next_lidar_obs, next_state_obs = self.get_observation()
            reward, info = self.compute_reward(action)
            episode_reward += reward

            # Compute value for advantage calculation
            with torch.no_grad():
                value = self.value_network(next_visual_obs, torch.cat([next_lidar_obs, next_state_obs], dim=1))

            # Update observations
            visual_obs, lidar_obs, state_obs = next_visual_obs, next_lidar_obs, next_state_obs

            # Perform learning step (simplified)
            self.update_networks(visual_obs, lidar_obs, state_obs, action, reward, value, log_prob)

            episode_steps += 1
            self.total_steps += 1

            # Check termination conditions
            robot_pos, _ = self.robot.get_world_pose()
            if np.any(np.abs(robot_pos[:2]) > 5.0):  # Out of bounds
                done = True
            elif info['distance'] < 0.3:  # Reached target
                done = True

        # Log episode information
        self.writer.add_scalar('Reward/Episode', episode_reward, self.episode_count)
        self.writer.add_scalar('Steps/Episode', episode_steps, self.episode_count)
        self.writer.add_scalar('Distance/Min', info['distance'], self.episode_count)

        self.episode_count += 1

        # Reset environment for next episode
        self.reset_environment()

        return episode_reward, episode_steps

    def apply_action_to_robot(self, action):
        """Apply action to robot in simulation"""
        # This would convert the neural network output to robot commands
        # For a differential drive robot, action[0] might be linear velocity, action[1] angular velocity
        # For a manipulator, actions might correspond to joint velocities

        # Example for differential drive:
        linear_vel = float(action[0][0]) * 1.0  # Scale as needed
        angular_vel = float(action[0][1]) * 0.5

        # Apply velocities to robot (implementation depends on robot type)
        # This is a simplified example - actual implementation would depend on robot control interface
        pass

    def update_networks(self, visual_obs, lidar_obs, state_obs, action, reward, value, log_prob):
        """Update policy and value networks"""
        # This would implement the actual learning algorithm (PPO, A2C, etc.)
        # For brevity, this is simplified

        # Combine state observations for value network
        combined_state = torch.cat([lidar_obs, state_obs], dim=1)

        # Compute value loss (simplified)
        predicted_value = self.value_network(visual_obs, combined_state)
        value_loss = nn.MSELoss()(predicted_value, torch.tensor([reward], device=self.device))

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute policy loss (simplified)
        # In a real implementation, this would use advantage estimation
        advantage = reward - predicted_value.detach()
        policy_loss = -(log_prob * advantage).mean()

        # Update policy network
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self, total_episodes=1000):
        """Main training loop"""
        for episode in range(total_episodes):
            episode_reward, episode_steps = self.train_step()

            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {episode_reward:.2f}, Steps: {episode_steps}")

                # Save model checkpoint periodically
                self.save_checkpoint(episode)

        # Close writer
        self.writer.close()

        print("Training completed!")

    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{episode}.pth')

        torch.save({
            'episode': episode,
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")