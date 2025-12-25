---
sidebar_position: 5
---

# Isaac Simulation for Robot Learning

## Advanced Simulation for Robotics Development

Isaac Sim provides high-fidelity simulation capabilities that enable safe, efficient, and cost-effective development of robotic systems. This module covers advanced simulation techniques for robotics applications.

### Isaac Sim Architecture

Isaac Sim is built on NVIDIA Omniverse and provides:

- **Photorealistic rendering**: High-quality graphics for visual perception
- **Accurate physics simulation**: Realistic physical interactions
- **Large-scale environments**: Complex scenes for testing
- **Hardware acceleration**: GPU-accelerated simulation
- **Real-time performance**: Fast simulation for rapid iteration

### Isaac Sim Setup and Configuration

#### Basic Scene Setup

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

# Create a world instance
my_world = World(stage_units_in_meters=1.0)

# Get Isaac Sim assets path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("Could not find Isaac Sim assets path")
else:
    # Add a robot to the scene
    my_world.scene.add(
        Robot(
            prim_path="/World/Robot",
            name="my_robot",
            usd_path=assets_root_path + "/Isaac/Robots/Carter/carter.usd",
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
    )

    # Add objects to the scene
    my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/Object",
            name="my_object",
            position=np.array([0.5, 0.5, 0.1]),
            size=0.1,
            color=np.array([0.8, 0.1, 0.1])
        )
    )
```

#### Advanced Scene Configuration

```python
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.carb import set_carb_setting

class AdvancedRobotScene(Scene):
    def __init__(self, name: str = "advanced_robot_scene"):
        super().__init__(name=name)

        # Set advanced physics settings
        set_carb_setting("persistent.dt", 1.0/60.0)  # Physics update rate
        set_carb_setting("physics.maxSubSteps", 4)   # Substeps for stability

        # Add ground plane
        self.ground_plane = GroundPlane(
            prim_path="/World/defaultGroundPlane",
            name="default_ground_plane",
            size=15.0
        )

        # Create physics material
        self.physics_material = PhysicsMaterial(
            prim_path="/World/Looks/physics_material",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

    def scene_setup(self, world_params):
        """Setup the scene with advanced configuration"""
        super().scene_setup(world_params)

        # Add lighting
        self._setup_lighting()

        # Add camera for perception
        self._setup_cameras()

        # Add sensors
        self._setup_sensors()

    def _setup_lighting(self):
        """Setup advanced lighting for photorealistic rendering"""
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.stage import get_current_stage

        stage = get_current_stage()

        # Create dome light for environment lighting
        create_prim(
            prim_path="/World/DomeLight",
            prim_type="DomeLight",
            position=(0, 0, 0),
            attributes={
                "inputs:color": (0.2, 0.2, 0.2),
                "inputs:intensity": 300
            }
        )

        # Create directional light for shadows
        create_prim(
            prim_path="/World/DirectionalLight",
            prim_type="DistantLight",
            position=(0, 0, 10),
            rotation=(0, 0, 0, 1),
            attributes={
                "inputs:color": (0.8, 0.8, 0.8),
                "inputs:intensity": 1500
            }
        )

    def _setup_cameras(self):
        """Setup cameras for perception tasks"""
        from omni.isaac.sensor import Camera

        # Create RGB camera
        self.rgb_camera = Camera(
            prim_path="/World/Robot/base_link/chassis/camera",
            name="rgb_camera",
            position=np.array([0.1, 0.0, 0.1]),
            frequency=20,
            resolution=(640, 480)
        )

        # Create depth camera
        self.depth_camera = Camera(
            prim_path="/World/Robot/base_link/chassis/depth_camera",
            name="depth_camera",
            position=np.array([0.1, 0.0, 0.1]),
            frequency=20,
            resolution=(640, 480),
            depth_enabled=True
        )

    def _setup_sensors(self):
        """Setup various sensors for the robot"""
        from omni.isaac.range_sensor import LidarRtx
        from omni.isaac.core.sensors import ImuSensor

        # Create 3D LiDAR
        self.lidar = LidarRtx(
            prim_path="/World/Robot/base_link/chassis/lidar",
            name="Lidar",
            translation=np.array([0.1, 0.0, 0.2]),
            config="Example_Rotary",
            depth_range=(0.1, 25.0),
            frequency=10
        )

        # Create IMU
        self.imu = ImuSensor(
            prim_path="/World/Robot/base_link/chassis/imu",
            name="imu_sensor",
            translation=np.array([0.0, 0.0, 0.0])
        )
```

### Isaac Sim Physics Configuration

#### Advanced Physics Settings

```python
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdPhysics, PhysxSchema

def configure_advanced_physics():
    """Configure advanced physics settings for accurate simulation"""
    stage = get_current_stage()

    # Set global physics scene settings
    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    if scene_prim.IsValid():
        physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

        # Configure physics solver
        physx_scene_api.GetSolverTypeAttr().Set("TGS")  # Use TGS solver for stability
        physx_scene_api.GetMaxPositionIterationsAttr().Set(8)  # More iterations for stability
        physx_scene_api.GetMaxVelocityIterationsAttr().Set(1)  # Velocity iterations

        # Set gravity
        physx_scene_api.GetGravityAttr().Set(-9.81)

        # Configure broadphase
        physx_scene_api.GetBroadphaseTypeAttr().Set("MBP")  # Multi-box pruning

    # Set Carb settings for physics
    set_carb_setting("physics.maxDepenetrationVelocity", 100.0)
    set_carb_setting("physics.contactOffset", 0.001)
    set_carb_setting("physics.restOffset", 0.0)
    set_carb_setting("physics.gpu.maxParticles", 33554432)
    set_carb_setting("physics.gpu.maxDiffusePairs", 16777216)
    set_carb_setting("physics.gpu.maxContacts", 33554432)
    set_carb_setting("physics.gpu.maxJoints", 16777216)
    set_carb_setting("physics.gpu.maxBounds", 16777216)
    set_carb_setting("physics.gpu.maxIslands", 16777216)
```

#### Material Configuration

```python
def create_advanced_materials():
    """Create advanced materials for realistic simulation"""
    from omni.isaac.core.materials import OmniPBRMaterial
    from omni.isaac.core.utils.materials import set_material
    from omni.kit import usd

    # Create realistic robot material
    robot_material = OmniPBRMaterial(
        prim_path="/World/Looks/robot_material",
        name="robot_material",
        diffuse_color=(0.7, 0.7, 0.7),  # Metallic gray
        metallic=0.8,
        roughness=0.2,
        specular_level=0.5
    )

    # Create floor material with texture
    floor_material = OmniPBRMaterial(
        prim_path="/World/Looks/floor_material",
        name="floor_material",
        diffuse_texture="path/to/floor_texture.png",
        roughness=0.5,
        metallic=0.0
    )

    # Create object materials
    cube_material = OmniPBRMaterial(
        prim_path="/World/Looks/cube_material",
        name="cube_material",
        diffuse_color=(0.8, 0.1, 0.1),  # Red
        metallic=0.1,
        roughness=0.7
    )
```

### Isaac Sim Perception Simulation

#### Camera Simulation

```python
from omni.isaac.sensor import Camera
import carb
import numpy as np

class PerceptionSimulator:
    def __init__(self, robot_prim_path):
        self.robot_prim_path = robot_prim_path
        self.cameras = {}

    def add_rgb_camera(self, name, position, resolution=(640, 480)):
        """Add RGB camera to robot"""
        camera_path = f"{self.robot_prim_path}/camera/{name}"

        camera = Camera(
            prim_path=camera_path,
            name=f"{name}_camera",
            position=np.array(position),
            frequency=30,
            resolution=resolution
        )

        self.cameras[name] = camera
        return camera

    def add_depth_camera(self, name, position, resolution=(640, 480)):
        """Add depth camera to robot"""
        camera_path = f"{self.robot_prim_path}/camera/{name}_depth"

        depth_camera = Camera(
            prim_path=camera_path,
            name=f"{name}_depth_camera",
            position=np.array(position),
            frequency=30,
            resolution=resolution,
            depth_enabled=True
        )

        self.cameras[f"{name}_depth"] = depth_camera
        return depth_camera

    def add_segmentation_camera(self, name, position, resolution=(640, 480)):
        """Add semantic segmentation camera to robot"""
        camera_path = f"{self.robot_prim_path}/camera/{name}_seg"

        seg_camera = Camera(
            prim_path=camera_path,
            name=f"{name}_seg_camera",
            position=np.array(position),
            frequency=30,
            resolution=resolution
        )

        # Enable semantic segmentation
        from omni.isaac.core.utils.semantics import add_semantics
        add_semantics(seg_camera.prim, "class")

        self.cameras[f"{name}_seg"] = seg_camera
        return seg_camera

    def get_camera_data(self, camera_name):
        """Get data from specified camera"""
        if camera_name in self.cameras:
            camera = self.cameras[camera_name]

            # Get RGB data
            rgb_data = camera.get_rgb()

            # Get depth data if it's a depth camera
            if "depth" in camera_name:
                depth_data = camera.get_depth()
                return {"rgb": rgb_data, "depth": depth_data}
            else:
                return {"rgb": rgb_data}
        else:
            carb.log_warn(f"Camera {camera_name} not found")
            return None
```

#### LiDAR Simulation

```python
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core.utils.prims import get_prim_at_path
import carb

class LidarSimulator:
    def __init__(self, robot_prim_path):
        self.robot_prim_path = robot_prim_path
        self.lidars = {}

    def add_lidar(self, name, position, config="Example_Rotary",
                  depth_range=(0.1, 25.0), frequency=10):
        """Add LiDAR sensor to robot"""
        lidar_path = f"{self.robot_prim_path}/lidar/{name}"

        lidar = LidarRtx(
            prim_path=lidar_path,
            name=f"{name}_lidar",
            translation=np.array(position),
            config=config,
            depth_range=depth_range,
            frequency=frequency
        )

        self.lidars[name] = lidar
        return lidar

    def get_lidar_data(self, lidar_name):
        """Get data from specified LiDAR"""
        if lidar_name in self.lidars:
            lidar = self.lidars[lidar_name]
            return lidar.get_linear_depth_data()
        else:
            carb.log_warn(f"Lidar {lidar_name} not found")
            return None

    def configure_lidar_settings(self, lidar_name, **settings):
        """Configure advanced LiDAR settings"""
        if lidar_name in self.lidars:
            lidar = self.lidars[lidar_name]

            # Update settings
            if "rotation_frequency" in settings:
                lidar._rotation_frequency = settings["rotation_frequency"]
            if "horizontal_resolution" in settings:
                lidar._horizontal_resolution = settings["horizontal_resolution"]
            if "vertical_resolution" in settings:
                lidar._vertical_resolution = settings["vertical_resolution"]
```

### Isaac Sim Domain Randomization

#### Environment Variation

```python
import random
import numpy as np

class DomainRandomization:
    def __init__(self, world):
        self.world = world
        self.randomization_params = {
            'lighting': {'range': (0.5, 2.0), 'probability': 0.8},
            'textures': {'variations': 10, 'probability': 0.9},
            'object_positions': {'range': 0.5, 'probability': 0.7},
            'material_properties': {'range': 0.2, 'probability': 0.6},
            'physics_properties': {'range': 0.1, 'probability': 0.5}
        }

    def randomize_lighting(self):
        """Randomize lighting conditions in the scene"""
        from omni.isaac.core.utils.prims import get_prim_at_path
        from pxr import UsdLux

        # Get all lights in the scene
        lights = self.world.scene.get_lights()

        for light in lights:
            if random.random() < self.randomization_params['lighting']['probability']:
                # Randomize light intensity
                intensity_mult = random.uniform(
                    self.randomization_params['lighting']['range'][0],
                    self.randomization_params['lighting']['range'][1]
                )

                # Apply to light attributes
                light.GetIntensityAttr().Set(light.GetIntensityAttr().Get() * intensity_mult)

    def randomize_textures(self):
        """Randomize textures in the environment"""
        from omni.isaac.core.materials import OmniPBRMaterial

        # Get all materials in the scene
        materials = self.world.scene.get_materials()

        for material in materials:
            if random.random() < self.randomization_params['textures']['probability']:
                # Randomly select a texture variation
                tex_variation = random.randint(0, self.randomization_params['textures']['variations'] - 1)

                # Apply different texture based on variation
                # This would load different texture files based on the variation
                texture_path = f"path/to/texture_{tex_variation}.png"
                material.set_diffuse_texture(texture_path)

    def randomize_object_positions(self):
        """Randomize positions of objects in the scene"""
        # Get all dynamic objects in the scene
        objects = self.world.scene.get_objects()

        for obj in objects:
            if random.random() < self.randomization_params['object_positions']['probability']:
                # Randomize position within range
                pos_offset = np.random.uniform(
                    -self.randomization_params['object_positions']['range'],
                    self.randomization_params['object_positions']['range'],
                    size=3
                )

                # Get current position and add offset
                current_pos = obj.get_world_pose()[0]
                new_pos = current_pos + pos_offset

                # Set new position
                obj.set_world_pose(position=new_pos)

    def randomize_material_properties(self):
        """Randomize material properties like friction and restitution"""
        # Get all rigid bodies in the scene
        rigid_bodies = self.world.scene.get_rigid_bodies()

        for body in rigid_bodies:
            if random.random() < self.randomization_params['material_properties']['probability']:
                # Randomize friction coefficient
                friction_offset = random.uniform(
                    -self.randomization_params['material_properties']['range'],
                    self.randomization_params['material_properties']['range']
                )

                # Apply to body
                current_friction = body.get_physics_material().static_friction
                new_friction = max(0.0, current_friction + friction_offset)
                body.set_physics_material_static_friction(new_friction)

    def randomize_physics_properties(self):
        """Randomize physics properties like gravity and damping"""
        # Randomize gravity
        if random.random() < self.randomization_params['physics_properties']['probability']:
            gravity_offset = random.uniform(
                -self.randomization_params['physics_properties']['range'],
                self.randomization_params['physics_properties']['range']
            ) * 9.81

            # Apply new gravity
            self.world.scene._physics_context.set_gravity(-9.81 + gravity_offset)

    def apply_domain_randomization(self):
        """Apply all domain randomization techniques"""
        self.randomize_lighting()
        self.randomize_textures()
        self.randomize_object_positions()
        self.randomize_material_properties()
        self.randomize_physics_properties()

        # Notify that randomization is complete
        carb.log_info("Domain randomization applied to simulation environment")
```

### Isaac Sim Integration with ROS 2

#### ROS 2 Bridge for Simulation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot

class IsaacSimRosBridge(Node):
    def __init__(self):
        super().__init__('isaac_sim_ros_bridge')

        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # Initialize ROS 2 publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)

        # Initialize ROS 2 subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize Isaac Sim components
        self.setup_isaac_components()

        # Create timer for simulation loop
        self.sim_timer = self.create_timer(1.0/60.0, self.simulation_loop)  # 60 Hz

    def setup_isaac_components(self):
        """Setup Isaac Sim components"""
        # Add robot to simulation
        self.robot = self.world.scene.add(
            Robot(
                prim_path="/World/Robot",
                name="sim_robot",
                usd_path="path/to/robot.usd",
                position=np.array([0.0, 0.0, 0.1]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        )

        # Add sensors to robot
        from omni.isaac.sensor import Camera
        from omni.isaac.range_sensor import LidarRtx
        from omni.isaac.core.sensors import ImuSensor

        self.rgb_camera = Camera(
            prim_path="/World/Robot/base_link/chassis/camera",
            name="rgb_camera",
            position=np.array([0.1, 0.0, 0.1]),
            frequency=30,
            resolution=(640, 480)
        )

        self.lidar = LidarRtx(
            prim_path="/World/Robot/base_link/chassis/lidar",
            name="Lidar",
            translation=np.array([0.1, 0.0, 0.2]),
            config="Example_Rotary",
            depth_range=(0.1, 25.0),
            frequency=10
        )

        self.imu_sensor = ImuSensor(
            prim_path="/World/Robot/base_link/chassis/imu",
            name="imu_sensor",
            translation=np.array([0.0, 0.0, 0.0])
        )

    def simulation_loop(self):
        """Main simulation loop that publishes sensor data"""
        # Step the simulation
        self.world.step(render=True)

        # Publish sensor data
        self.publish_camera_data()
        self.publish_lidar_data()
        self.publish_imu_data()
        self.publish_joint_states()

    def publish_camera_data(self):
        """Publish camera data to ROS 2 topics"""
        try:
            # Get RGB data from Isaac camera
            rgb_data = self.rgb_camera.get_rgb()

            if rgb_data is not None:
                # Convert to ROS Image message
                img_msg = self.bridge.cv2_to_imgmsg(rgb_data, encoding="rgb8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.header.frame_id = "camera_rgb_optical_frame"

                # Publish RGB image
                self.rgb_pub.publish(img_msg)

            # Get depth data if available
            depth_data = self.rgb_camera.get_depth()

            if depth_data is not None:
                # Convert depth to ROS Image message
                depth_msg = self.bridge.cv2_to_imgmsg(depth_data, encoding="passthrough")
                depth_msg.header.stamp = self.get_clock().now().to_msg()
                depth_msg.header.frame_id = "camera_depth_optical_frame"

                # Publish depth image
                self.depth_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing camera data: {e}")

    def publish_lidar_data(self):
        """Publish LiDAR data to ROS 2 topic"""
        try:
            # Get LiDAR data from Isaac
            lidar_data = self.lidar.get_linear_depth_data()

            if lidar_data is not None:
                # Create LaserScan message
                scan_msg = LaserScan()
                scan_msg.header.stamp = self.get_clock().now().to_msg()
                scan_msg.header.frame_id = "lidar_link"

                # Set LiDAR parameters
                scan_msg.angle_min = -np.pi / 2  # 90 degrees
                scan_msg.angle_max = np.pi / 2   # 90 degrees
                scan_msg.angle_increment = np.pi / 180  # 1 degree per increment
                scan_msg.time_increment = 0.0
                scan_msg.scan_time = 1.0 / 10.0  # 10 Hz
                scan_msg.range_min = 0.1
                scan_msg.range_max = 25.0

                # Set ranges from LiDAR data
                scan_msg.ranges = lidar_data.flatten().tolist()

                # Publish scan data
                self.scan_pub.publish(scan_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing LiDAR data: {e}")

    def publish_imu_data(self):
        """Publish IMU data to ROS 2 topic"""
        try:
            # Get IMU data from Isaac
            imu_data = self.imu_sensor.get_measured()

            if imu_data is not None:
                # Create IMU message
                imu_msg = Imu()
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = "imu_link"

                # Set orientation (simplified - in real implementation would use proper conversion)
                imu_msg.orientation.x = 0.0
                imu_msg.orientation.y = 0.0
                imu_msg.orientation.z = 0.0
                imu_msg.orientation.w = 1.0

                # Set angular velocity
                imu_msg.angular_velocity.x = imu_data.angular_velocity[0]
                imu_msg.angular_velocity.y = imu_data.angular_velocity[1]
                imu_msg.angular_velocity.z = imu_data.angular_velocity[2]

                # Set linear acceleration
                imu_msg.linear_acceleration.x = imu_data.linear_acceleration[0]
                imu_msg.linear_acceleration.y = imu_data.linear_acceleration[1]
                imu_msg.linear_acceleration.z = imu_data.linear_acceleration[2]

                # Publish IMU data
                self.imu_pub.publish(imu_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing IMU data: {e}")

    def publish_joint_states(self):
        """Publish joint states to ROS 2 topic"""
        try:
            # Get joint positions from robot
            joint_positions = self.robot.get_joint_positions()
            joint_velocities = self.robot.get_joint_velocities()

            if joint_positions is not None and joint_velocities is not None:
                # Create JointState message
                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.header.frame_id = "base_link"

                # Set joint names (would come from robot description)
                joint_msg.name = self.robot.dof_names  # Assuming this attribute exists

                # Set positions and velocities
                joint_msg.position = joint_positions.tolist()
                joint_msg.velocity = joint_velocities.tolist()

                # Publish joint states
                self.joint_pub.publish(joint_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing joint states: {e}")

    def cmd_vel_callback(self, msg):
        """Handle velocity commands from ROS 2"""
        try:
            # Convert Twist command to robot movement
            linear_vel = msg.linear.x
            angular_vel = msg.angular.z

            # Apply velocity to robot (implementation depends on robot type)
            self.apply_robot_velocity(linear_vel, angular_vel)

        except Exception as e:
            self.get_logger().error(f"Error processing cmd_vel: {e}")

    def apply_robot_velocity(self, linear_vel, angular_vel):
        """Apply velocity to the simulated robot"""
        # This would implement the actual robot movement in simulation
        # For a differential drive robot, this would set wheel velocities
        # For other robots, it might set joint velocities or apply forces
        pass
```

### Isaac Sim Performance Optimization

#### Multi-Scene Simulation

```python
from omni.isaac.core import World
from omni.isaac.core.scenes import Scene
import asyncio
import multiprocessing as mp

class MultiSceneSimulator:
    def __init__(self, num_scenes=4):
        self.num_scenes = num_scenes
        self.scenes = []
        self.worlds = []

        # Create multiple worlds for parallel training
        for i in range(self.num_scenes):
            world = World(stage_units_in_meters=1.0)
            self.worlds.append(world)

    def setup_parallel_training_scenes(self):
        """Setup multiple parallel scenes for training"""
        for i, world in enumerate(self.worlds):
            # Set up each scene with different randomizations
            self.setup_scene(world, scene_id=i)

    def setup_scene(self, world, scene_id):
        """Setup a single scene with specific configuration"""
        # Add robot to scene
        robot_position = np.array([float(scene_id % 3), float(scene_id // 3), 0.1])

        robot = world.scene.add(
            Robot(
                prim_path=f"/World/Robot_{scene_id}",
                name=f"robot_{scene_id}",
                usd_path="path/to/robot.usd",
                position=robot_position,
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        )

        # Add random objects to scene
        for j in range(5):
            obj_pos = robot_position + np.random.uniform(-2, 2, size=3)
            obj_pos[2] = 0.1  # Keep objects above ground

            world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Object_{scene_id}_{j}",
                    name=f"object_{scene_id}_{j}",
                    position=obj_pos,
                    size=0.1,
                    color=np.random.rand(3)
                )
            )

    def run_parallel_simulation(self):
        """Run all scenes in parallel"""
        # This would run all scenes simultaneously for faster training
        async def run_all_worlds():
            tasks = []
            for world in self.worlds:
                task = asyncio.create_task(self.run_world_step(world))
                tasks.append(task)

            await asyncio.gather(*tasks)

        asyncio.run(run_all_worlds())

    async def run_world_step(self, world):
        """Step a single world"""
        world.step(render=True)
        await asyncio.sleep(0)  # Yield control to other coroutines
```

### Best Practices for Isaac Simulation

#### Simulation Quality Assurance

1. **Validation**: Regularly validate simulation against real-world data
2. **Calibration**: Calibrate sensors to match real hardware characteristics
3. **Verification**: Verify physics parameters match real-world values
4. **Documentation**: Document all simulation assumptions and limitations
5. **Testing**: Test edge cases and failure scenarios

#### Performance Considerations

- **Level of Detail**: Balance visual quality with performance requirements
- **Physics Accuracy**: Choose appropriate physics settings for task requirements
- **Scene Complexity**: Manage scene complexity for real-time performance
- **Resource Management**: Efficiently manage GPU and CPU resources
- **Caching**: Cache expensive computations where possible

Isaac Sim provides powerful simulation capabilities for robotics development, enabling safe, efficient, and cost-effective testing of complex robotic systems before deployment to real hardware.