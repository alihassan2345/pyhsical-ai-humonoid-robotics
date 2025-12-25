---
sidebar_label: "URDF vs SDF Models"
---

# URDF vs SDF Models

## Introduction

Understanding the relationship between URDF (Unified Robot Description Format) and SDF (Simulation Description Format) is crucial for effective robot simulation in Gazebo. While URDF is primarily used for robot description in ROS, SDF is Gazebo's native format for simulation. This chapter explores the differences, similarities, and best practices for working with both formats in the context of Physical AI and humanoid robotics.

## URDF: The ROS Standard

### Overview

URDF (Unified Robot Description Format) is the standard format for representing robot models in ROS. It focuses on:

- **Kinematic Structure**: Joint and link relationships
- **Visual Properties**: How the robot appears
- **Collision Properties**: Collision detection geometry
- **Physical Properties**: Mass and inertia parameters

### URDF Structure

A typical URDF model consists of:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Joints define connections between links -->
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>
</robot>
```

### URDF Strengths

- **ROS Integration**: Native support in ROS toolchain
- **Kinematic Focus**: Excellent for motion planning and control
- **Simplicity**: Straightforward format for basic robot models
- **Community**: Extensive library of existing URDF models

## SDF: The Gazebo Native Format

### Overview

SDF (Simulation Description Format) is Gazebo's native format, designed specifically for simulation. It provides:

- **Simulation Features**: Physics, sensors, plugins, and controllers
- **Environment Modeling**: Complete world description
- **Advanced Features**: Complex materials, lighting, and effects

### SDF Structure

A typical SDF model includes:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="my_model">
    <pose>0 0 0 0 0 0</pose>
    <link name="link1">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.1 0.1 0.1</size>
          </box>
        </geometry>
      </collision>
      <sensor name="camera" type="camera">
        <camera>
          <horizontal_fov>1.047</horizontal_fov>
        </camera>
      </sensor>
    </link>
    <joint name="joint1" type="revolute">
      <parent>link1</parent>
      <child>link2</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1</velocity>
        </limit>
      </axis>
    </joint>
  </model>
</sdf>
```

### SDF Strengths

- **Simulation Focus**: Rich simulation-specific features
- **Flexibility**: Support for complex environments and scenarios
- **Extensibility**: Easy to add custom simulation elements
- **World Description**: Complete environment modeling

## Key Differences

### Structural Differences

| Aspect | URDF | SDF |
|--------|------|-----|
| **Purpose** | Robot description | Simulation environment |
| **Scope** | Individual robots | Complete worlds |
| **Syntax** | More concise | More verbose but flexible |
| **Extensibility** | Through XACRO macros | Through native XML structure |

### Feature Comparison

**URDF Features:**
- Joint and link definitions
- Visual and collision geometry
- Inertial properties
- Basic material definitions
- Transmission elements

**SDF Features:**
- All URDF features plus:
- Sensors and plugins
- Physics engine configuration
- Environment lighting
- Complex world elements
- Animation and effects

## Conversion Between Formats

### URDF to SDF

Gazebo automatically converts URDF to SDF using the `libgazebo_ros_factory.so` plugin:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or use the ROS 2 tool
ros2 run xacro xacro robot.urdf.xacro --inorder > robot.urdf
gz sdf -p robot.urdf > robot.sdf
```

### Gazebo ROS Integration

The conversion process adds Gazebo-specific elements to URDF:

```xml
<!-- In URDF, add Gazebo-specific elements -->
<gazebo reference="my_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <parameters>$(find my_robot_description)/config/control.yaml</parameters>
  </plugin>
</gazebo>
```

These elements are processed during the URDF-to-SDF conversion to add simulation-specific features.

## Best Practices for Physical AI

### When to Use URDF

Use URDF when:

- **Robot Description**: Modeling the robot's kinematic structure
- **ROS Integration**: Need tight integration with ROS toolchain
- **Motion Planning**: Working with MoveIt! or similar planners
- **Control Development**: Developing ROS-based controllers
- **Simplicity**: Basic robot model without complex simulation needs

### When to Use SDF

Use SDF when:

- **Complete Simulation**: Need to model entire environments
- **Advanced Sensors**: Complex sensor configurations
- **Custom Plugins**: Simulation-specific functionality
- **World Modeling**: Complex environments with multiple objects
- **Performance**: Need to optimize simulation-specific parameters

## Humanoid Robot Considerations

### Complex Kinematic Chains

Humanoid robots require careful consideration of format choice:

**URDF Advantages:**
- Clean kinematic chain definition
- Easy integration with kinematic solvers
- Standard tools for inverse kinematics

**SDF Advantages:**
- Better support for complex contact modeling
- Advanced sensor integration
- Custom balance control plugins

### Example: Humanoid Model Structure

For a humanoid robot, you might use both formats:

```xml
<!-- robot.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <!-- Define the robot kinematics using URDF -->
  <xacro:include filename="$(find humanoid_description)/urdf/humanoid.urdf.xacro"/>

  <!-- Add Gazebo-specific elements -->
  <gazebo>
    <plugin name="humanoid_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <gazebo reference="head_camera_link">
    <sensor type="camera" name="head_camera">
      <!-- Camera configuration -->
    </sensor>
  </gazebo>
</robot>
```

## Practical Examples

### Simple Robot in URDF

```xml
<!-- simple_robot.urdf.xacro -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>
</robot>
```

### World with Robot in SDF

```xml
<!-- world_with_robot.sdf -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include the robot model -->
    <include>
      <uri>model://simple_robot</uri>
      <pose>0 0 0.1 0 0 0</pose>
    </include>

    <!-- Add obstacles -->
    <model name="obstacle">
      <pose>1 1 0 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 1.0</size>
            </box>
          </geometry>
        </collision>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>
  </world>
</sdf>
```

## Tools and Utilities

### URDF Tools

- **check_urdf**: Validate URDF syntax and structure
- **urdf_to_graphiz**: Generate kinematic tree diagrams
- **xacro**: Process URDF macros and variables
- **RViz**: Visualize URDF models

### SDF Tools

- **gz sdf**: Validate and convert SDF files
- **Gazebo GUI**: Visualize and edit SDF models
- **SDF Schema**: Validate against SDF specifications

## Performance Considerations

### URDF Performance

- **Conversion Overhead**: URDF to SDF conversion time
- **Memory Usage**: URDF models are typically smaller
- **Parsing Speed**: Faster parsing for simple models

### SDF Performance

- **Native Processing**: No conversion needed
- **Simulation Efficiency**: Optimized for simulation
- **Complexity Handling**: Better for complex scenarios

## Migration Strategies

### From URDF to SDF

When you need to migrate or extend a URDF model:

1. **Start with URDF**: Use URDF for robot kinematics
2. **Add Gazebo Elements**: Include Gazebo-specific elements in URDF
3. **Convert When Needed**: Generate SDF for complex scenarios
4. **Maintain Both**: Keep both formats synchronized

### Hybrid Approach

Many successful projects use both formats:

- **URDF for Robot Description**: Maintain robot kinematics in URDF
- **SDF for Simulation**: Use SDF for complete simulation scenarios
- **XACRO for Complex Models**: Use XACRO macros for parameterized models

## Learning Objectives

After completing this chapter, you should be able to:
- Understand the differences between URDF and SDF formats
- Choose the appropriate format for different use cases
- Convert between formats when necessary
- Apply best practices for humanoid robot modeling
- Integrate both formats in simulation workflows

## Key Takeaways

- URDF and SDF serve different but complementary purposes
- URDF is ideal for robot kinematics and ROS integration
- SDF is ideal for complete simulation environments
- Both formats can be used together in effective workflows
- Understanding both formats is essential for advanced robotics simulation