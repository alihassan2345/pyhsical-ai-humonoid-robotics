---
sidebar_label: "URDF for Humanoid Robot Modeling"
---

# URDF for Humanoid Robot Modeling

## Introduction

Unified Robot Description Format (URDF) is the standard format for representing robot models in ROS. For humanoid robotics, URDF provides the foundation for modeling complex multi-link mechanisms with appropriate kinematic chains, visual representations, and physical properties. This chapter explores the application of URDF to humanoid robot modeling.

## URDF Fundamentals

### What is URDF?

URDF (Unified Robot Description Format) is an XML-based format for representing robot models. It describes:

- **Kinematic Structure**: Joint and link relationships
- **Visual Properties**: How the robot appears in simulation
- **Collision Properties**: How the robot interacts with its environment
- **Physical Properties**: Mass, inertia, and other physical characteristics

### URDF Structure

A basic URDF model consists of:

- **Links**: Rigid bodies that make up the robot
- **Joints**: Connections between links with specific degrees of freedom
- **Materials**: Visual appearance properties
- **Gazebo-specific extensions**: Additional properties for simulation

## Link Definition

Links represent rigid bodies in the robot model:

```xml
<link name="link_name">
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
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

### Link Components

1. **Inertial**: Physical properties for dynamics simulation
2. **Visual**: Appearance in visualization tools
3. **Collision**: Shape for collision detection

## Joint Definition

Joints define the connection between links:

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Joint Types

- **revolute**: Rotational joint with 1 DOF
- **prismatic**: Linear joint with 1 DOF
- **continuous**: Continuous rotational joint (no limits)
- **fixed**: No movement between links
- **floating**: 6 DOF movement
- **planar**: Planar movement with 3 DOF

## Humanoid Robot Kinematic Structure

### Anthropomorphic Design Principles

Humanoid robots follow human-like kinematic structures:

- **Trunk**: Torso with multiple degrees of freedom
- **Arms**: Shoulder, elbow, and wrist joints
- **Legs**: Hip, knee, and ankle joints
- **Head**: Neck joint for orientation

### Typical Humanoid Configuration

A basic humanoid configuration includes:

```
base_link
├── torso
│   ├── head
│   ├── left_arm
│   │   ├── left_forearm
│   │   └── left_hand
│   ├── right_arm
│   │   ├── right_forearm
│   │   └── right_hand
│   ├── left_leg
│   │   ├── left_lower_leg
│   │   └── left_foot
│   └── right_leg
│       ├── right_lower_leg
│       └── right_foot
```

## Humanoid-Specific Considerations

### Balance and Stability

Humanoid robots require careful attention to:

- **Center of Mass**: Positioning for stable locomotion
- **Zero Moment Point (ZMP)**: Dynamic stability criteria
- **Support Polygon**: Area of stable stance

### Degrees of Freedom

Humanoid robots typically have 20+ degrees of freedom:

- **Legs**: 6 DOF each (hip: 3, knee: 1, ankle: 2)
- **Arms**: 7 DOF each (shoulder: 3, elbow: 1, wrist: 2, hand: 1)
- **Trunk**: 2-3 DOF for upper body movement
- **Head**: 2-3 DOF for gaze control

### Anthropometric Proportions

Humanoid models should follow human-like proportions:

- **Leg Length**: ~45% of total height
- **Arm Length**: ~40% of total height
- **Trunk Length**: ~30% of total height
- **Head Size**: ~8% of total height

## Advanced URDF Features

### Transmission Elements

Transmissions define the relationship between actuators and joints:

```xml
<transmission name="transmission_joint1">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint1">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor1">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Safety Controllers

Joint limits and safety constraints:

```xml
<joint name="shoulder_joint" type="revolute">
  <limit lower="-2.0" upper="2.0" effort="100" velocity="2"/>
  <safety_controller k_position="20" k_velocity="400"
                    soft_lower_limit="-1.9" soft_upper_limit="1.9"/>
</joint>
```

### Gazebo-Specific Extensions

Gazebo simulation requires additional tags:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>
```

## URDF Best Practices for Humanoid Robots

### Model Organization

- **Hierarchical Structure**: Organize links and joints logically
- **Naming Conventions**: Use consistent, descriptive names
- **Modular Design**: Break complex models into subassemblies

### Performance Optimization

- **Collision Simplification**: Use simplified shapes for collision
- **Visual Detail**: High detail for visualization, low for collision
- **Link Reduction**: Minimize unnecessary links for performance

### Accuracy Considerations

- **Mass Properties**: Accurate inertial parameters for dynamics
- **Joint Limits**: Realistic limits based on hardware capabilities
- **Physical Dimensions**: Accurate measurements for simulation fidelity

## Example: Simple Humanoid Model

Here's a minimal humanoid model example:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Additional limbs would follow similar patterns -->
</robot>
```

## Tools for URDF Development

### Model Validation

- **check_urdf**: Validate URDF syntax and structure
- **urdf_to_graphiz**: Generate kinematic tree diagrams
- **rviz**: Visualize robot model in real-time

### Model Creation

- **SolidWorks to URDF**: Export from CAD software
- **Blender**: Create and export robot models
- **Manual Editing**: Direct XML editing for precision

## Integration with Physical AI Systems

URDF models integrate with Physical AI systems through:

- **Simulation**: Accurate physics simulation in Gazebo
- **Planning**: Motion planning algorithms use kinematic models
- **Control**: Robot controllers use kinematic information
- **Perception**: Vision systems can recognize and track robot parts

## Troubleshooting Common Issues

### Kinematic Errors

- **Invalid Joint Limits**: Ensure limits are physically achievable
- **Disconnected Links**: Verify all links are connected through joints
- **Singularities**: Avoid configurations that cause mathematical singularities

### Simulation Issues

- **Unstable Dynamics**: Check mass and inertia parameters
- **Collision Penetration**: Verify collision geometry
- **Actuator Limits**: Ensure joint limits match hardware capabilities

## Learning Objectives

After completing this chapter, you should be able to:
- Create URDF models for humanoid robots
- Define appropriate kinematic structures for anthropomorphic designs
- Apply best practices for humanoid robot modeling
- Integrate URDF models with simulation and control systems
- Troubleshoot common URDF issues

## Key Takeaways

- URDF provides the standard format for robot modeling in ROS
- Humanoid robots require careful attention to anthropomorphic design principles
- Proper mass and inertia properties are crucial for accurate simulation
- URDF models enable integration across simulation, planning, and control
- Validation and testing are essential for robust robot models