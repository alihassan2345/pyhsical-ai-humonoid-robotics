---
sidebar_label: "Launch Files and Parameter Management"
---

# Launch Files and Parameter Management

## Introduction

Launch files and parameter management are critical components of ROS 2 system configuration. They enable the deployment of complex robot systems with multiple nodes, each configured with appropriate parameters, in a reproducible and maintainable manner.

## Launch Files Overview

Launch files allow you to start multiple ROS 2 nodes simultaneously with specific configurations. They provide a declarative way to define robot system compositions and manage their lifecycle.

### Launch File Benefits

- **System Composition**: Start multiple nodes with a single command
- **Configuration Management**: Set parameters and configurations for each node
- **Lifecycle Control**: Manage node startup order and dependencies
- **Environment Setup**: Configure environment variables and remappings
- **Reproducibility**: Ensure consistent system deployment across environments

### Launch File Structure

Launch files can be written in Python or XML (though Python is recommended for its flexibility):

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='my_node',
            name='my_node_name',
            parameters=[
                {'param1': 'value1'},
                '/path/to/params.yaml'
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        )
    ])
```

## Launch File Components

### Node Actions

The `Node` action is the primary component for launching ROS 2 nodes:

```python
from launch_ros.actions import Node

my_node = Node(
    package='package_name',
    executable='executable_name',  # or 'name' for node name
    name='node_name',              # override default node name
    namespace='namespace',         # node namespace
    parameters=[
        {'param1': 'value1'},
        {'param2': 42},
        '/path/to/params.yaml'
    ],
    remappings=[
        ('original_topic', 'remapped_topic'),
        ('original_service', 'remapped_service')
    ],
    arguments=['arg1', 'arg2'],    # command line arguments
    condition=IfCondition(LaunchConfiguration('use_node'))  # conditional launch
)
```

### Launch Arguments

Launch arguments allow runtime customization of launch files:

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    # Use launch arguments in node configuration
    node = Node(
        package='my_package',
        executable='my_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        node
    ])
```

### Conditions and Control Flow

Launch files support conditional execution and control flow:

```python
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration

# Conditional node launch
conditional_node = Node(
    package='my_package',
    executable='my_node',
    condition=IfCondition(LaunchConfiguration('enable_feature'))
)

# Multiple conditions
complex_condition = Node(
    package='my_package',
    executable='my_node',
    condition=IfCondition(
        PythonExpression([
            LaunchConfiguration('condition1'),
            ' and ',
            LaunchConfiguration('condition2')
        ])
    )
)
```

## Parameter Management

### Parameter Sources

ROS 2 supports multiple parameter sources:

1. **Command Line**: Parameters specified at launch time
2. **YAML Files**: Structured parameter definitions
3. **Code Defaults**: Parameters defined in the node
4. **Dynamic Reconfiguration**: Runtime parameter updates

### YAML Parameter Files

YAML files provide a structured way to define parameters:

```yaml
/**:  # Applies to all nodes
  ros__parameters:
    use_sim_time: false
    log_level: "info"

my_node:  # Applies to specific node
  ros__parameters:
    param1: "value1"
    param2: 42
    param3:
      nested_param1: "nested_value"
      nested_param2: [1, 2, 3]
```

### Parameter Declaration and Usage

Within nodes, parameters are declared and used as follows:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('param1', 'default_value')
        self.declare_parameter('param2', 42)
        self.declare_parameter('param3', [1.0, 2.0, 3.0])

        # Get parameter values
        param1_value = self.get_parameter('param1').value
        param2_value = self.get_parameter('param2').value

        # Get parameter with default fallback
        param3_value = self.get_parameter_or('param3', [0.0, 0.0, 0.0])

    def update_parameter_callback(self, parameters):
        """Callback for parameter updates"""
        for param in parameters:
            if param.name == 'param1':
                self.get_logger().info(f'Parameter param1 updated to: {param.value}')
        return SetParametersResult(successful=True)

    def setup_parameter_callback(self):
        """Setup parameter callback"""
        self.add_on_set_parameters_callback(self.update_parameter_callback)
```

## Advanced Launch Features

### Composable Nodes

Launch files support composable nodes for improved performance:

```python
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    container = ComposableNodeContainer(
        name='my_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='image_proc',
                plugin='image_proc::RectifyNode',
                name='rectify_node',
                parameters=[{'param1': 'value1'}]
            ),
            ComposableNode(
                package='image_view',
                plugin='image_view::ImageViewNode',
                name='image_view_node'
            )
        ],
        output='screen',
    )

    return LaunchDescription([container])
```

### Timer Actions

Launch files can include timer-based actions:

```python
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    delayed_node = TimerAction(
        period=5.0,  # Wait 5 seconds
        actions=[
            Node(
                package='my_package',
                executable='delayed_node',
                name='delayed_node'
            )
        ]
    )

    return LaunchDescription([delayed_node])
```

## Launch System Integration

### Process Management

Launch files provide process management capabilities:

```python
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart

# Register event handlers
event_handler = RegisterEventHandler(
    OnProcessExit(
        target_action=node_action,
        on_exit=[some_other_action]
    )
)
```

### Signal Handling

Launch files handle signals gracefully:

```python
from launch.actions import Shutdown

# Conditional shutdown
shutdown_action = RegisterEventHandler(
    OnProcessExit(
        target_action=critical_node,
        on_exit=Shutdown(reason='Critical node exited')
    )
)
```

## Best Practices

### Launch File Organization

- **Modularity**: Break complex systems into smaller, reusable launch files
- **Inheritance**: Use launch file inclusion to build upon base configurations
- **Documentation**: Comment launch files to explain system composition
- **Validation**: Test launch files in different environments

### Parameter Organization

- **Hierarchical Structure**: Use namespaces to organize related parameters
- **Validation**: Validate parameter values at startup
- **Documentation**: Document parameter meanings and valid ranges
- **Defaults**: Provide sensible default values

### Performance Considerations

- **Startup Order**: Consider dependencies when ordering node startup
- **Resource Allocation**: Be mindful of resource requirements for each node
- **Monitoring**: Include monitoring nodes for system health

## Integration with Physical AI Systems

Launch files and parameter management are essential for Physical AI systems:

- **System Deployment**: Deploy complex robot systems with consistent configurations
- **Environment Adaptation**: Adjust parameters for different operating environments
- **Hardware Abstraction**: Parameterize hardware-specific configurations
- **Safety Configuration**: Configure safety parameters and limits

## Common Patterns

### Simulation vs. Real Robot

```python
# Launch file with conditional configuration
def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Simulation-specific nodes
    sim_nodes = GroupAction(
        condition=IfCondition(use_sim_time),
        actions=[simulator_node]
    )

    # Real robot nodes
    real_nodes = GroupAction(
        condition=UnlessCondition(use_sim_time),
        actions=[hardware_interface_nodes]
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        sim_nodes,
        real_nodes
    ])
```

### Multi-Robot Systems

Launch files support multi-robot deployments:

```python
# Launch file for multiple robots
def generate_launch_description():
    robots = ['robot1', 'robot2', 'robot3']
    launch_actions = []

    for robot_name in robots:
        robot_launch = GroupAction(
            actions=[
                # Robot-specific nodes with namespace
                Node(
                    package='navigation',
                    executable='nav2',
                    namespace=robot_name,
                    parameters=[f'{robot_name}_params.yaml']
                )
            ]
        )
        launch_actions.append(robot_launch)

    return LaunchDescription(launch_actions)
```

## Learning Objectives

After completing this chapter, you should be able to:
- Create and use launch files for complex system deployment
- Manage parameters using YAML files and launch arguments
- Implement advanced launch features like composable nodes
- Apply best practices for system configuration
- Design launch files for Physical AI systems

## Key Takeaways

- Launch files enable reproducible system deployment
- Parameter management supports environment-specific configurations
- Advanced launch features improve system performance and reliability
- Proper organization enables maintainable robot systems
- Launch files are essential for Physical AI system deployment