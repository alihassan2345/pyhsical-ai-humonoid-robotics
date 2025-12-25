---
sidebar_label: "Python-based ROS 2 Development using rclpy"
---

# Python-based ROS 2 Development using rclpy

## Introduction

Python is one of the most popular languages for robotics development due to its simplicity, extensive library ecosystem, and rapid prototyping capabilities. The `rclpy` client library provides Python bindings for ROS 2, enabling developers to create ROS 2 nodes, publishers, subscribers, services, and actions using Python.

## Setting Up Python ROS 2 Development

### Prerequisites

Before starting Python-based ROS 2 development, ensure you have:

- A properly installed ROS 2 distribution (Humble Hawksbill or later recommended)
- Python 3.8 or higher
- pip package manager
- Acolutionary development environment

### Installation and Setup

ROS 2 Python packages are typically installed as part of the ROS 2 distribution. The core packages include:

- `rclpy`: The Python client library for ROS 2
- `std_msgs`: Standard message types
- `sensor_msgs`: Common sensor message types
- `geometry_msgs`: 3D geometry message types
- `nav_msgs`: Navigation-related message types

## Basic Node Structure

A minimal ROS 2 Python node follows this structure:

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Class Implementation

```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

## Publishers and Subscribers

### Creating Publishers

Publishers are created using the `create_publisher()` method:

```python
from std_msgs.msg import String

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
```

### Creating Subscribers

Subscribers are created using the `create_subscription()` method:

```python
from std_msgs.msg import String

class SubscriberNode(Node):
    def __init__(self):
        super().__init__('subscriber_node')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

## Services and Clients

### Creating Services

Services are created using the `create_service()` method:

```python
from example_interfaces.srv import AddTwoInts

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response
```

### Creating Clients

Clients are created using the `create_client()` method:

```python
from example_interfaces.srv import AddTwoInts

class ClientNode(Node):
    def __init__(self):
        super().__init__('client_node')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
```

## Actions

### Creating Action Servers

Action servers are created using the `ActionServer` class:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result
```

### Creating Action Clients

Action clients are created using the `ActionClient` class:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.sequence))
```

## Parameter Management

ROS 2 nodes can declare and use parameters:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')

        # Get parameter value
        my_param = self.get_parameter('my_parameter').value

        # Parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'my_parameter' and param.type_ == Parameter.Type.STRING:
                self.get_logger().info(f'Parameter updated: {param.value}')
        return SetParametersResult(successful=True)
```

## Lifecycle Nodes

For complex systems, ROS 2 provides lifecycle nodes:

```python
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleNodeExample(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_node')

    def on_configure(self, state: LifecycleState):
        self.get_logger().info(f'Configuring node: {self.get_name()}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        self.get_logger().info(f'Activating node: {self.get_name()}')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        self.get_logger().info(f'Deactivating node: {self.get_name()}')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        self.get_logger().info(f'Cleaning up node: {self.get_name()}')
        return TransitionCallbackReturn.SUCCESS
```

## Working with Messages and Services

### Message Types

ROS 2 provides standard message types in various packages:

- `std_msgs`: Basic data types (String, Int32, Float64, etc.)
- `geometry_msgs`: 3D geometry types (Point, Pose, Twist, etc.)
- `sensor_msgs`: Sensor data types (LaserScan, Image, JointState, etc.)
- `nav_msgs`: Navigation types (Odometry, Path, OccupancyGrid, etc.)

### Creating Custom Messages

Custom messages are defined in `.msg` files and automatically generated:

```
# CustomMessage.msg
string name
int32 id
float64[] values
geometry_msgs/Pose pose
```

## Best Practices for Python ROS 2 Development

### Code Organization

- Use proper Python packaging with `setup.py` or `pyproject.toml`
- Follow PEP 8 style guidelines
- Use type hints for better code documentation
- Implement proper error handling

### Performance Considerations

- Minimize message copying in callbacks
- Use appropriate QoS settings
- Consider threading for CPU-intensive operations
- Profile code for performance bottlenecks

### Testing

- Write unit tests for individual nodes
- Use `launch_testing` for integration tests
- Test with realistic message data
- Verify behavior under various conditions

## Integration with Physical AI Systems

Python-based ROS 2 development is particularly valuable for Physical AI systems because:

- **Rapid Prototyping**: Quick development and testing of robot behaviors
- **Algorithm Development**: Implementation of perception, planning, and control algorithms
- **Simulation Integration**: Easy integration with simulation environments
- **Machine Learning**: Natural integration with ML libraries like TensorFlow and PyTorch

## Learning Objectives

After completing this chapter, you should be able to:
- Create ROS 2 nodes using Python and rclpy
- Implement publishers, subscribers, services, and actions
- Manage parameters in ROS 2 nodes
- Use lifecycle nodes for complex systems
- Apply best practices for Python ROS 2 development

## Key Takeaways

- Python provides an accessible entry point for ROS 2 development
- rclpy provides comprehensive bindings for all ROS 2 features
- Proper code organization and testing are essential for robust systems
- Python's ecosystem enables integration with advanced algorithms
- Performance considerations are important for real-time systems