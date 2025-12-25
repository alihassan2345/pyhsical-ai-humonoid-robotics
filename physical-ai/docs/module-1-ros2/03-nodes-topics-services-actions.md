---
sidebar_label: "Nodes, Topics, Services, and Actions"
---

# Nodes, Topics, Services, and Actions

## Introduction

This chapter delves into the four fundamental communication primitives of ROS 2: nodes, topics, services, and actions. Understanding these concepts is crucial for designing effective robot systems that leverage the full power of the ROS 2 middleware.

## Nodes: The Execution Units

Nodes are the basic building blocks of ROS 2 applications. Each node represents a single process that performs a specific function within the robot system.

### Node Characteristics

- **Process Isolation**: Each node runs in its own process, providing fault isolation
- **Resource Management**: Nodes can be managed independently for resource allocation
- **Language Independence**: Nodes can be written in different programming languages
- **Communication Interface**: Nodes expose their functionality through topics, services, and actions

### Node Lifecycle

ROS 2 nodes follow a well-defined lifecycle:

```
Unconfigured → Inactive → Active → Inactive → Finalized
```

This lifecycle enables complex systems to be configured, activated, and deactivated in a controlled manner.

### Node Implementation

Nodes in ROS 2 are implemented using the rclcpp (C++) or rclpy (Python) client libraries. A minimal node implementation includes:

1. Node initialization
2. Communication interface definition
3. Main execution loop
4. Cleanup procedures

## Topics: Publish-Subscribe Communication

Topics enable asynchronous, one-to-many communication using the publish-subscribe pattern. This pattern is ideal for sensor data distribution, state broadcasting, and other scenarios where data is continuously generated.

### Topic Characteristics

- **Asynchronous**: Publishers and subscribers operate independently
- **Many-to-Many**: Multiple publishers can write to a topic; multiple subscribers can read from it
- **Data-Driven**: Communication occurs when data is available
- **Fire-and-Forget**: Publishers don't know or care about subscribers

### Quality of Service (QoS) for Topics

QoS settings allow fine-tuning of topic behavior:

- **Reliability**: Best effort or reliable delivery
- **Durability**: Volatile or persistent data storage
- **History**: Keep last N samples or keep all samples
- **Rate Limits**: Maximum publishing frequency

### Use Cases for Topics

- Sensor data distribution (LIDAR, cameras, IMU)
- Robot state broadcasting (joint positions, odometry)
- Event notifications
- Parameter updates

## Services: Request-Response Communication

Services provide synchronous, request-response communication between nodes. This pattern is ideal for operations that require a specific response or confirmation.

### Service Characteristics

- **Synchronous**: Client blocks until response is received
- **One-to-One**: One client communicates with one server at a time
- **Request-Response**: Each request generates exactly one response
- **Blocking**: Client waits for service completion

### Service Implementation

Service communication involves:
1. Service definition (interface specification)
2. Service server (request processing)
3. Service client (request generation)

### Use Cases for Services

- System configuration
- Calibration procedures
- Data queries
- Command execution with confirmation

## Actions: Goal-Oriented Communication

Actions provide asynchronous, goal-oriented communication with feedback and status updates. This pattern is ideal for long-running operations that require monitoring and cancellation.

### Action Characteristics

- **Asynchronous**: Client doesn't block during execution
- **Goal-Oriented**: Operations are initiated with specific goals
- **Feedback**: Continuous updates during execution
- **Cancelation**: Ability to interrupt long-running operations
- **Result**: Final outcome when operation completes

### Action Components

An action consists of three message types:
- **Goal**: Defines the requested operation and parameters
- **Feedback**: Continuous updates during execution
- **Result**: Final outcome of the operation

### Action States

Actions transition through the following states:
```
PENDING → ACTIVE → (SUCCEEDED/ABORTED/CANCELED)
```

### Use Cases for Actions

- Navigation to goal locations
- Object manipulation tasks
- Calibration procedures
- Data processing jobs

## Communication Pattern Comparison

| Pattern | Synchronization | Multiplicity | Use Case | Example |
|---------|----------------|--------------|----------|---------|
| Topic | Asynchronous | Many-to-Many | Continuous data | Sensor streams |
| Service | Synchronous | One-to-One | Request-response | Configuration |
| Action | Asynchronous | One-to-One | Long-running tasks | Navigation |

## Design Patterns

### Publisher-Subscriber Pattern
```python
# Publisher
publisher = node.create_publisher(StringMsg, 'topic_name', 10)

# Subscriber
subscriber = node.create_subscription(StringMsg, 'topic_name', callback, 10)
```

### Client-Service Pattern
```python
# Service Server
service = node.create_service(RequestType, 'service_name', callback)

# Service Client
client = node.create_client(RequestType, 'service_name')
```

### Action Pattern
```python
# Action Server
action_server = ActionServer(node, ActionType, 'action_name', execute_callback)

# Action Client
action_client = ActionClient(node, ActionType, 'action_name')
```

## Best Practices

### Topic Design
- Use descriptive names following ROS naming conventions
- Consider QoS settings based on application requirements
- Implement appropriate message serialization
- Consider bandwidth and frequency constraints

### Service Design
- Keep service calls short and deterministic
- Use appropriate error handling
- Consider service availability and redundancy
- Document service interfaces clearly

### Action Design
- Use actions for operations that take significant time
- Provide meaningful feedback messages
- Implement proper cancellation handling
- Consider timeout mechanisms

## Integration in Physical AI Systems

These communication primitives are essential for Physical AI systems:

- **Topics**: Sensor data distribution, state broadcasting
- **Services**: Configuration, calibration, emergency stops
- **Actions**: Navigation, manipulation, complex behaviors

## Learning Objectives

After completing this chapter, you should be able to:
- Implement nodes with different communication interfaces
- Choose appropriate communication patterns for different scenarios
- Configure Quality of Service settings for topics
- Design effective service and action interfaces
- Apply best practices for ROS 2 communication

## Key Takeaways

- Each communication pattern serves specific use cases in robot systems
- Quality of Service settings enable fine-tuning of communication behavior
- Actions provide the most sophisticated communication pattern for long-running tasks
- Proper selection of communication patterns is crucial for system performance
- All patterns support the distributed nature of robot systems