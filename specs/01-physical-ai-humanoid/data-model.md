# Data Model: Physical AI & Humanoid Robotics

## Entity: Physical AI Curriculum
- **Description**: The complete educational framework encompassing all modules from introduction to capstone project
- **Attributes**:
  - curriculum_id: unique identifier
  - title: "Physical AI & Humanoid Robotics"
  - description: Comprehensive curriculum covering Physical AI principles
  - modules: Array of learning modules
  - target_audience: ["Advanced AI students", "Robotics engineers", "Developers with Python knowledge"]
  - difficulty_level: "Upper-intermediate to advanced"
- **Validation rules**:
  - Must include all specified modules (0-6)
  - Must have valid target audience
  - Must have appropriate difficulty level
- **Relationships**: Contains multiple Learning Modules

## Entity: Learning Module
- **Description**: Individual educational unit covering specific concepts within the Physical AI curriculum
- **Attributes**:
  - module_id: unique identifier (0-6)
  - title: Module title (e.g., "Introduction to Physical AI", "ROS 2 Systems")
  - purpose: Description of the module's purpose
  - topics: Array of topics covered
  - learning_outcomes: Array of learning outcomes
  - prerequisites: Array of required knowledge
  - content_path: Path to the module's content files
- **Validation rules**:
  - Must have a unique module_id
  - Must have at least one topic
  - Must have at least one learning outcome
  - Content path must exist
- **Relationships**: Belongs to one Physical AI Curriculum, contains multiple Content Sections

## Entity: Content Section
- **Description**: Individual section within a learning module containing specific educational content
- **Attributes**:
  - section_id: unique identifier
  - title: Section title
  - content_type: Type of content (text, code, diagram, video, exercise)
  - content: The actual content (markdown/MDX)
  - order: Position within the module
  - module_id: Reference to parent module
- **Validation rules**:
  - Must have a valid content_type
  - Must have content
  - Order must be unique within module
- **Relationships**: Belongs to one Learning Module

## Entity: Simulation Environment
- **Description**: Virtual world with physics, gravity, collisions, and sensor modeling for testing AI-robot systems
- **Attributes**:
  - env_id: unique identifier
  - name: Environment name
  - description: Description of the environment
  - physics_engine: "Gazebo" or "Unity" or both
  - objects: Array of objects in the environment
  - sensors: Array of sensor types available
  - configuration_path: Path to environment configuration
- **Validation rules**:
  - Must have a valid physics_engine
  - Configuration path must exist
- **Relationships**: Used by multiple Learning Modules, referenced by Content Sections

## Entity: ROS 2 System
- **Description**: Robot Operating System architecture including nodes, topics, services, and actions for robotic communication
- **Attributes**:
  - system_id: unique identifier
  - name: System name
  - ros_distribution: ROS 2 distribution (e.g., "Humble Hawksbill")
  - nodes: Array of node definitions
  - topics: Array of topic definitions
  - services: Array of service definitions
  - actions: Array of action definitions
  - launch_files: Array of launch file paths
- **Validation rules**:
  - Must have a valid ROS distribution
  - Node definitions must be valid
- **Relationships**: Referenced by multiple Learning Modules, used in Content Sections

## Entity: Humanoid Robot Model
- **Description**: Kinematic structure representing a human-like robot with limbs, joints, and actuation capabilities
- **Attributes**:
  - robot_id: unique identifier
  - name: Robot name
  - description: Description of the robot
  - urdf_path: Path to URDF file
  - kinematic_chain: Definition of joint structure
  - sensors: Array of sensor types on the robot
  - actuators: Array of actuator types
  - capabilities: Array of robot capabilities
- **Validation rules**:
  - URDF path must be valid
  - Kinematic chain must be properly defined
- **Relationships**: Used in multiple Learning Modules, referenced in Content Sections

## Entity: AI Integration Pipeline
- **Description**: Processing system that connects vision, language, and action for robotic control
- **Attributes**:
  - pipeline_id: unique identifier
  - name: Pipeline name
  - components: Array of pipeline components (vision, language, action)
  - input_types: Array of input types (camera, microphone, sensors)
  - output_types: Array of output types (motor commands, speech)
  - configuration_path: Path to pipeline configuration
- **Validation rules**:
  - Must have at least one component
  - Input and output types must be valid
- **Relationships**: Referenced by multiple Learning Modules, implemented in Content Sections

## Entity: Capstone Project
- **Description**: Integrated system demonstrating all learned concepts working together in a complete autonomous humanoid robot
- **Attributes**:
  - project_id: unique identifier
  - name: "Autonomous Humanoid Robot"
  - objective: Description of the project objective
  - capabilities: Array of required capabilities
  - evaluation_criteria: Array of evaluation criteria
  - implementation_path: Path to project files
- **Validation rules**:
  - Must include all specified capabilities
  - Evaluation criteria must be measurable
- **Relationships**: Integrates multiple Learning Modules, references multiple Content Sections

## State Transitions: Learning Module Progression
- **States**: Not Started → In Progress → Completed
- **Transitions**:
  - Not Started → In Progress: When learner begins module
  - In Progress → Completed: When learner meets all learning outcomes
  - In Progress → Not Started: When learner resets progress

## Validation Rules Summary
- All entities must have unique identifiers
- All content paths must exist and be accessible
- All learning modules must map to the specified curriculum requirements
- All relationships must be valid and properly referenced
- All configuration files must be syntactically correct