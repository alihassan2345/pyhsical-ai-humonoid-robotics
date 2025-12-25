---
sidebar_label: "LLM-Based Action Planning and Task Decomposition"
---

# LLM-Based Action Planning and Task Decomposition

## Introduction

Large Language Models (LLMs) serve as the cognitive engine of the Autonomous Humanoid system, transforming high-level user commands into detailed action sequences. This chapter explores the integration of LLMs with robotics, focusing on task decomposition, action planning, and the mapping of natural language instructions to executable robot behaviors.

## LLM Integration Architecture

### System Overview

The LLM-based action planning system operates as a bridge between natural language understanding and robot execution:

```
Structured Command → LLM Prompt → Task Decomposition → Action Sequence → Robot Execution
```

The system must handle the translation from abstract, high-level commands to concrete, executable robot actions while maintaining safety, efficiency, and correctness.

### LLM Selection Criteria

When selecting an LLM for robotics applications, consider:

- **Reasoning Capabilities**: Ability to decompose complex tasks
- **Knowledge Depth**: Understanding of physical world concepts
- **Instruction Following**: Ability to follow structured instructions
- **Safety Constraints**: Built-in safety and ethical considerations
- **Latency Requirements**: Response time for real-time applications
- **Cost Efficiency**: Operational cost for continuous use

## Task Decomposition Framework

### Hierarchical Task Structure

LLMs decompose complex tasks into hierarchical structures:

```
High-Level Goal (e.g., "Clean the table")
├── Subtask 1: Navigate to table location
├── Subtask 2: Identify objects on table
├── Subtask 3: Determine which objects to move
├── Subtask 4: Plan grasp for each object
├── Subtask 5: Execute grasps and place objects
└── Subtask 6: Verify task completion
```

### Task Decomposition Algorithm

```python
import openai
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    INTERACTION = "interaction"
    INFORMATION = "information"

@dataclass
class RobotAction:
    action_type: TaskType
    parameters: Dict[str, Any]
    priority: int
    prerequisites: List[str]  # Other actions that must complete first
    estimated_duration: float  # In seconds

@dataclass
class TaskPlan:
    goal: str
    actions: List[RobotAction]
    dependencies: Dict[str, List[str]]  # action_id -> [prerequisite_action_ids]

class LLMTaskDecomposer:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.robot_capabilities = self._get_robot_capabilities()

    def decompose_task(self, user_command: str, context: Dict[str, Any]) -> TaskPlan:
        """Decompose a user command into executable robot actions"""
        prompt = self._create_decomposition_prompt(user_command, context)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            functions=[
                {
                    "name": "create_task_plan",
                    "description": "Create a detailed task plan for the robot",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal": {"type": "string", "description": "The original user goal"},
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action_type": {"type": "string", "enum": ["navigation", "manipulation", "perception", "interaction", "information"]},
                                        "parameters": {"type": "object"},
                                        "priority": {"type": "integer"},
                                        "prerequisites": {"type": "array", "items": {"type": "string"}},
                                        "estimated_duration": {"type": "number"}
                                    },
                                    "required": ["action_type", "parameters", "priority", "estimated_duration"]
                                }
                            },
                            "dependencies": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        },
                        "required": ["goal", "actions", "dependencies"]
                    }
                }
            ],
            function_call={"name": "create_task_plan"}
        )

        # Parse the response
        function_call = response.choices[0].message.function_call
        plan_data = json.loads(function_call.arguments)

        return self._convert_to_task_plan(plan_data)

    def _get_system_prompt(self) -> str:
        """System prompt for the LLM"""
        return f"""
        You are an expert robotic task planner. Your role is to decompose high-level human commands into detailed, executable robot actions.

        Robot Capabilities:
        {json.dumps(self.robot_capabilities, indent=2)}

        Guidelines:
        1. Break down complex tasks into simple, executable actions
        2. Consider the robot's physical limitations and capabilities
        3. Ensure each action is achievable with the robot's hardware
        4. Account for dependencies between actions
        5. Estimate realistic time requirements
        6. Prioritize safety in all action sequences
        7. Include perception actions to verify world state when necessary

        Action Types:
        - navigation: Moving the robot to specific locations
        - manipulation: Grasping, moving, or manipulating objects
        - perception: Sensing, detecting, or identifying objects/environment
        - interaction: Communicating with humans or other systems
        - information: Processing or providing information

        Always respond using the create_task_plan function with properly structured data.
        """

    def _create_decomposition_prompt(self, command: str, context: Dict[str, Any]) -> str:
        """Create the prompt for task decomposition"""
        return f"""
        User Command: "{command}"

        Current Context:
        {json.dumps(context, indent=2)}

        Please decompose this command into a sequence of executable robot actions.
        """

    def _get_robot_capabilities(self) -> Dict[str, Any]:
        """Define robot capabilities for the LLM"""
        return {
            "navigation": {
                "max_speed": "0.5 m/s",
                "turn_speed": "0.5 rad/s",
                "reachable_area": "indoor environment",
                "obstacle_avoidance": True
            },
            "manipulation": {
                "arm_dof": 7,
                "gripper_type": "parallel_jaw",
                "max_payload": "2 kg",
                "reachable_workspace": "hemisphere in front of robot",
                "precision": "centimeter level"
            },
            "perception": {
                "camera": {
                    "resolution": "1920x1080",
                    "fov": "60 degrees",
                    "range": "0.1m to 10m"
                },
                "lidar": {
                    "range": "0.1m to 20m",
                    "fov": "360 degrees horizontal, 30 degrees vertical"
                },
                "microphone": {
                    "voice_recognition": True,
                    "directional": True
                }
            },
            "communication": {
                "speech_synthesis": True,
                "text_display": True
            }
        }

    def _convert_to_task_plan(self, plan_data: Dict) -> TaskPlan:
        """Convert LLM response to TaskPlan object"""
        actions = []
        for action_data in plan_data['actions']:
            action = RobotAction(
                action_type=TaskType(action_data['action_type']),
                parameters=action_data['parameters'],
                priority=action_data['priority'],
                prerequisites=action_data['prerequisites'],
                estimated_duration=action_data['estimated_duration']
            )
            actions.append(action)

        return TaskPlan(
            goal=plan_data['goal'],
            actions=actions,
            dependencies=plan_data['dependencies']
        )
```

## Action Planning and Execution

### Sequential vs Parallel Execution

The LLM must consider which actions can be executed in parallel:

```python
class ActionScheduler:
    def __init__(self):
        self.completed_actions = set()
        self.currently_executing = set()

    def get_ready_actions(self, plan: TaskPlan) -> List[RobotAction]:
        """Get actions that are ready to execute (prerequisites satisfied)"""
        ready_actions = []

        for action in plan.actions:
            # Check if all prerequisites are completed
            prereq_satisfied = all(
                prereq in self.completed_actions
                for prereq in action.prerequisites
            )

            # Check if action is not already executing
            not_executing = action not in self.currently_executing

            if prereq_satisfied and not_executing:
                ready_actions.append(action)

        # Sort by priority (higher priority first)
        ready_actions.sort(key=lambda x: x.priority, reverse=True)

        return ready_actions

    def update_action_status(self, action_id: str, status: str):
        """Update the status of an action"""
        if status == "completed":
            self.completed_actions.add(action_id)
            if action_id in self.currently_executing:
                self.currently_executing.remove(action_id)
        elif status == "started":
            self.currently_executing.add(action_id)
```

### Safety and Validation Layer

LLM-generated plans must be validated for safety:

```python
class PlanValidator:
    def __init__(self):
        self.safety_constraints = {
            "max_navigation_speed": 0.5,  # m/s
            "max_manipulation_payload": 2.0,  # kg
            "safe_distance_to_humans": 0.5,  # meters
            "reachable_workspace": self._define_workspace()
        }

    def validate_plan(self, plan: TaskPlan, current_state: Dict) -> tuple[bool, List[str]]:
        """Validate the plan for safety and feasibility"""
        issues = []

        for action in plan.actions:
            action_issues = self._validate_action(action, current_state)
            issues.extend(action_issues)

        is_valid = len(issues) == 0
        return is_valid, issues

    def _validate_action(self, action: RobotAction, current_state: Dict) -> List[str]:
        """Validate individual action"""
        issues = []

        if action.action_type == TaskType.NAVIGATION:
            # Validate navigation target
            target = action.parameters.get('target')
            if not self._is_safe_navigation_target(target, current_state):
                issues.append(f"Navigation to {target} is unsafe")

        elif action.action_type == TaskType.MANIPULATION:
            # Validate manipulation parameters
            object_weight = action.parameters.get('object_weight', 0)
            if object_weight > self.safety_constraints['max_manipulation_payload']:
                issues.append(f"Object weighs {object_weight}kg, exceeds payload limit")

        elif action.action_type == TaskType.PERCEPTION:
            # Validate perception request
            target = action.parameters.get('target')
            if not self._is_perceivable(target, current_state):
                issues.append(f"Target {target} is not perceivable from current position")

        return issues

    def _is_safe_navigation_target(self, target: Dict, current_state: Dict) -> bool:
        """Check if navigation target is safe"""
        # Implement safety checks
        return True  # Placeholder

    def _is_perceivable(self, target: Dict, current_state: Dict) -> bool:
        """Check if target is perceivable"""
        # Implement perception checks
        return True  # Placeholder

    def _define_workspace(self) -> Dict:
        """Define robot's reachable workspace"""
        # Define workspace boundaries
        return {
            "x_range": (-1.0, 1.0),
            "y_range": (-1.0, 1.0),
            "z_range": (0.0, 1.5)
        }
```

## LLM Prompt Engineering for Robotics

### Context-Rich Prompts

Effective LLM prompting for robotics requires detailed context:

```python
class RobotLLMPrompter:
    def __init__(self):
        self.system_context = self._build_system_context()

    def _build_system_context(self) -> str:
        """Build comprehensive system context for LLM"""
        return f"""
        You are controlling an autonomous humanoid robot with the following specifications:

        PHYSICAL CAPABILITIES:
        - Height: 1.5 meters
        - Weight: 50 kg
        - Mobility: Bipedal walking with balance control
        - Manipulation: 2 arms with 7 DOF each, parallel jaw grippers
        - Sensors: RGB-D camera, LiDAR, IMU, force/torque sensors

        ENVIRONMENTAL CONSTRAINTS:
        - Indoor environments only
        - Human-safe operation required (maintain 0.5m distance)
        - Navigation speed limited to 0.5 m/s
        - Maximum payload: 2 kg per arm

        SAFETY PROTOCOLS:
        - Always verify object properties before manipulation
        - Stop if humans come within 0.5m during navigation
        - Request clarification for ambiguous commands
        - Fail safely if uncertain about action safety

        COMMUNICATION:
        - Provide status updates to user
        - Ask for clarification when needed
        - Report task progress and completion

        ACTION VOCABULARY:
        Navigation: move_to(location), approach_object(object_name)
        Manipulation: grasp_object(object_name), place_object(location), release_object()
        Perception: detect_objects(), locate_object(object_name), verify_grasp()
        Interaction: speak(text), gesture(action)

        Remember: Safety first, accuracy second, efficiency third.
        """

    def create_task_planning_prompt(self, user_goal: str, current_state: Dict) -> str:
        """Create a detailed prompt for task planning"""
        return f"""
        {self.system_context}

        CURRENT ROBOT STATE:
        {json.dumps(current_state, indent=2)}

        USER GOAL: {user_goal}

        Please decompose this goal into a sequence of specific, executable actions.
        Each action should be concrete and achievable by the robot.
        Consider the current state and environmental constraints.

        For each action, specify:
        1. Action type (navigation, manipulation, perception, interaction)
        2. Parameters needed for execution
        3. Priority level (1-10, with 10 being highest priority)
        4. Prerequisites (what must be completed first)
        5. Estimated time for completion

        Return your plan in the structured format using the create_task_plan function.
        """

    def create_replanning_prompt(self, original_goal: str, failure_context: Dict) -> str:
        """Create prompt for replanning after failure"""
        return f"""
        {self.system_context}

        ORIGINAL GOAL: {original_goal}

        FAILURE CONTEXT:
        {json.dumps(failure_context, indent=2)}

        The robot encountered an issue while executing the original plan.
        Please create a new plan that addresses the failure while still achieving the goal.
        Consider alternative approaches that might be more robust.

        Return your new plan in the structured format.
        """
```

## Integration with ROS 2 Action System

### Action Server Implementation

The LLM-generated plans integrate with ROS 2's action system:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from ..interfaces.action import ExecuteTaskPlan  # Custom action definition

class LLMActionPlanner(Node):
    def __init__(self):
        super().__init__('llm_action_planner')

        # Initialize LLM components
        self.task_decomposer = LLMTaskDecomposer(api_key=self.get_parameter('openai_api_key').value)
        self.action_scheduler = ActionScheduler()
        self.plan_validator = PlanValidator()

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecuteTaskPlan,
            'execute_task_plan',
            self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Subscribe to robot state
        self.state_sub = self.create_subscription(
            RobotState,
            '/robot/state',
            self.state_callback,
            10
        )

        self.current_robot_state = {}
        self.active_plan = None

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        self.get_logger().info(f'Received task planning goal: {goal_request.user_command}')

        # Validate the goal
        if self._is_valid_goal(goal_request):
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the task planning goal"""
        feedback_msg = ExecuteTaskPlan.Feedback()
        result = ExecuteTaskPlan.Result()

        try:
            # Decompose the user command into a task plan
            context = self._build_context(goal_handle.request.user_command)
            plan = self.task_decomposer.decompose_task(
                goal_handle.request.user_command,
                context
            )

            # Validate the plan
            is_valid, issues = self.plan_validator.validate_plan(plan, self.current_robot_state)
            if not is_valid:
                result.success = False
                result.error_message = f"Plan validation failed: {', '.join(issues)}"
                goal_handle.abort()
                return result

            self.active_plan = plan
            feedback_msg.current_task = "Plan generated and validated"
            goal_handle.publish_feedback(feedback_msg)

            # Execute the plan
            execution_result = self._execute_plan(plan, goal_handle, feedback_msg)

            result.success = execution_result['success']
            result.error_message = execution_result.get('error', '')
            result.completed_tasks = execution_result.get('completed_tasks', [])

            if execution_result['success']:
                goal_handle.succeed()
            else:
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f'Error in task planning: {e}')
            result.success = False
            result.error_message = str(e)
            goal_handle.abort()

        return result

    def _execute_plan(self, plan: TaskPlan, goal_handle, feedback_msg):
        """Execute the task plan step by step"""
        completed_tasks = []
        failed_tasks = []

        for i, action in enumerate(plan.actions):
            if goal_handle.is_cancel_requested:
                return {'success': False, 'error': 'Goal canceled'}

            try:
                # Execute the action
                action_result = self._execute_single_action(action)

                if action_result['success']:
                    completed_tasks.append(action.action_type.value)
                    feedback_msg.current_task = f"Completed: {action.action_type.value}"
                    feedback_msg.progress = (i + 1) / len(plan.actions) * 100
                    goal_handle.publish_feedback(feedback_msg)
                else:
                    failed_tasks.append({
                        'action': action.action_type.value,
                        'error': action_result.get('error', 'Unknown error')
                    })
                    # Try to recover or continue
                    if not self._can_continue_after_failure(action, action_result):
                        break

            except Exception as e:
                failed_tasks.append({
                    'action': action.action_type.value,
                    'error': str(e)
                })
                self.get_logger().error(f'Error executing action {action.action_type}: {e}')
                break

        success = len(failed_tasks) == 0
        return {
            'success': success,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks
        }

    def _execute_single_action(self, action: RobotAction):
        """Execute a single robot action"""
        # This would call specific ROS 2 services or actions
        # based on the action type
        import time
        time.sleep(action.estimated_duration)  # Placeholder

        return {'success': True}

    def _build_context(self, user_command: str) -> Dict:
        """Build context for task decomposition"""
        return {
            'user_command': user_command,
            'robot_state': self.current_robot_state,
            'environment_map': self._get_environment_map(),
            'object_database': self._get_known_objects(),
            'current_time': self.get_clock().now().to_msg()
        }

    def state_callback(self, msg):
        """Update robot state"""
        self.current_robot_state = self._convert_state_message(msg)

    def _is_valid_goal(self, goal_request) -> bool:
        """Validate if the goal is acceptable"""
        # Check if we have required parameters
        api_key = self.get_parameter('openai_api_key').value
        return api_key is not None and api_key != ''
```

## Handling Uncertainty and Replanning

### Dynamic Replanning

The system must handle unexpected situations and replan accordingly:

```python
class DynamicReplanner:
    def __init__(self, llm_client: LLMTaskDecomposer):
        self.llm_client = llm_client
        self.failure_history = []

    def handle_execution_failure(self, failed_action: RobotAction, failure_context: Dict) -> Dict:
        """Handle action failure and generate recovery plan"""
        # Log the failure
        self.failure_history.append({
            'action': failed_action,
            'context': failure_context,
            'timestamp': self.get_current_time()
        })

        # Generate recovery plan
        recovery_prompt = self._create_recovery_prompt(failed_action, failure_context)

        try:
            recovery_plan = self.llm_client.decompose_task(
                recovery_prompt['user_command'],
                recovery_prompt['context']
            )

            return {
                'success': True,
                'recovery_plan': recovery_plan,
                'reasoning': recovery_prompt['reasoning']
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Recovery planning failed: {str(e)}",
                'fallback_action': self._get_safe_fallback(failed_action)
            }

    def _create_recovery_prompt(self, failed_action: RobotAction, context: Dict) -> Dict:
        """Create prompt for recovery planning"""
        return {
            'user_command': f"Recover from failure in {failed_action.action_type.value} action",
            'context': {
                'failed_action': failed_action.__dict__,
                'failure_context': context,
                'original_goal': context.get('original_goal', 'Unknown'),
                'available_alternatives': self._get_available_alternatives(failed_action, context)
            },
            'reasoning': f"Original {failed_action.action_type.value} action failed due to {context.get('failure_reason', 'unknown')}. Need alternative approach."
        }

    def _get_available_alternatives(self, failed_action: RobotAction, context: Dict) -> List[str]:
        """Get available alternative approaches"""
        alternatives = []

        if failed_action.action_type == TaskType.MANIPULATION:
            alternatives.extend([
                "Try different grasp approach",
                "Request human assistance",
                "Use alternative manipulation strategy",
                "Navigate to different approach angle"
            ])
        elif failed_action.action_type == TaskType.NAVIGATION:
            alternatives.extend([
                "Find alternative path",
                "Wait for obstacle to clear",
                "Request human guidance",
                "Use different navigation mode"
            ])

        return alternatives

    def _get_safe_fallback(self, failed_action: RobotAction) -> RobotAction:
        """Get a safe fallback action"""
        return RobotAction(
            action_type=TaskType.INTERACTION,
            parameters={'message': f'Unable to complete {failed_action.action_type.value} task. Requesting human assistance.'},
            priority=10,
            prerequisites=[],
            estimated_duration=2.0
        )
```

## Performance Optimization

### Caching and Pattern Recognition

To improve efficiency, the system can cache common task patterns:

```python
import hashlib
from functools import lru_cache

class OptimizedLLMPlanner:
    def __init__(self, llm_client: LLMTaskDecomposer):
        self.llm_client = llm_client
        self.plan_cache = {}
        self.pattern_recognizer = PatternRecognizer()

    @lru_cache(maxsize=100)
    def get_cached_plan(self, command_hash: str) -> TaskPlan:
        """Get cached plan if available"""
        # This is a simplified version - in practice, you'd have a more
        # sophisticated caching mechanism that considers context
        pass

    def get_task_plan(self, user_command: str, context: Dict) -> TaskPlan:
        """Get task plan with caching and optimization"""
        # Create a hash of the command and relevant context
        cache_key = self._create_cache_key(user_command, context)

        # Check cache first
        if cache_key in self.plan_cache:
            cached_plan = self.plan_cache[cache_key]
            if self._is_plan_still_valid(cached_plan, context):
                return cached_plan

        # Check for similar patterns
        similar_pattern = self.pattern_recognizer.find_similar_pattern(user_command)
        if similar_pattern:
            adapted_plan = self._adapt_plan_for_context(similar_pattern, context)
            if adapted_plan:
                self.plan_cache[cache_key] = adapted_plan
                return adapted_plan

        # Generate new plan
        new_plan = self.llm_client.decompose_task(user_command, context)
        self.plan_cache[cache_key] = new_plan

        return new_plan

    def _create_cache_key(self, command: str, context: Dict) -> str:
        """Create cache key for command and context"""
        cache_data = {
            'command': command,
            'location': context.get('location', 'unknown'),
            'objects': sorted(context.get('visible_objects', []))
        }
        return hashlib.md5(str(cache_data).encode()).hexdigest()

    def _is_plan_still_valid(self, plan: TaskPlan, context: Dict) -> bool:
        """Check if cached plan is still valid for current context"""
        # Implement validation logic
        return True  # Placeholder
```

## Learning Objectives

After completing this chapter, you should be able to:
- Implement LLM-based task decomposition for robotics applications
- Design safe and validated action planning systems
- Integrate LLMs with ROS 2 action servers
- Handle uncertainty and dynamic replanning
- Optimize LLM usage for real-time robotics applications

## Key Takeaways

- LLMs provide powerful reasoning capabilities for robotics
- Safety validation is essential for LLM-generated plans
- Context-aware prompting improves plan quality
- Dynamic replanning handles real-world uncertainties
- Caching and optimization improve system performance