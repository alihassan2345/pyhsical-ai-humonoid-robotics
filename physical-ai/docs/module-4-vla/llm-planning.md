---
sidebar_position: 3
---

# LLM-Based Cognitive Planning

## Using Large Language Models for Robot Task Planning

Large Language Models (LLMs) can serve as powerful cognitive planners for robotics, enabling robots to understand complex natural language commands and generate appropriate action sequences. This module covers the implementation of LLM-based planning for humanoid robots.

### LLM Planning Architecture

The LLM planning architecture follows this pattern:

```
Natural Language Command → LLM Intent Parser → Task Decomposition → Action Sequencing → Execution Planning → Robot Control
```

### LLM-Based Task Planning Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import openai
import json
import asyncio
import threading
import queue
from typing import Dict, List, Any, Optional
import time

class LLMPlanningNode(Node):
    def __init__(self):
        super().__init__('llm_planning_node')

        # Initialize OpenAI client
        # self.openai_client = OpenAI(api_key=self.get_parameter_or('openai_api_key', 'YOUR_API_KEY'))

        # Create subscribers
        self.voice_cmd_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        self.text_cmd_sub = self.create_subscription(
            String,
            '/text_command',
            self.text_command_callback,
            10
        )

        # Create publishers
        self.plan_pub = self.create_publisher(
            String, '/task_plan', 10
        )
        self.action_pub = self.create_publisher(
            String, '/robot_action', 10
        )
        self.status_pub = self.create_publisher(
            String, '/planning_status', 10
        )

        # Initialize planning components
        self.command_queue = queue.Queue(maxsize=10)
        self.planning_thread = threading.Thread(target=self.planning_loop)
        self.planning_thread.daemon = True
        self.planning_thread.start()

        # Maintain context for multi-turn planning
        self.planning_context = PlanningContext()

        # Configuration
        self.declare_parameter('model_name', 'gpt-3.5-turbo')
        self.declare_parameter('temperature', 0.1)
        self.declare_parameter('max_tokens', 1000)
        self.declare_parameter('timeout', 30.0)

        self.get_logger().info('LLM planning node initialized')

    def voice_command_callback(self, msg):
        """Handle voice commands"""
        try:
            command_data = json.loads(msg.data) if msg.data.startswith('{') else {"command": msg.data}
            self.command_queue.put_nowait({
                'source': 'voice',
                'data': command_data,
                'timestamp': time.time()
            })
        except queue.Full:
            self.get_logger().warn('Command queue full, dropping voice command')

    def text_command_callback(self, msg):
        """Handle text commands"""
        try:
            command_data = json.loads(msg.data) if msg.data.startswith('{') else {"command": msg.data}
            self.command_queue.put_nowait({
                'source': 'text',
                'data': command_data,
                'timestamp': time.time()
            })
        except queue.Full:
            self.get_logger().warn('Command queue full, dropping text command')

    def planning_loop(self):
        """Main planning loop running in background thread"""
        while rclpy.ok():
            try:
                # Get command from queue
                command_item = self.command_queue.get(timeout=1.0)

                # Process the command
                result = self.process_command(command_item)

                # Publish results
                if result:
                    status_msg = String()
                    status_msg.data = json.dumps(result)
                    self.status_pub.publish(status_msg)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Planning error: {e}')

    def process_command(self, command_item):
        """Process a command through LLM-based planning"""
        try:
            source = command_item['source']
            data = command_item['data']
            timestamp = command_item['timestamp']

            command_text = data.get('command', data.get('text', ''))

            self.get_logger().info(f'Processing {source} command: {command_text}')

            # Plan the task using LLM
            plan = self.generate_task_plan(command_text)

            if plan:
                # Publish the plan
                plan_msg = String()
                plan_msg.data = json.dumps(plan)
                self.plan_pub.publish(plan_msg)

                # Generate immediate actions if needed
                immediate_actions = self.extract_immediate_actions(plan)
                for action in immediate_actions:
                    action_msg = String()
                    action_msg.data = json.dumps(action)
                    self.action_pub.publish(action_msg)

                # Update planning context
                self.planning_context.add_completed_plan(plan)

                return {
                    'success': True,
                    'message': 'Plan generated successfully',
                    'plan': plan,
                    'command': command_text,
                    'timestamp': timestamp
                }
            else:
                return {
                    'success': False,
                    'message': 'Could not generate plan',
                    'command': command_text,
                    'timestamp': timestamp
                }

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            return {
                'success': False,
                'message': f'Processing error: {str(e)}',
                'command': command_item['data'].get('command', ''),
                'timestamp': command_item['timestamp']
            }

    def generate_task_plan(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Generate task plan using LLM"""
        try:
            # Create a structured prompt for task planning
            prompt = self.create_planning_prompt(command_text)

            # In a real implementation, call the LLM
            # For simulation, return a mock plan
            return self.mock_generate_task_plan(command_text)

        except Exception as e:
            self.get_logger().error(f'LLM planning error: {e}')
            return None

    def create_planning_prompt(self, command_text: str) -> str:
        """Create structured prompt for LLM task planning"""
        context = self.planning_context.get_context_summary()

        prompt = f"""
        You are a cognitive planner for a humanoid robot. Generate a detailed task plan to execute the following command.

        Current Context:
        {context}

        Command: "{command_text}"

        Generate a detailed task plan in JSON format with the following structure:
        {{
            "command": "Original command text",
            "intent": "High-level intent of the command",
            "task_sequence": [
                {{
                    "step": 1,
                    "action": "action_type",
                    "target": "target_object_or_location",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "description": "Brief description of the step",
                    "prerequisites": ["list of conditions that must be met"],
                    "expected_outcome": "Expected result of this step"
                }}
            ],
            "estimated_duration": "Estimated time in seconds",
            "potential_obstacles": ["list of potential obstacles"],
            "safety_considerations": ["list of safety considerations"]
        }}

        Action types include: navigate, detect, manipulate, wait, communicate, etc.
        Make the plan detailed enough for a robot to execute but high-level enough to be flexible.
        Consider the robot's capabilities and limitations.
        """

        return prompt

    def mock_generate_task_plan(self, command_text: str) -> Dict[str, Any]:
        """Mock task plan generation for demonstration"""
        import re

        command_lower = command_text.lower()

        # Parse command to identify key elements
        plan = {
            "command": command_text,
            "intent": self.identify_intent(command_text),
            "task_sequence": [],
            "estimated_duration": 120,
            "potential_obstacles": [],
            "safety_considerations": ["Maintain safe distance from obstacles", "Avoid sudden movements"]
        }

        # Navigation commands
        if any(word in command_lower for word in ['go to', 'navigate to', 'move to', 'walk to', 'drive to']):
            # Extract destination
            match = re.search(r'(?:go to|navigate to|move to|walk to|drive to)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                destination = match.group(1).strip()
                plan["task_sequence"].extend([
                    {
                        "step": 1,
                        "action": "navigate",
                        "target": destination,
                        "parameters": {"speed": "normal", "avoid_obstacles": True},
                        "description": f"Navigate to {destination}",
                        "prerequisites": ["robot is powered on", "navigation system is ready"],
                        "expected_outcome": f"Robot reaches {destination}"
                    }
                ])

        # Manipulation commands
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take', 'get', 'grab', 'lift']):
            match = re.search(r'(?:pick up|grasp|take|get|grab|lift)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                object_name = match.group(1).strip()
                plan["task_sequence"].extend([
                    {
                        "step": 1,
                        "action": "detect",
                        "target": object_name,
                        "parameters": {"search_radius": 2.0},
                        "description": f"Locate {object_name}",
                        "prerequisites": ["robot has visual sensors operational"],
                        "expected_outcome": f"{object_name} is detected and localized"
                    },
                    {
                        "step": 2,
                        "action": "navigate",
                        "target": f"near_{object_name}",
                        "parameters": {"approach_distance": 0.5, "speed": "slow"},
                        "description": f"Approach {object_name}",
                        "prerequisites": [f"{object_name} is detected"],
                        "expected_outcome": "Robot is positioned near object for manipulation"
                    },
                    {
                        "step": 3,
                        "action": "manipulate",
                        "target": object_name,
                        "parameters": {"operation": "grasp", "precision": "high"},
                        "description": f"Grasp {object_name}",
                        "prerequisites": ["robot is positioned correctly", "object is reachable"],
                        "expected_outcome": f"{object_name} is grasped successfully"
                    }
                ])

        # Complex commands involving multiple steps
        elif 'bring' in command_lower or 'fetch' in command_lower:
            # Example: "Bring me the red cup from the kitchen"
            plan["task_sequence"].extend([
                {
                    "step": 1,
                    "action": "navigate",
                    "target": "kitchen",
                    "parameters": {"speed": "normal", "avoid_obstacles": True},
                    "description": "Navigate to kitchen",
                    "prerequisites": ["robot knows kitchen location"],
                    "expected_outcome": "Robot arrives in kitchen"
                },
                {
                    "step": 2,
                    "action": "detect",
                    "target": "red cup",
                    "parameters": {"search_radius": 2.0},
                    "description": "Look for red cup",
                    "prerequisites": ["robot is in kitchen"],
                    "expected_outcome": "Red cup is detected"
                },
                {
                    "step": 3,
                    "action": "manipulate",
                    "target": "red cup",
                    "parameters": {"operation": "grasp", "precision": "high"},
                    "description": "Grasp the red cup",
                    "prerequisites": ["red cup is detected and reachable"],
                    "expected_outcome": "Red cup is grasped"
                },
                {
                    "step": 4,
                    "action": "navigate",
                    "target": "user_location",
                    "parameters": {"speed": "careful", "avoid_obstacles": True},
                    "description": "Return to user with red cup",
                    "prerequisites": ["red cup is grasped"],
                    "expected_outcome": "Robot delivers red cup to user"
                }
            ])

        # Default: simple command
        else:
            plan["task_sequence"].append({
                "step": 1,
                "action": "unknown",
                "target": command_text,
                "parameters": {},
                "description": f"Process unknown command: {command_text}",
                "prerequisites": [],
                "expected_outcome": "Command processed"
            })

        return plan

    def identify_intent(self, command_text: str) -> str:
        """Identify high-level intent from command text"""
        command_lower = command_text.lower()

        if any(word in command_lower for word in ['go to', 'navigate', 'move to', 'walk to', 'drive to']):
            return "navigation"
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take', 'get', 'grab', 'lift']):
            return "manipulation"
        elif any(word in command_lower for word in ['find', 'look for', 'locate', 'search for']):
            return "detection"
        elif any(word in command_lower for word in ['bring', 'fetch', 'carry']):
            return "transport"
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return "stop"
        else:
            return "unknown"

    def extract_immediate_actions(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract immediate actions from the plan"""
        immediate_actions = []

        # Add first few steps as immediate actions
        for step in plan.get('task_sequence', [])[:3]:  # First 3 steps
            immediate_actions.append({
                'action': step['action'],
                'target': step['target'],
                'parameters': step['parameters'],
                'step_number': step['step'],
                'plan_id': hash(plan['command'])  # Simple plan identifier
            })

        return immediate_actions


class PlanningContext:
    """Maintain context for LLM planning"""

    def __init__(self):
        self.completed_plans = []
        self.failed_attempts = []
        self.robot_capabilities = self.get_robot_capabilities()
        self.environment_state = self.get_environment_state()

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get robot capabilities for planning context"""
        return {
            "locomotion": ["walking", "navigation", "obstacle avoidance"],
            "manipulation": ["grasping", "lifting", "placing"],
            "sensing": ["vision", "lidar", "touch", "audio"],
            "communication": ["speech", "gesture", "display"],
            "max_payload": 5.0,  # kg
            "max_reach": 1.5,    # meters
            "battery_life": 120, # minutes
            "speeds": {
                "slow": 0.3,    # m/s
                "normal": 0.6,
                "fast": 1.0
            }
        }

    def get_environment_state(self) -> Dict[str, Any]:
        """Get current environment state for planning context"""
        # In a real implementation, this would come from perception system
        return {
            "known_locations": ["kitchen", "living room", "bedroom", "office"],
            "obstacles": [],
            "lighting": "normal",
            "noise_level": "low"
        }

    def add_completed_plan(self, plan: Dict[str, Any]):
        """Add completed plan to context"""
        self.completed_plans.append({
            'plan': plan,
            'timestamp': time.time(),
            'success': True  # Simplified - in real implementation, track actual success
        })

        # Keep only recent plans to limit context size
        if len(self.completed_plans) > 10:
            self.completed_plans = self.completed_plans[-10:]

    def add_failed_attempt(self, command: str, error: str):
        """Add failed planning attempt to context"""
        self.failed_attempts.append({
            'command': command,
            'error': error,
            'timestamp': time.time()
        })

        # Keep only recent failures
        if len(self.failed_attempts) > 5:
            self.failed_attempts = self.failed_attempts[-5:]

    def get_context_summary(self) -> str:
        """Get summary of current context for LLM"""
        context_parts = []

        # Recent successful plans
        if self.completed_plans:
            recent_plans = self.completed_plans[-3:]  # Last 3 plans
            context_parts.append("Recently completed tasks:")
            for plan in recent_plans:
                context_parts.append(f"- {plan['plan']['command']}")

        # Recent failures
        if self.failed_attempts:
            context_parts.append("\nRecent planning failures to avoid:")
            for failure in self.failed_attempts[-2:]:  # Last 2 failures
                context_parts.append(f"- {failure['command']}: {failure['error']}")

        # Robot capabilities
        context_parts.append(f"\nRobot capabilities: {json.dumps(self.robot_capabilities)}")

        # Environment state
        context_parts.append(f"Current environment: {json.dumps(self.environment_state)}")

        return "\n".join(context_parts)


class AdvancedLLMPlanningNode(LLMPlanningNode):
    """Advanced LLM planning with multi-step reasoning and error recovery"""

    def __init__(self):
        super().__init__()

        # Initialize advanced planning components
        self.reasoning_engine = MultiStepReasoningEngine()
        self.error_recovery = ErrorRecoverySystem()
        self.plan_validator = PlanValidator()

        # Add more sophisticated publishers
        self.detailed_plan_pub = self.create_publisher(
            String, '/detailed_task_plan', 10
        )

    def generate_task_plan(self, command_text: str) -> Optional[Dict[str, Any]]:
        """Generate task plan with advanced reasoning"""
        try:
            # Use multi-step reasoning for complex commands
            if self.is_complex_command(command_text):
                plan = self.reasoning_engine.generate_plan_with_reasoning(command_text)
            else:
                plan = self.mock_generate_task_plan(command_text)

            # Validate the plan
            validation_result = self.plan_validator.validate_plan(plan)

            if not validation_result['valid']:
                # Try to repair the plan
                plan = self.plan_validator.repair_plan(plan, validation_result['issues'])

            # Add execution metadata
            plan['metadata'] = {
                'generation_method': 'llm_multi_step_reasoning' if self.is_complex_command(command_text) else 'llm_basic',
                'timestamp': time.time(),
                'confidence': validation_result.get('confidence', 0.8),
                'risk_assessment': self.assess_plan_risk(plan)
            }

            return plan

        except Exception as e:
            self.get_logger().error(f'Advanced planning error: {e}')
            return None

    def is_complex_command(self, command_text: str) -> bool:
        """Determine if command requires complex reasoning"""
        complex_indicators = [
            'and', 'then', 'after', 'before', 'while', 'until',
            'if', 'when', 'condition', 'require', 'need'
        ]

        command_lower = command_text.lower()
        return any(indicator in command_lower for indicator in complex_indicators)

    def assess_plan_risk(self, plan: Dict[str, Any]) -> Dict[str, float]:
        """Assess risk level of plan execution"""
        risk_factors = {
            'navigation_risk': 0.0,
            'manipulation_risk': 0.0,
            'time_constraint_risk': 0.0,
            'resource_constraint_risk': 0.0
        }

        for step in plan.get('task_sequence', []):
            action = step.get('action', '')
            if action == 'navigate':
                risk_factors['navigation_risk'] += 0.2
            elif action == 'manipulate':
                risk_factors['manipulation_risk'] += 0.3
            elif action == 'wait':
                risk_factors['time_constraint_risk'] += 0.1

        return risk_factors


class MultiStepReasoningEngine:
    """Engine for multi-step reasoning in task planning"""

    def __init__(self):
        self.subtask_decomposer = SubtaskDecomposer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.resource_allocator = ResourceAllocator()

    def generate_plan_with_reasoning(self, command_text: str) -> Dict[str, Any]:
        """Generate plan using multi-step reasoning"""
        # Step 1: Decompose complex command into subtasks
        subtasks = self.subtask_decomposer.decompose(command_text)

        # Step 2: Analyze dependencies between subtasks
        dependencies = self.dependency_analyzer.analyze_dependencies(subtasks)

        # Step 3: Allocate resources for each subtask
        resource_allocations = self.resource_allocator.allocate_resources(subtasks)

        # Step 4: Create execution plan with reasoning trace
        plan = self.create_reasoned_plan(command_text, subtasks, dependencies, resource_allocations)

        return plan

    def create_reasoned_plan(self, command_text: str, subtasks: List[Dict],
                           dependencies: Dict, resource_allocations: Dict) -> Dict[str, Any]:
        """Create plan with reasoning trace"""
        reasoned_plan = {
            "command": command_text,
            "intent": "complex_task_with_reasoning",
            "reasoning_trace": [],
            "task_sequence": [],
            "dependencies": dependencies,
            "resource_allocations": resource_allocations,
            "estimated_duration": self.estimate_duration(subtasks),
            "confidence": 0.85
        }

        # Add reasoning steps
        for i, subtask in enumerate(subtasks):
            reasoning_step = {
                "step_number": i + 1,
                "subtask": subtask,
                "reasoning": self.generate_reasoning_for_subtask(subtask),
                "resources_needed": resource_allocations.get(subtask['id'], {}),
                "estimated_time": self.estimate_subtask_time(subtask)
            }
            reasoned_plan["reasoning_trace"].append(reasoning_step)

            # Add to task sequence
            task_step = {
                "step": i + 1,
                "action": subtask['action'],
                "target": subtask['target'],
                "parameters": subtask.get('parameters', {}),
                "description": subtask['description'],
                "prerequisites": dependencies.get(subtask['id'], []),
                "expected_outcome": subtask['expected_outcome']
            }
            reasoned_plan["task_sequence"].append(task_step)

        return reasoned_plan

    def generate_reasoning_for_subtask(self, subtask: Dict) -> str:
        """Generate reasoning for a subtask"""
        return f"Perform {subtask['action']} on {subtask['target']} because it's needed for the overall task."

    def estimate_duration(self, subtasks: List[Dict]) -> float:
        """Estimate total duration for all subtasks"""
        return sum(self.estimate_subtask_time(st) for st in subtasks)

    def estimate_subtask_time(self, subtask: Dict) -> float:
        """Estimate time for a single subtask"""
        base_times = {
            'navigate': 30.0,
            'detect': 20.0,
            'manipulate': 25.0,
            'communicate': 10.0,
            'wait': 5.0
        }
        return base_times.get(subtask['action'], 15.0)


class SubtaskDecomposer:
    """Decompose complex tasks into simpler subtasks"""

    def __init__(self):
        self.decomposition_rules = self.load_decomposition_rules()

    def load_decomposition_rules(self) -> Dict:
        """Load rules for task decomposition"""
        return {
            'transport_object': [
                {'action': 'navigate', 'target': 'source_location', 'description': 'Go to object location'},
                {'action': 'detect', 'target': 'object', 'description': 'Find the object'},
                {'action': 'manipulate', 'target': 'object', 'description': 'Grasp the object'},
                {'action': 'navigate', 'target': 'destination', 'description': 'Go to destination'},
                {'action': 'manipulate', 'target': 'object', 'description': 'Place the object'}
            ],
            'prepare_workspace': [
                {'action': 'detect', 'target': 'workspace', 'description': 'Identify workspace'},
                {'action': 'navigate', 'target': 'workspace', 'description': 'Move to workspace'},
                {'action': 'detect', 'target': 'obstacles', 'description': 'Identify obstacles'},
                {'action': 'manipulate', 'target': 'obstacles', 'description': 'Clear obstacles'},
                {'action': 'communicate', 'target': 'status', 'description': 'Confirm workspace is ready'}
            ]
        }

    def decompose(self, command_text: str) -> List[Dict]:
        """Decompose command into subtasks"""
        command_lower = command_text.lower()

        # Identify task type
        if 'bring' in command_lower or 'fetch' in command_lower or 'carry' in command_lower:
            task_type = 'transport_object'
        elif 'prepare' in command_lower or 'clean' in command_lower:
            task_type = 'prepare_workspace'
        else:
            # For unknown types, use simple decomposition
            return self.simple_decomposition(command_text)

        # Apply decomposition rules
        if task_type in self.decomposition_rules:
            subtasks = []
            for i, rule_step in enumerate(self.decomposition_rules[task_type]):
                subtask = rule_step.copy()
                subtask['id'] = f'subtask_{i+1}'
                subtasks.append(subtask)
            return subtasks
        else:
            return self.simple_decomposition(command_text)

    def simple_decomposition(self, command_text: str) -> List[Dict]:
        """Simple decomposition for unknown command types"""
        return [{
            'id': 'subtask_1',
            'action': 'unknown',
            'target': command_text,
            'description': f'Process: {command_text}',
            'expected_outcome': 'Command processed'
        }]


class DependencyAnalyzer:
    """Analyze dependencies between subtasks"""

    def analyze_dependencies(self, subtasks: List[Dict]) -> Dict:
        """Analyze dependencies between subtasks"""
        dependencies = {}

        for i, subtask in enumerate(subtasks):
            task_id = subtask['id']
            deps = []

            # Previous task dependency (for sequential tasks)
            if i > 0:
                deps.append(subtasks[i-1]['id'])

            # Resource dependencies
            resource_deps = self.get_resource_dependencies(subtask)
            deps.extend(resource_deps)

            dependencies[task_id] = deps

        return dependencies

    def get_resource_dependencies(self, subtask: Dict) -> List[str]:
        """Get resource-based dependencies"""
        # In a real implementation, this would check resource availability
        # For simulation, return empty list
        return []


class ResourceAllocator:
    """Allocate resources for subtasks"""

    def __init__(self):
        self.resources = {
            'navigation_system': True,
            'manipulator_arm': True,
            'camera_system': True,
            'battery_level': 0.8,  # 80% charge
            'memory_available': True
        }

    def allocate_resources(self, subtasks: List[Dict]) -> Dict:
        """Allocate resources for each subtask"""
        allocations = {}

        for subtask in subtasks:
            task_id = subtask['id']
            required_resources = self.get_required_resources(subtask['action'])

            allocations[task_id] = {
                'required': required_resources,
                'available': self.check_resource_availability(required_resources),
                'priority': self.get_task_priority(subtask)
            }

        return allocations

    def get_required_resources(self, action: str) -> List[str]:
        """Get resources required for an action"""
        resource_requirements = {
            'navigate': ['navigation_system', 'battery_level'],
            'detect': ['camera_system', 'battery_level'],
            'manipulate': ['manipulator_arm', 'camera_system'],
            'communicate': ['camera_system'],
            'wait': ['battery_level']
        }
        return resource_requirements.get(action, [])

    def check_resource_availability(self, required_resources: List[str]) -> Dict[str, bool]:
        """Check availability of required resources"""
        availability = {}
        for resource in required_resources:
            availability[resource] = self.resources.get(resource, False)
        return availability

    def get_task_priority(self, subtask: Dict) -> int:
        """Get priority for a subtask"""
        # Higher priority for safety-critical tasks
        if 'safety' in subtask.get('description', '').lower():
            return 3  # High priority
        elif 'navigation' in subtask['action']:
            return 2  # Medium priority
        else:
            return 1  # Low priority


class PlanValidator:
    """Validate and repair task plans"""

    def __init__(self):
        self.validation_rules = self.load_validation_rules()

    def load_validation_rules(self) -> Dict:
        """Load validation rules"""
        return {
            'action_validity': self.validate_action_validity,
            'resource_availability': self.validate_resource_availability,
            'safety_compliance': self.validate_safety_compliance,
            'logical_consistency': self.validate_logical_consistency
        }

    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a task plan"""
        results = {
            'valid': True,
            'issues': [],
            'confidence': 0.9,
            'warnings': []
        }

        for rule_name, validator in self.validation_rules.items():
            rule_result = validator(plan)
            if not rule_result['valid']:
                results['valid'] = False
                results['issues'].extend(rule_result['issues'])
            elif rule_result.get('warnings'):
                results['warnings'].extend(rule_result['warnings'])

        return results

    def validate_action_validity(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all actions are valid"""
        valid_actions = ['navigate', 'detect', 'manipulate', 'communicate', 'wait', 'stop']

        issues = []
        warnings = []

        for step in plan.get('task_sequence', []):
            action = step.get('action', 'unknown')
            if action not in valid_actions:
                issues.append(f"Invalid action '{action}' in step {step.get('step', 'unknown')}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def validate_resource_availability(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate resource availability for plan execution"""
        issues = []
        warnings = []

        # Check if plan requires more time than battery allows
        if plan.get('estimated_duration', 0) > 120:  # More than 2 hours
            warnings.append("Plan may exceed battery life")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def validate_safety_compliance(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate safety compliance"""
        issues = []
        warnings = []

        # Check for potentially unsafe actions
        for step in plan.get('task_sequence', []):
            target = step.get('target', '').lower()
            if any(unsafe in target for unsafe in ['fire', 'hot', 'dangerous']):
                issues.append(f"Potentially unsafe target '{target}' in step {step.get('step', 'unknown')}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def validate_logical_consistency(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical consistency of plan"""
        issues = []
        warnings = []

        # Check for contradictory actions
        actions = [step.get('action') for step in plan.get('task_sequence', [])]

        if 'navigate' in actions and 'stop' in actions:
            # Check if stop comes after navigate appropriately
            navigate_idx = next((i for i, a in enumerate(actions) if a == 'navigate'), -1)
            stop_idx = next((i for i, a in enumerate(actions) if a == 'stop'), -1)

            if navigate_idx != -1 and stop_idx != -1 and stop_idx < navigate_idx:
                warnings.append("Stop action appears before navigate action - check sequence")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }

    def repair_plan(self, plan: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
        """Attempt to repair a plan with issues"""
        repaired_plan = plan.copy()

        # Attempt repairs based on issue types
        for issue in issues:
            if 'invalid action' in issue.lower():
                # Replace invalid action with default
                for step in repaired_plan.get('task_sequence', []):
                    if step.get('action', 'unknown') not in ['navigate', 'detect', 'manipulate', 'communicate', 'wait', 'stop']:
                        step['action'] = 'unknown'  # or some safe default

        return repaired_plan


class ErrorRecoverySystem:
    """System for recovering from planning and execution errors"""

    def __init__(self):
        self.recovery_strategies = self.load_recovery_strategies()

    def load_recovery_strategies(self) -> Dict:
        """Load error recovery strategies"""
        return {
            'navigation_failure': self.handle_navigation_failure,
            'detection_failure': self.handle_detection_failure,
            'manipulation_failure': self.handle_manipulation_failure,
            'communication_failure': self.handle_communication_failure
        }

    def handle_error(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an error and suggest recovery"""
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)
        else:
            return self.generic_error_recovery(error_type, context)

    def handle_navigation_failure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle navigation failure"""
        return {
            'action': 'alternative_route',
            'suggestion': 'Try alternative path or manual guidance',
            'retry_allowed': True,
            'escalate': False
        }

    def handle_detection_failure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detection failure"""
        return {
            'action': 'enhanced_sensing',
            'suggestion': 'Use different sensors or approach angle',
            'retry_allowed': True,
            'escalate': False
        }

    def handle_manipulation_failure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manipulation failure"""
        return {
            'action': 'reposition',
            'suggestion': 'Reposition robot or adjust grasp strategy',
            'retry_allowed': True,
            'escalate': True  # Manipulation failures may require human intervention
        }

    def handle_communication_failure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle communication failure"""
        return {
            'action': 'fallback_communication',
            'suggestion': 'Use visual or gesture-based communication',
            'retry_allowed': True,
            'escalate': False
        }

    def generic_error_recovery(self, error_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic error recovery for unknown error types"""
        return {
            'action': 'stop_and_report',
            'suggestion': f'Unknown error type: {error_type}. Stopping execution.',
            'retry_allowed': False,
            'escalate': True
        }


def main(args=None):
    rclpy.init(args=args)

    # Use advanced planning node
    planning_node = AdvancedLLMPlanningNode()

    try:
        rclpy.spin(planning_node)
    except KeyboardInterrupt:
        planning_node.get_logger().info('LLM planning node stopped by user')
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Planning Techniques

#### Hierarchical Task Networks (HTN) with LLM

```python
class HTNPlanningWithLLM:
    """Hierarchical Task Network planning enhanced with LLM reasoning"""

    def __init__(self):
        self.task_methods = self.define_task_methods()
        self.method_refiner = MethodRefiner()

    def define_task_methods(self) -> Dict:
        """Define methods for high-level tasks"""
        return {
            'transport_object': [
                {
                    'name': 'fetch_and_carry',
                    'conditions': ['robot_can_navigate', 'robot_can_manipulate'],
                    'decomposition': [
                        {'task': 'navigate_to', 'args': {'location': '?source'}},
                        {'task': 'grasp_object', 'args': {'object': '?obj'}},
                        {'task': 'navigate_to', 'args': {'location': '?dest'}},
                        {'task': 'place_object', 'args': {'object': '?obj'}}
                    ]
                }
            ],
            'clean_workspace': [
                {
                    'name': 'clear_and_organize',
                    'conditions': ['workspace_identified', 'robot_has_manipulator'],
                    'decomposition': [
                        {'task': 'detect_obstacles', 'args': {'area': '?workspace'}},
                        {'task': 'remove_obstacle', 'args': {'obstacle': '?obs1'}},
                        {'task': 'remove_obstacle', 'args': {'obstacle': '?obs2'}},
                        {'task': 'verify_clean', 'args': {'area': '?workspace'}}
                    ]
                }
            ]
        }

    def plan_with_hierarchical_decomposition(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan using hierarchical task decomposition"""
        # Use LLM to select appropriate method for the goal
        selected_method = self.select_method_for_goal(goal)

        if selected_method:
            # Decompose the high-level task using the selected method
            subtasks = self.decompose_task(selected_method, goal)

            # Refine the plan using LLM
            refined_plan = self.method_refiner.refine_plan(subtasks, goal)

            return refined_plan
        else:
            # Fall back to basic planning
            return self.fallback_basic_planning(goal)

    def select_method_for_goal(self, goal: Dict[str, Any]) -> Optional[Dict]:
        """Select appropriate method for achieving the goal"""
        goal_type = goal.get('type', 'unknown')

        if goal_type in self.task_methods:
            methods = self.task_methods[goal_type]

            # Use LLM to select the best method based on context
            context = self.get_current_context()

            # For simulation, return the first method
            return methods[0] if methods else None

        return None

    def decompose_task(self, method: Dict, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a task using the specified method"""
        decomposition = method['decomposition']

        # Instantiate variables in the decomposition
        instantiated_tasks = []
        for task_def in decomposition:
            instantiated_task = self.instantiate_task_variables(task_def, goal)
            instantiated_tasks.append(instantiated_task)

        return instantiated_tasks

    def instantiate_task_variables(self, task_def: Dict, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Instantiate variables in a task definition"""
        instantiated = task_def.copy()
        args = instantiated['args'].copy()

        # Replace variables with actual values from goal
        for var_name, var_value in args.items():
            if var_value.startswith('?'):  # Variable to be instantiated
                # Look for corresponding value in goal
                if var_name in goal:
                    args[var_name] = goal[var_name]
                elif var_value[1:] in goal:  # Remove '?' and try
                    args[var_name] = goal[var_value[1:]]
                else:
                    # Use default or ask LLM for value
                    args[var_name] = self.get_default_value_for_variable(var_name, goal)

        instantiated['args'] = args
        return instantiated

    def get_default_value_for_variable(self, var_name: str, goal: Dict[str, Any]) -> Any:
        """Get default value for a variable"""
        defaults = {
            'source': 'current_location',
            'dest': 'default_destination',
            'obj': 'unknown_object',
            'workspace': 'current_workspace',
            'obstacle': 'first_detected_obstacle'
        }

        return defaults.get(var_name, f'default_{var_name}')

    def get_current_context(self) -> Dict[str, Any]:
        """Get current execution context"""
        # In a real implementation, this would come from perception and state tracking
        return {
            'robot_capabilities': ['navigation', 'manipulation'],
            'environment_state': 'indoor',
            'available_tools': ['gripper', 'camera']
        }

    def fallback_basic_planning(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fall back to basic planning if HTN fails"""
        # Convert goal to a simple task sequence
        return [
            {
                'step': 1,
                'action': 'unknown_action',
                'target': goal.get('target', 'unknown'),
                'parameters': goal.get('parameters', {}),
                'description': f'Achieve goal: {goal}'
            }
        ]


class MethodRefiner:
    """Refine HTN methods using LLM insights"""

    def __init__(self):
        # self.openai_client = OpenAI()  # Initialize as needed
        pass

    def refine_plan(self, plan: List[Dict[str, Any]], goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine plan using LLM insights"""
        try:
            # Create a prompt asking for plan refinement
            prompt = f"""
            You are a task planner for a humanoid robot. Refine the following plan to achieve the goal.

            Goal: {json.dumps(goal, indent=2)}

            Current Plan:
            {json.dumps(plan, indent=2)}

            Provide a refined plan that:
            1. Improves efficiency
            2. Adds safety checks
            3. Makes actions more specific
            4. Adds error handling where appropriate

            Return the refined plan in the same format as the input.
            """

            # In a real implementation, call the LLM
            # For simulation, return the original plan
            return self.mock_refine_plan(plan)

        except Exception as e:
            print(f"Plan refinement error: {e}")
            return plan

    def mock_refine_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock plan refinement for demonstration"""
        refined_plan = []

        for i, task in enumerate(plan):
            refined_task = task.copy()

            # Add safety checks
            if i > 0:  # Not the first task
                refined_task['safety_check'] = f"Verify preconditions for step {i+1}"

            # Make actions more specific
            if refined_task['action'] == 'navigate_to':
                refined_task['parameters'] = refined_task.get('parameters', {})
                refined_task['parameters']['avoid_obstacles'] = True
                refined_task['parameters']['speed'] = 'normal'

            refined_plan.append(refined_task)

        return refined_plan
```

### Isaac Integration for LLM Planning

#### Isaac Task Planning Integration

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.sensors import Camera
from omni.isaac.range_sensor import LidarRtx
import numpy as np

class IsaacLLMPlanner:
    def __init__(self, world: World, robot_prim_path: str):
        self.world = world
        self.robot = world.scene.get_object(robot_prim_path)

        # Initialize Isaac-specific sensors for planning
        self.camera = self.get_robot_camera()
        self.lidar = self.get_robot_lidar()

        # Planning context for Isaac simulation
        self.isaac_planning_context = IsaacPlanningContext(world)

    def get_robot_camera(self):
        """Get robot's camera sensor"""
        # Find camera attached to robot
        for prim in self.world.scene.objects:
            if "camera" in prim.name.lower() and self.robot.name in prim.name:
                return prim
        return None

    def get_robot_lidar(self):
        """Get robot's LiDAR sensor"""
        # Find LiDAR attached to robot
        for prim in self.world.scene.objects:
            if "lidar" in prim.name.lower() and self.robot.name in prim.name:
                return prim
        return None

    def plan_with_simulation_context(self, command: str) -> Dict[str, Any]:
        """Plan considering Isaac simulation context"""
        # Get current simulation state
        sim_context = self.isaac_planning_context.get_context()

        # Create enhanced planning prompt with simulation context
        enhanced_command = self.enhance_command_with_context(command, sim_context)

        # Generate plan using the enhanced command
        plan = self.generate_task_plan(enhanced_command)

        # Validate plan against simulation constraints
        validated_plan = self.validate_plan_in_simulation(plan)

        return validated_plan

    def enhance_command_with_context(self, command: str, sim_context: Dict) -> str:
        """Enhance command with simulation context"""
        # Add context information to command
        context_info = (
            f"The robot is currently in a simulated environment with known obstacles "
            f"and navigable locations. The robot's current position is {sim_context['robot_pose']}. "
            f"Known locations include: {', '.join(sim_context['known_locations'])}. "
            f"Detected obstacles: {sim_context['detected_obstacles']}. "
        )

        return f"{context_info} {command}"

    def validate_plan_in_simulation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plan against simulation constraints"""
        validated_plan = plan.copy()
        issues = []

        for step in validated_plan.get('task_sequence', []):
            action = step.get('action', '')
            target = step.get('target', '')

            if action == 'navigate':
                # Check if target location is navigable
                if not self.is_navigable_location(target):
                    issues.append(f"Location {target} may not be navigable")
                    # Suggest alternative
                    alternative = self.find_closest_navigable_location(target)
                    if alternative:
                        step['target'] = alternative
                        issues.append(f"Suggested alternative location: {alternative}")

            elif action == 'manipulate':
                # Check if object is manipulable
                if not self.is_manipulable_object(target):
                    issues.append(f"Object {target} may not be manipulable in simulation")

        validated_plan['validation_issues'] = issues
        validated_plan['is_valid_in_simulation'] = len(issues) == 0

        return validated_plan

    def is_navigable_location(self, location: str) -> bool:
        """Check if location is navigable in simulation"""
        # In simulation, check against known navigable locations
        known_navigable = self.isaac_planning_context.get_known_navigable_locations()
        return location.lower() in [loc.lower() for loc in known_navigable]

    def find_closest_navigable_location(self, target_location: str) -> Optional[str]:
        """Find closest navigable location to target"""
        known_locations = self.isaac_planning_context.get_known_navigable_locations()

        if known_locations:
            # For simulation, return the first known location
            return known_locations[0]

        return None

    def is_manipulable_object(self, object_name: str) -> bool:
        """Check if object is manipulable in simulation"""
        # In simulation, check if object exists and is manipulable
        scene_objects = self.isaac_planning_context.get_scene_objects()
        manipulable_objects = [obj for obj in scene_objects if obj.get('manipulable', False)]

        return any(obj['name'].lower() == object_name.lower() for obj in manipulable_objects)


class IsaacPlanningContext:
    """Context manager for Isaac simulation planning"""

    def __init__(self, world: World):
        self.world = world
        self.scene_analysis = self.analyze_scene()

    def analyze_scene(self) -> Dict[str, Any]:
        """Analyze current scene for planning context"""
        analysis = {
            'robot_pose': self.get_robot_pose(),
            'known_locations': self.get_known_locations(),
            'detected_obstacles': self.get_detected_obstacles(),
            'scene_objects': self.get_scene_objects(),
            'navigable_areas': self.get_navigable_areas()
        }
        return analysis

    def get_robot_pose(self) -> Dict[str, float]:
        """Get robot's current pose"""
        if self.world and self.world.scene and hasattr(self.world.scene, 'get_object'):
            robot = self.world.scene.get_object('/World/Robot')  # Adjust path as needed
            if robot:
                pos, orn = robot.get_world_pose()
                return {
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'z': float(pos[2]),
                    'qx': float(orn[0]),
                    'qy': float(orn[1]),
                    'qz': float(orn[2]),
                    'qw': float(orn[3])
                }

        # Default pose if robot not found
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0}

    def get_known_locations(self) -> List[str]:
        """Get known navigable locations in scene"""
        # In simulation, predefined locations
        return ['kitchen', 'living_room', 'bedroom', 'office', 'dining_room', 'hallway']

    def get_detected_obstacles(self) -> List[Dict[str, Any]]:
        """Get currently detected obstacles"""
        # In simulation, return obstacles from scene
        obstacles = []

        # For demonstration, add some simulated obstacles
        obstacles.append({
            'name': 'simulated_table',
            'position': {'x': 1.0, 'y': 0.5, 'z': 0.0},
            'type': 'furniture'
        })
        obstacles.append({
            'name': 'simulated_chair',
            'position': {'x': -0.5, 'y': 1.0, 'z': 0.0},
            'type': 'furniture'
        })

        return obstacles

    def get_scene_objects(self) -> List[Dict[str, Any]]:
        """Get objects in the scene"""
        objects = []

        # For demonstration, add some simulated objects
        objects.append({
            'name': 'red_cup',
            'position': {'x': 0.8, 'y': 0.3, 'z': 0.1},
            'manipulable': True,
            'type': 'container'
        })
        objects.append({
            'name': 'blue_book',
            'position': {'x': 0.9, 'y': 0.4, 'z': 0.1},
            'manipulable': True,
            'type': 'book'
        })
        objects.append({
            'name': 'wooden_box',
            'position': {'x': 1.2, 'y': 0.2, 'z': 0.1},
            'manipulable': True,
            'type': 'container'
        })

        return objects

    def get_navigable_areas(self) -> List[Dict[str, Any]]:
        """Get navigable areas in the scene"""
        # Define navigable areas in simulation
        return [
            {'name': 'main_floor', 'bounds': {'min_x': -5, 'max_x': 5, 'min_y': -5, 'max_y': 5}},
            {'name': 'kitchen_area', 'bounds': {'min_x': 2, 'max_x': 4, 'min_y': 0, 'max_y': 2}},
            {'name': 'living_room_area', 'bounds': {'min_x': -1, 'max_x': 1, 'min_y': -1, 'max_y': 1}}
        ]

    def get_known_navigable_locations(self) -> List[str]:
        """Get locations known to be navigable"""
        return ['kitchen', 'living_room', 'bedroom', 'office', 'dining_room']

    def get_context(self) -> Dict[str, Any]:
        """Get complete context"""
        return self.analysis
```

### Planning Quality Assurance

#### Plan Validation and Optimization

```python
class PlanQualityAssurance:
    """Ensure plan quality, safety, and optimality"""

    def __init__(self):
        self.quality_metrics = {
            'efficiency': 0.0,
            'safety': 0.0,
            'completeness': 0.0,
            'feasibility': 0.0
        }

    def validate_plan_comprehensive(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive plan validation"""
        validation_results = {
            'overall_validity': True,
            'quality_metrics': self.quality_metrics.copy(),
            'issues': [],
            'optimizations': [],
            'safety_checks': [],
            'efficiency_analysis': {}
        }

        # Check completeness
        completeness_result = self.check_plan_completeness(plan)
        validation_results['quality_metrics']['completeness'] = completeness_result['score']
        if not completeness_result['complete']:
            validation_results['issues'].extend(completeness_result['issues'])

        # Check feasibility
        feasibility_result = self.check_plan_feasibility(plan)
        validation_results['quality_metrics']['feasibility'] = feasibility_result['score']
        if not feasibility_result['feasible']:
            validation_results['issues'].extend(feasibility_result['issues'])

        # Check safety
        safety_result = self.check_plan_safety(plan)
        validation_results['quality_metrics']['safety'] = safety_result['score']
        validation_results['safety_checks'] = safety_result['checks']

        # Check efficiency
        efficiency_result = self.analyze_plan_efficiency(plan)
        validation_results['quality_metrics']['efficiency'] = efficiency_result['score']
        validation_results['efficiency_analysis'] = efficiency_result['analysis']
        validation_results['optimizations'] = efficiency_result['optimizations']

        # Overall validity
        validation_results['overall_validity'] = all(
            metric >= 0.7 for metric in validation_results['quality_metrics'].values()
        )

        return validation_results

    def check_plan_completeness(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check if plan is complete"""
        issues = []
        required_fields = ['command', 'intent', 'task_sequence']

        for field in required_fields:
            if field not in plan:
                issues.append(f'Missing required field: {field}')

        # Check task sequence completeness
        task_sequence = plan.get('task_sequence', [])
        for i, task in enumerate(task_sequence):
            required_task_fields = ['step', 'action', 'target']
            for field in required_task_fields:
                if field not in task:
                    issues.append(f'Missing field {field} in task {i+1}')

        completeness_score = 1.0 - (len(issues) * 0.1)  # Deduct 0.1 per issue
        completeness_score = max(0.0, completeness_score)  # Ensure non-negative

        return {
            'complete': len(issues) == 0,
            'score': completeness_score,
            'issues': issues
        }

    def check_plan_feasibility(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check if plan is feasible"""
        issues = []

        # Check if actions are supported
        supported_actions = ['navigate', 'detect', 'manipulate', 'communicate', 'wait', 'stop']
        task_sequence = plan.get('task_sequence', [])

        for i, task in enumerate(task_sequence):
            action = task.get('action', 'unknown')
            if action not in supported_actions:
                issues.append(f'Unsupported action {action} in task {i+1}')

        # Check resource feasibility
        resource_issues = self.check_resource_feasibility(plan)
        issues.extend(resource_issues)

        feasibility_score = 1.0 - (len(issues) * 0.1)
        feasibility_score = max(0.0, feasibility_score)

        return {
            'feasible': len(issues) == 0,
            'score': feasibility_score,
            'issues': issues
        }

    def check_resource_feasibility(self, plan: Dict[str, Any]) -> List[str]:
        """Check if plan is feasible given robot resources"""
        issues = []

        # Estimate resource usage
        total_battery_usage = 0
        for task in plan.get('task_sequence', []):
            action = task.get('action', '')
            if action == 'navigate':
                total_battery_usage += 0.1  # Estimate
            elif action == 'manipulate':
                total_battery_usage += 0.05
            elif action == 'detect':
                total_battery_usage += 0.02

        # Check if battery usage is reasonable
        if total_battery_usage > 0.8:  # More than 80% of battery
            issues.append(f'Estimated battery usage ({total_battery_usage:.2f}) exceeds safe threshold')

        return issues

    def check_plan_safety(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Check plan safety"""
        safety_checks = []
        issues = []

        # Check for potentially unsafe actions
        for i, task in enumerate(plan.get('task_sequence', [])):
            action = task.get('action', '')
            target = task.get('target', '').lower()

            # Safety check for navigation near hazards
            if action == 'navigate' and any(hazard in target for hazard in ['fire', 'hot', 'danger', 'unsafe']):
                safety_issue = {
                    'type': 'hazard_navigation',
                    'task': i+1,
                    'severity': 'high',
                    'recommendation': 'Verify safety before navigation'
                }
                safety_checks.append(safety_issue)
                issues.append(f'Potential safety hazard in navigation task {i+1}')

            # Safety check for manipulation of hazardous objects
            if action == 'manipulate' and any(hazard in target for hazard in ['knife', 'blade', 'sharp', 'hot']):
                safety_issue = {
                    'type': 'hazardous_manipulation',
                    'task': i+1,
                    'severity': 'high',
                    'recommendation': 'Use caution or avoid manipulation'
                }
                safety_checks.append(safety_issue)
                issues.append(f'Potential safety hazard in manipulation task {i+1}')

        safety_score = 1.0 - (len(issues) * 0.2)  # Deduct 0.2 per safety issue
        safety_score = max(0.0, safety_score)

        return {
            'score': safety_score,
            'checks': safety_checks,
            'issues': issues
        }

    def analyze_plan_efficiency(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze plan efficiency and suggest optimizations"""
        analysis = {}
        optimizations = []

        # Analyze task sequence for optimization opportunities
        task_sequence = plan.get('task_sequence', [])
        analysis['total_tasks'] = len(task_sequence)

        # Look for redundant tasks
        redundant_tasks = self.find_redundant_tasks(task_sequence)
        if redundant_tasks:
            analysis['redundant_tasks'] = redundant_tasks
            optimizations.append({
                'type': 'remove_redundancy',
                'tasks': redundant_tasks,
                'savings': f'Remove {len(redundant_tasks)} redundant tasks'
            })

        # Look for tasks that can be parallelized
        parallelizable_tasks = self.find_parallelizable_tasks(task_sequence)
        if parallelizable_tasks:
            analysis['parallelizable'] = parallelizable_tasks
            optimizations.append({
                'type': 'parallelize',
                'tasks': parallelizable_tasks,
                'benefit': 'Potential for parallel execution'
            })

        # Estimate efficiency improvement
        estimated_improvement = len(redundant_tasks) * 0.1 + len(parallelizable_tasks) * 0.05
        efficiency_score = 1.0 - estimated_improvement
        efficiency_score = max(0.1, efficiency_score)  # Minimum 0.1 efficiency

        return {
            'score': efficiency_score,
            'analysis': analysis,
            'optimizations': optimizations
        }

    def find_redundant_tasks(self, task_sequence: List[Dict]) -> List[int]:
        """Find redundant tasks in sequence"""
        redundant_indices = []

        for i in range(len(task_sequence) - 1):
            current_task = task_sequence[i]
            next_task = task_sequence[i + 1]

            # Check if tasks are redundant (same action, same target)
            if (current_task.get('action') == next_task.get('action') and
                current_task.get('target') == next_task.get('target')):
                redundant_indices.append(i + 1)  # Mark next task as redundant

        return redundant_indices

    def find_parallelizable_tasks(self, task_sequence: List[Dict]) -> List[List[int]]:
        """Find tasks that can be executed in parallel"""
        parallelizable_groups = []

        # Simple analysis: tasks that don't depend on each other's results
        # In a real implementation, this would be more sophisticated
        for i in range(len(task_sequence) - 1):
            current_task = task_sequence[i]
            next_task = task_sequence[i + 1]

            # If tasks are independent, they could potentially be parallelized
            if self.are_tasks_independent(current_task, next_task):
                parallelizable_groups.append([i, i + 1])

        return parallelizable_groups

    def are_tasks_independent(self, task1: Dict, task2: Dict) -> bool:
        """Check if two tasks are independent"""
        # For now, consider tasks independent if they don't target the same resource
        same_target = task1.get('target') == task2.get('target')
        same_resource = self.get_resource_usage(task1) & self.get_resource_usage(task2)

        return not (same_target or same_resource)

    def get_resource_usage(self, task: Dict) -> set:
        """Get resources used by a task"""
        action = task.get('action', '')
        resource_map = {
            'navigate': {'navigation_system', 'battery'},
            'manipulate': {'manipulator', 'camera', 'battery'},
            'detect': {'camera', 'battery'},
            'communicate': {'speakers', 'microphone'},
            'wait': {'battery'}
        }
        return resource_map.get(action, set())
```

The LLM-based cognitive planning system enables humanoid robots to understand complex natural language commands and generate appropriate action sequences. The system includes multi-step reasoning, plan validation, error recovery, and simulation integration to ensure safe and effective task execution.