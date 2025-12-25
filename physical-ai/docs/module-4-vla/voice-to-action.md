---
sidebar_position: 2
---

# Voice-to-Action Pipeline

## Converting Natural Language Commands to Robot Actions

The voice-to-action pipeline is a critical component that transforms spoken language commands into executable robot actions. This module covers the complete pipeline from speech recognition to action execution.

### Voice-to-Action Architecture

The voice-to-action pipeline follows this sequence:

```
Voice Input → Speech Recognition → Natural Language Processing → Intent Extraction → Action Planning → Action Execution → Feedback
```

### Complete Voice-to-Action Implementation

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist, PoseStamped
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import speech_recognition as sr
import openai
import json
import asyncio
import threading
import queue
from typing import Dict, Any, Optional
import time

class VoiceToActionNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')

        # Initialize components
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize OpenAI client
        # openai.api_key = self.get_parameter_or('openai_api_key', 'YOUR_API_KEY')

        # Create publishers
        self.voice_cmd_pub = self.create_publisher(
            String, '/voice_command', 10
        )
        self.action_status_pub = self.create_publisher(
            String, '/action_status', 10
        )

        # Create subscribers
        self.voice_sub = self.create_subscription(
            String,
            '/recognized_text',
            self.voice_callback,
            10
        )

        # Create action clients
        self.nav_action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )
        self.manipulation_action_client = ActionClient(
            self,
            ManipulateObject,
            'manipulate_object'
        )

        # Voice processing components
        self.voice_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self.voice_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Command history and context
        self.command_history = []
        self.context = {}

        # Configuration
        self.declare_parameter('command_timeout', 30.0)
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('listen_continuously', True)

        self.get_logger().info('Voice-to-action node initialized')

    def voice_processing_loop(self):
        """Background thread for voice processing"""
        while rclpy.ok():
            try:
                # Get voice command from queue
                voice_data = self.voice_queue.get(timeout=1.0)

                # Process the command
                result = self.process_voice_command(voice_data)

                # Publish results
                status_msg = String()
                status_msg.data = json.dumps(result)
                self.action_status_pub.publish(status_msg)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Voice processing error: {e}')

    def voice_callback(self, msg):
        """Handle incoming voice commands"""
        try:
            # Add to processing queue
            self.voice_queue.put_nowait(msg.data)
        except queue.Full:
            self.get_logger().warn('Voice queue full, dropping command')

    def process_voice_command(self, command_text):
        """Process a voice command through the complete pipeline"""
        start_time = time.time()

        try:
            # Step 1: Validate command
            if not self.validate_command(command_text):
                return {
                    'success': False,
                    'message': 'Invalid command',
                    'command': command_text,
                    'timestamp': start_time
                }

            # Step 2: Parse intent using LLM
            intent = self.parse_intent_with_llm(command_text)

            if not intent:
                return {
                    'success': False,
                    'message': 'Could not understand command',
                    'command': command_text,
                    'timestamp': start_time
                }

            # Step 3: Plan action sequence
            action_plan = self.plan_action_sequence(intent)

            if not action_plan:
                return {
                    'success': False,
                    'message': 'Could not plan action sequence',
                    'intent': intent,
                    'command': command_text,
                    'timestamp': start_time
                }

            # Step 4: Execute action plan
            execution_result = self.execute_action_plan(action_plan)

            # Step 5: Return results
            result = {
                'success': execution_result['success'],
                'message': execution_result['message'],
                'intent': intent,
                'action_plan': action_plan,
                'execution_result': execution_result,
                'command': command_text,
                'timestamp': start_time,
                'duration': time.time() - start_time
            }

            # Update command history
            self.command_history.append(result)

            return result

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')
            return {
                'success': False,
                'message': f'Processing error: {str(e)}',
                'command': command_text,
                'timestamp': start_time
            }

    def validate_command(self, command_text):
        """Validate if command is appropriate for processing"""
        # Check for minimum length
        if len(command_text.strip()) < 3:
            return False

        # Check for potentially harmful commands
        harmful_keywords = ['shutdown', 'power off', 'terminate']
        if any(keyword in command_text.lower() for keyword in harmful_keywords):
            return False

        # Check for robot-specific commands
        valid_actions = [
            'navigate', 'go to', 'move to', 'walk to', 'drive to',
            'pick up', 'grasp', 'take', 'get', 'grab', 'lift',
            'put', 'place', 'drop', 'release',
            'find', 'look for', 'locate', 'search',
            'stop', 'wait', 'pause', 'continue'
        ]

        # At least one valid action keyword should be present
        command_lower = command_text.lower()
        return any(action in command_lower for action in valid_actions)

    def parse_intent_with_llm(self, command_text):
        """Parse intent using large language model"""
        try:
            # Create a structured prompt for intent parsing
            prompt = f"""
            You are a voice command parser for a humanoid robot.
            Parse the following command and return the intent in JSON format.

            Command: "{command_text}"

            Return a JSON object with these fields:
            - action: The main action (navigate, manipulate, detect, etc.)
            - target: The target of the action (location, object, etc.)
            - parameters: Additional parameters as needed
            - operation: Specific operation (for manipulation tasks)

            Example responses:
            {{
                "action": "navigate",
                "target": "kitchen",
                "parameters": {{"speed": "normal"}},
                "operation": null
            }}

            {{
                "action": "manipulate",
                "target": "red cup",
                "parameters": {{"height": 0.5}},
                "operation": "grasp"
            }}

            {{
                "action": "detect",
                "target": "keys",
                "parameters": {{"location": "table"}},
                "operation": null
            }}
            """

            # In a real implementation, this would call the LLM
            # For simulation, return a mock response
            return self.mock_intent_parsing(command_text)

        except Exception as e:
            self.get_logger().error(f'LLM intent parsing error: {e}')
            return None

    def mock_intent_parsing(self, command_text):
        """Mock intent parsing for demonstration"""
        import re

        command_lower = command_text.lower()

        # Navigation commands
        if any(word in command_lower for word in ['go to', 'navigate to', 'move to', 'walk to', 'drive to']):
            # Extract location
            match = re.search(r'(?:go to|navigate to|move to|walk to|drive to)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                location = match.group(1).strip()
                return {
                    "action": "navigate",
                    "target": location,
                    "parameters": {"speed": "normal"},
                    "operation": None
                }

        # Manipulation commands
        elif any(word in command_lower for word in ['pick up', 'grasp', 'take', 'get', 'grab', 'lift']):
            # Extract object
            match = re.search(r'(?:pick up|grasp|take|get|grab|lift)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                obj = match.group(1).strip()
                return {
                    "action": "manipulate",
                    "target": obj,
                    "parameters": {"height": 0.5},
                    "operation": "grasp"
                }

        # Placement commands
        elif any(word in command_lower for word in ['put', 'place', 'drop', 'release']):
            # Extract object and target
            match = re.search(r'(?:put|place|drop|release)\s+(.+?)\s+(?:on|at|in)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                obj = match.group(1).strip()
                target = match.group(2).strip()
                return {
                    "action": "manipulate",
                    "target": obj,
                    "parameters": {"destination": target},
                    "operation": "place"
                }

        # Detection commands
        elif any(word in command_lower for word in ['find', 'look for', 'locate', 'search for']):
            # Extract object to find
            match = re.search(r'(?:find|look for|locate|search for)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                obj = match.group(1).strip()
                return {
                    "action": "detect",
                    "target": obj,
                    "parameters": {"search_radius": 2.0},
                    "operation": None
                }

        # Stop commands
        elif any(word in command_lower for word in ['stop', 'halt', 'pause']):
            return {
                "action": "stop",
                "target": None,
                "parameters": {},
                "operation": None
            }

        # Unknown command
        return {
            "action": "unknown",
            "target": command_text,
            "parameters": {},
            "operation": None
        }

    def plan_action_sequence(self, intent):
        """Plan sequence of actions based on intent"""
        action_plan = []

        action = intent['action']
        target = intent['target']
        parameters = intent.get('parameters', {})
        operation = intent.get('operation')

        if action == 'navigate':
            # Plan navigation to target location
            nav_action = {
                'type': 'navigation',
                'target_location': target,
                'parameters': parameters
            }
            action_plan.append(nav_action)

        elif action == 'manipulate':
            if operation == 'grasp':
                # Plan navigation to object, then grasp
                nav_action = {
                    'type': 'navigation',
                    'target_location': f'near_{target}',
                    'parameters': {'approach_distance': 0.5}
                }
                action_plan.append(nav_action)

                grasp_action = {
                    'type': 'manipulation',
                    'operation': 'grasp',
                    'target_object': target,
                    'parameters': parameters
                }
                action_plan.append(grasp_action)

            elif operation == 'place':
                # Plan navigation to placement location, then place
                nav_action = {
                    'type': 'navigation',
                    'target_location': parameters.get('destination', 'default_place'),
                    'parameters': {'approach_distance': 0.5}
                }
                action_plan.append(nav_action)

                place_action = {
                    'type': 'manipulation',
                    'operation': 'place',
                    'target_object': target,
                    'parameters': parameters
                }
                action_plan.append(place_action)

        elif action == 'detect':
            # Plan search for object
            search_action = {
                'type': 'search',
                'target_object': target,
                'parameters': parameters
            }
            action_plan.append(search_action)

        elif action == 'stop':
            # Plan to stop current actions
            stop_action = {
                'type': 'stop',
                'parameters': {}
            }
            action_plan.append(stop_action)

        elif action == 'unknown':
            # No action for unknown commands
            pass

        return action_plan

    def execute_action_plan(self, action_plan):
        """Execute the planned sequence of actions"""
        results = []

        for i, action in enumerate(action_plan):
            self.get_logger().info(f'Executing action {i+1}/{len(action_plan)}: {action["type"]}')

            try:
                if action['type'] == 'navigation':
                    result = self.execute_navigation_action(action)
                elif action['type'] == 'manipulation':
                    result = self.execute_manipulation_action(action)
                elif action['type'] == 'search':
                    result = self.execute_search_action(action)
                elif action['type'] == 'stop':
                    result = self.execute_stop_action(action)
                else:
                    result = {'success': False, 'message': f'Unknown action type: {action["type"]}'}

                results.append(result)

                # Check if action failed and should stop execution
                if not result['success'] and action.get('critical', True):
                    return {
                        'success': False,
                        'message': f'Action failed: {result["message"]}',
                        'results': results
                    }

            except Exception as e:
                error_result = {
                    'success': False,
                    'message': f'Execution error: {str(e)}'
                }
                results.append(error_result)
                return {
                    'success': False,
                    'message': f'Action execution error: {str(e)}',
                    'results': results
                }

        return {
            'success': all(r['success'] for r in results),
            'message': 'Action plan completed successfully',
            'results': results
        }

    def execute_navigation_action(self, action):
        """Execute navigation action"""
        try:
            # In a real implementation, this would send a navigation goal
            # For simulation, we'll mock the navigation

            target_location = action['target_location']
            parameters = action.get('parameters', {})

            # Simulate navigation
            self.get_logger().info(f'Navigating to {target_location}')

            # Mock navigation success
            success = True  # In real implementation, this would come from navigation feedback

            return {
                'success': success,
                'message': f'Navigation to {target_location} completed',
                'action': 'navigation',
                'target': target_location
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Navigation error: {str(e)}',
                'action': 'navigation'
            }

    def execute_manipulation_action(self, action):
        """Execute manipulation action"""
        try:
            operation = action['operation']
            target_object = action['target_object']
            parameters = action.get('parameters', {})

            self.get_logger().info(f'Performing {operation} on {target_object}')

            # Simulate manipulation
            success = True  # In real implementation, this would come from manipulation feedback

            return {
                'success': success,
                'message': f'{operation.capitalize()} of {target_object} completed',
                'action': 'manipulation',
                'operation': operation,
                'target': target_object
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Manipulation error: {str(e)}',
                'action': 'manipulation'
            }

    def execute_search_action(self, action):
        """Execute search action"""
        try:
            target_object = action['target_object']
            parameters = action.get('parameters', {})

            self.get_logger().info(f'Searching for {target_object}')

            # Simulate search
            found = True  # In real implementation, this would come from perception system

            return {
                'success': found,
                'message': f'Search for {target_object} {"successful" if found else "unsuccessful"}',
                'action': 'search',
                'target': target_object
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Search error: {str(e)}',
                'action': 'search'
            }

    def execute_stop_action(self, action):
        """Execute stop action"""
        try:
            # In a real implementation, this would cancel current goals
            self.nav_action_client.cancel_all_goals_async()
            self.manipulation_action_client.cancel_all_goals_async()

            return {
                'success': True,
                'message': 'All actions stopped',
                'action': 'stop'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Stop error: {str(e)}',
                'action': 'stop'
            }


class AdvancedVoiceToActionNode(VoiceToActionNode):
    """Advanced voice-to-action with context awareness and multi-step planning"""

    def __init__(self):
        super().__init__()

        # Context management
        self.conversation_context = ConversationContext()
        self.task_manager = TaskManager()

        # Create additional publishers for context
        self.context_pub = self.create_publisher(
            String, '/conversation_context', 10
        )

    def process_voice_command(self, command_text):
        """Enhanced processing with context awareness"""
        start_time = time.time()

        try:
            # Update conversation context
            self.conversation_context.add_user_input(command_text)

            # Apply context to command interpretation
            contextual_command = self.conversation_context.apply_context(command_text)

            # Parse intent with context
            intent = self.parse_intent_with_context(contextual_command)

            # Plan with context
            action_plan = self.plan_action_sequence_with_context(intent)

            # Execute with context
            execution_result = self.execute_action_plan_with_context(action_plan)

            # Update context with results
            self.conversation_context.add_system_output(execution_result)

            # Publish updated context
            context_msg = String()
            context_msg.data = json.dumps(self.conversation_context.get_context())
            self.context_pub.publish(context_msg)

            result = {
                'success': execution_result['success'],
                'message': execution_result['message'],
                'intent': intent,
                'action_plan': action_plan,
                'execution_result': execution_result,
                'command': command_text,
                'context': self.conversation_context.get_context(),
                'timestamp': start_time,
                'duration': time.time() - start_time
            }

            # Update command history
            self.command_history.append(result)

            return result

        except Exception as e:
            self.get_logger().error(f'Error in advanced processing: {e}')
            return {
                'success': False,
                'message': f'Advanced processing error: {str(e)}',
                'command': command_text,
                'timestamp': start_time
            }

    def parse_intent_with_context(self, command_text):
        """Parse intent considering conversation context"""
        # Get relevant context
        recent_context = self.conversation_context.get_recent_context()

        # Create enhanced prompt with context
        prompt = f"""
        You are a voice command parser for a humanoid robot.
        Consider the conversation context when interpreting this command.

        Conversation Context:
        {json.dumps(recent_context, indent=2)}

        Command: "{command_text}"

        Parse the command considering the context and return the intent in JSON format.
        Pay attention to pronouns, references to previous objects/locations, and continuation of tasks.

        Return a JSON object with:
        - action: The main action
        - target: The target of the action
        - parameters: Additional parameters
        - operation: Specific operation
        - resolved_references: Dictionary of resolved pronouns/references
        """

        # In real implementation, call LLM with context
        # For simulation, use mock with context
        return self.mock_intent_parsing_with_context(command_text, recent_context)

    def mock_intent_parsing_with_context(self, command_text, context):
        """Mock intent parsing with context consideration"""
        import re

        # Handle pronouns and references
        resolved_command = command_text.lower()

        # If context contains recent object, resolve "it" or "that"
        if 'recent_object' in context:
            resolved_command = resolved_command.replace('it', context['recent_object'])
            resolved_command = resolved_command.replace('that', context['recent_object'])

        # If context contains recent location, resolve location references
        if 'recent_location' in context:
            resolved_command = resolved_command.replace('there', context['recent_location'])

        # Parse the resolved command
        intent = self.mock_intent_parsing(resolved_command)

        # Add resolved references
        intent['resolved_references'] = {
            'original_command': command_text,
            'resolved_command': resolved_command
        }

        return intent

    def plan_action_sequence_with_context(self, intent):
        """Plan actions considering context and history"""
        action_plan = self.plan_action_sequence(intent)

        # Enhance plan with context
        context_enhanced_plan = []
        for action in action_plan:
            # Add context-specific parameters
            enhanced_action = action.copy()

            # Example: adjust navigation based on previous locations
            if action['type'] == 'navigation':
                enhanced_action['parameters']['preferred_path'] = self.get_preferred_path(
                    action['target_location']
                )

            context_enhanced_plan.append(enhanced_action)

        return context_enhanced_plan

    def execute_action_plan_with_context(self, action_plan):
        """Execute plan with context awareness"""
        return self.execute_action_plan(action_plan)

    def get_preferred_path(self, target_location):
        """Get preferred path based on context and history"""
        # In real implementation, this would use path preferences learned from history
        return 'shortest'


class ConversationContext:
    """Manage conversation context for voice-to-action system"""

    def __init__(self):
        self.history = []
        self.current_context = {}
        self.max_context_items = 10

    def add_user_input(self, user_input):
        """Add user input to context"""
        self.history.append({
            'type': 'user_input',
            'content': user_input,
            'timestamp': time.time()
        })

        # Maintain history size
        if len(self.history) > self.max_context_items:
            self.history = self.history[-self.max_context_items:]

    def add_system_output(self, system_output):
        """Add system output to context"""
        self.history.append({
            'type': 'system_output',
            'content': system_output,
            'timestamp': time.time()
        })

        # Maintain history size
        if len(self.history) > self.max_context_items:
            self.history = self.history[-self.max_context_items:]

    def get_recent_context(self):
        """Get recent conversation context"""
        return {
            'history': self.history[-5:],  # Last 5 exchanges
            'current_context': self.current_context
        }

    def apply_context(self, command):
        """Apply context to command interpretation"""
        # This would resolve pronouns, references, etc.
        # For now, return command as-is
        return command

    def get_context(self):
        """Get current context state"""
        return {
            'history': self.history,
            'current_context': self.current_context,
            'summary': self.summarize_context()
        }

    def summarize_context(self):
        """Summarize current conversation context"""
        if not self.history:
            return "No conversation history"

        # Find recent objects and locations mentioned
        recent_objects = []
        recent_locations = []

        for item in self.history[-5:]:
            content = item.get('content', '')
            if isinstance(content, dict):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)

            # Extract potential objects and locations
            # This is a simplified version - in practice, use NLP
            if 'target' in content:
                recent_objects.append(content)
            if 'location' in content:
                recent_locations.append(content)

        return {
            'recent_objects': recent_objects[-3:],
            'recent_locations': recent_locations[-3:],
            'conversation_turns': len(self.history)
        }


class TaskManager:
    """Manage complex multi-step tasks"""

    def __init__(self):
        self.active_tasks = {}
        self.task_history = []

    def create_task(self, task_id, task_definition):
        """Create a new task"""
        task = {
            'id': task_id,
            'definition': task_definition,
            'status': 'created',
            'created_at': time.time(),
            'steps': [],
            'current_step': 0,
            'results': []
        }

        self.active_tasks[task_id] = task
        return task

    def update_task_status(self, task_id, status, result=None):
        """Update task status"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task['status'] = status
            task['updated_at'] = time.time()

            if result:
                task['results'].append(result)

    def get_active_tasks(self):
        """Get all active tasks"""
        return self.active_tasks


def main(args=None):
    rclpy.init(args=args)

    # Use advanced voice-to-action node
    voice_node = AdvancedVoiceToActionNode()

    try:
        rclpy.spin(voice_node)
    except KeyboardInterrupt:
        voice_node.get_logger().info('Voice-to-action node stopped by user')
    finally:
        voice_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Voice Processing Techniques

#### Speech Recognition with Context

```python
import speech_recognition as sr
import pyaudio
import threading
import queue
import time

class AdvancedSpeechRecognition:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Set recognition parameters
        self.recognizer.energy_threshold = 300  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True

        # Language model
        self.language = "en-US"

        # Processing queues
        self.audio_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()

        # Context for speech recognition
        self.speech_context = []

    def listen_continuously(self, callback=None):
        """Listen continuously for speech commands"""
        def audio_callback(recognizer, audio):
            try:
                # Recognize speech
                text = recognizer.recognize_google(audio, language=self.language)

                # Add to processing queue
                self.audio_queue.put_nowait({
                    'text': text,
                    'timestamp': time.time(),
                    'confidence': 0.9  # Google's API doesn't return confidence
                })

                # Call callback if provided
                if callback:
                    callback(text)

            except sr.UnknownValueError:
                pass  # Audio was not understood
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")

        # Start listening
        stop_listening = self.recognizer.listen_in_background(self.microphone, audio_callback)

        return stop_listening

    def recognize_with_context(self, audio_data, context_phrases=None):
        """Recognize speech with context-specific phrases"""
        try:
            # In real implementation, use context-specific language model
            # For now, use standard recognition with potential context hints
            if context_phrases:
                # This would use a custom language model with context phrases
                # Google Speech API doesn't support this directly, but alternatives do
                pass

            text = self.recognizer.recognize_google(audio_data, language=self.language)
            return text

        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

    def add_context_phrase(self, phrase):
        """Add a phrase to the recognition context"""
        if phrase not in self.speech_context:
            self.speech_context.append(phrase)

    def clear_context(self):
        """Clear speech recognition context"""
        self.speech_context.clear()
```

#### Intent Classification Pipeline

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class IntentClassifier:
    def __init__(self):
        # Initialize pre-trained model for intent classification
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'microsoft/DialoGPT-medium',
            num_labels=8  # Adjust based on number of intents
        )

        # Define intent classes
        self.intent_classes = [
            'navigate',
            'manipulate',
            'detect',
            'query',
            'stop',
            'greeting',
            'confirmation',
            'unknown'
        ]

        # Alternative: Simple keyword-based classifier
        self.keyword_classifier = KeywordIntentClassifier()

    def classify_intent(self, text):
        """Classify intent of text command"""
        # Try transformer-based classification first
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()

            return {
                'intent': self.intent_classes[predicted_class],
                'confidence': confidence,
                'all_scores': {self.intent_classes[i]: score.item()
                              for i, score in enumerate(predictions[0])}
            }
        except:
            # Fall back to keyword-based classification
            return self.keyword_classifier.classify_intent(text)

class KeywordIntentClassifier:
    """Simple keyword-based intent classifier for fallback"""

    def __init__(self):
        self.intent_keywords = {
            'navigate': [
                'go to', 'navigate to', 'move to', 'walk to', 'drive to', 'travel to',
                'reach', 'arrive at', 'head to', 'proceed to'
            ],
            'manipulate': [
                'pick up', 'grasp', 'take', 'get', 'grab', 'lift', 'catch',
                'put', 'place', 'drop', 'release', 'set down'
            ],
            'detect': [
                'find', 'look for', 'locate', 'search for', 'spot', 'identify',
                'where is', 'find the', 'see', 'show me'
            ],
            'query': [
                'what', 'how', 'where', 'when', 'who', 'which',
                'tell me', 'explain', 'describe', 'information'
            ],
            'stop': [
                'stop', 'halt', 'pause', 'wait', 'cease', 'quit'
            ],
            'greeting': [
                'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'
            ],
            'confirmation': [
                'yes', 'okay', 'sure', 'fine', 'alright', 'correct', 'right'
            ]
        }

    def classify_intent(self, text):
        """Classify intent using keyword matching"""
        text_lower = text.lower()

        scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[intent] = score

        # Find intent with highest score
        if scores:
            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]

            # Calculate confidence (normalize by length of text)
            confidence = min(max_score / len(text.split()), 1.0)

            return {
                'intent': best_intent if max_score > 0 else 'unknown',
                'confidence': confidence,
                'all_scores': scores
            }
        else:
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'all_scores': {}
            }
```

### Isaac Integration for Voice Commands

#### Isaac Voice Command Processing

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.objects import DynamicCuboid
import numpy as np

class IsaacVoiceCommandProcessor:
    def __init__(self, world: World):
        self.world = world
        self.robot = None
        self.scene_objects = {}

        # Initialize voice command mapping to Isaac actions
        self.command_mappings = {
            'navigate': self.execute_navigation,
            'move_to': self.execute_navigation,
            'go_to': self.execute_navigation,
            'pick_up': self.execute_manipulation,
            'grasp': self.execute_manipulation,
            'take': self.execute_manipulation,
            'place': self.execute_manipulation,
            'put': self.execute_manipulation,
            'find': self.execute_detection,
            'look_for': self.execute_detection,
            'locate': self.execute_detection
        }

    def set_robot(self, robot_prim_path: str):
        """Set the robot to control"""
        self.robot = self.world.scene.get_object(robot_prim_path)

    def execute_voice_command(self, command_data: dict):
        """Execute voice command in Isaac simulation"""
        intent = command_data.get('intent', {})
        action = intent.get('action', 'unknown')
        target = intent.get('target', '')

        if action in self.command_mappings:
            return self.command_mappings[action](target, intent.get('parameters', {}))
        else:
            return {
                'success': False,
                'message': f'Unknown action: {action}',
                'command': command_data
            }

    def execute_navigation(self, target_location: str, parameters: dict):
        """Execute navigation command in simulation"""
        try:
            # Convert location description to coordinates
            target_coords = self.resolve_location_to_coordinates(target_location)

            if target_coords is None:
                return {
                    'success': False,
                    'message': f'Could not resolve location: {target_location}',
                    'target': target_location
                }

            # Move robot to target coordinates
            if self.robot:
                self.robot.set_world_pose(position=target_coords)

                # Simulate navigation by interpolating position
                start_pos, _ = self.robot.get_world_pose()
                steps = 10

                for i in range(steps + 1):
                    ratio = i / steps
                    interp_pos = start_pos + ratio * (target_coords - start_pos)
                    self.robot.set_world_pose(position=interp_pos)

                    # Step the simulation
                    self.world.step(render=True)

                return {
                    'success': True,
                    'message': f'Navigated to {target_location}',
                    'target_coordinates': target_coords.tolist()
                }
            else:
                return {
                    'success': False,
                    'message': 'No robot set for navigation',
                    'target': target_location
                }

        except Exception as e:
            return {
                'success': False,
                'message': f'Navigation error: {str(e)}',
                'target': target_location
            }

    def execute_manipulation(self, target_object: str, parameters: dict):
        """Execute manipulation command in simulation"""
        try:
            # Find the object to manipulate
            target_prim = self.find_object_by_name(target_object)

            if target_prim is None:
                return {
                    'success': False,
                    'message': f'Could not find object: {target_object}',
                    'target': target_object
                }

            # Get object position
            obj_pos, obj_orn = target_prim.get_world_pose()

            # Approach the object
            approach_pos = obj_pos.copy()
            approach_pos[2] += 0.2  # Slightly above object

            # Move robot near object
            if self.robot:
                robot_pos, robot_orn = self.robot.get_world_pose()

                # Interpolate approach
                steps = 10
                for i in range(steps + 1):
                    ratio = i / steps
                    interp_pos = robot_pos + ratio * (approach_pos - robot_pos)
                    self.robot.set_world_pose(position=interp_pos)
                    self.world.step(render=True)

                # Perform manipulation (simplified)
                # In a real implementation, this would involve gripper control
                manipulation_result = self.simulate_manipulation(target_prim, parameters)

                return {
                    'success': manipulation_result['success'],
                    'message': manipulation_result['message'],
                    'manipulated_object': target_object,
                    'action': parameters.get('operation', 'grasp')
                }
            else:
                return {
                    'success': False,
                    'message': 'No robot set for manipulation',
                    'target': target_object
                }

        except Exception as e:
            return {
                'success': False,
                'message': f'Manipulation error: {str(e)}',
                'target': target_object
            }

    def execute_detection(self, target_object: str, parameters: dict):
        """Execute detection command in simulation"""
        try:
            # Search for the object in the scene
            found_objects = self.find_objects_by_name(target_object)

            if found_objects:
                # Return information about found objects
                results = []
                for obj in found_objects:
                    pos, orn = obj.get_world_pose()
                    results.append({
                        'name': obj.name,
                        'position': pos.tolist(),
                        'orientation': orn.tolist()
                    })

                return {
                    'success': True,
                    'message': f'Found {len(found_objects)} instances of {target_object}',
                    'found_objects': results
                }
            else:
                return {
                    'success': False,
                    'message': f'Could not find {target_object} in scene',
                    'searched_for': target_object
                }

        except Exception as e:
            return {
                'success': False,
                'message': f'Detection error: {str(e)}',
                'target': target_object
            }

    def resolve_location_to_coordinates(self, location_name: str):
        """Resolve location name to coordinates"""
        # Predefined locations in simulation
        location_map = {
            'kitchen': np.array([3.0, 1.0, 0.0]),
            'living room': np.array([0.0, 0.0, 0.0]),
            'bedroom': np.array([-2.0, 2.0, 0.0]),
            'office': np.array([1.0, -2.0, 0.0]),
            'dining room': np.array([2.0, 0.0, 0.0]),
            'entrance': np.array([0.0, 3.0, 0.0]),
            'center': np.array([0.0, 0.0, 0.0])
        }

        location_key = location_name.lower().strip()

        # Handle relative positions
        if 'near' in location_key:
            # For example: "near kitchen"
            for loc_name, coords in location_map.items():
                if loc_name in location_key:
                    # Return nearby coordinates
                    offset = np.random.uniform(-1, 1, size=2)
                    return np.array([coords[0] + offset[0], coords[1] + offset[1], coords[2]])

        return location_map.get(location_key)

    def find_object_by_name(self, name: str):
        """Find a single object by name"""
        for obj_name, obj in self.scene_objects.items():
            if name.lower() in obj_name.lower():
                return obj
        return None

    def find_objects_by_name(self, name: str):
        """Find all objects matching name"""
        matches = []
        for obj_name, obj in self.scene_objects.items():
            if name.lower() in obj_name.lower():
                matches.append(obj)
        return matches

    def simulate_manipulation(self, target_object, parameters):
        """Simulate manipulation action"""
        operation = parameters.get('operation', 'grasp')

        if operation in ['grasp', 'pick_up', 'take', 'grab']:
            # Simulate grasping
            return {
                'success': True,
                'message': f'Successfully grasped {target_object.name}'
            }
        elif operation in ['place', 'put', 'set_down']:
            # Simulate placing
            return {
                'success': True,
                'message': f'Successfully placed object'
            }
        else:
            return {
                'success': False,
                'message': f'Unknown manipulation operation: {operation}'
            }
```

### Voice Command Validation and Safety

#### Command Validation Pipeline

```python
class VoiceCommandValidator:
    def __init__(self):
        self.safety_keywords = [
            'shutdown', 'power off', 'terminate', 'destroy', 'damage',
            'break', 'harm', 'unsafe', 'dangerous', 'emergency'
        ]

        self.privileged_commands = [
            'system', 'admin', 'root', 'superuser', 'privileged'
        ]

    def validate_command(self, command_data: dict):
        """Validate voice command for safety and appropriateness"""
        intent = command_data.get('intent', {})
        command_text = command_data.get('command', '').lower()

        # Check for safety violations
        safety_violations = self.check_safety_violations(command_text)
        if safety_violations:
            return {
                'valid': False,
                'reason': 'Safety violation detected',
                'violations': safety_violations,
                'suggested_alternatives': self.get_safe_alternatives(intent)
            }

        # Check for privileged commands
        if self.contains_privileged_keywords(command_text):
            return {
                'valid': False,
                'reason': 'Privileged command requires authentication',
                'requires_auth': True
            }

        # Check command structure validity
        if not self.is_structurally_valid(intent):
            return {
                'valid': False,
                'reason': 'Command structure is invalid',
                'suggestions': self.get_structure_suggestions(intent)
            }

        # All checks passed
        return {
            'valid': True,
            'reason': 'Command is valid and safe',
            'confidence': command_data.get('confidence', 0.8)
        }

    def check_safety_violations(self, command_text: str):
        """Check for safety-related violations"""
        violations = []

        for keyword in self.safety_keywords:
            if keyword in command_text:
                violations.append({
                    'keyword': keyword,
                    'type': 'safety',
                    'severity': 'high' if keyword in ['harm', 'dangerous', 'emergency'] else 'medium'
                })

        return violations

    def contains_privileged_keywords(self, command_text: str):
        """Check if command contains privileged keywords"""
        return any(keyword in command_text for keyword in self.privileged_commands)

    def is_structurally_valid(self, intent: dict):
        """Check if intent has valid structure"""
        required_fields = ['action', 'target']
        return all(field in intent for field in required_fields)

    def get_safe_alternatives(self, intent: dict):
        """Get safe alternatives to unsafe commands"""
        action = intent.get('action', 'unknown')

        alternatives = {
            'shutdown': ['stop', 'pause', 'power down safely'],
            'damage': ['inspect', 'examine', 'analyze'],
            'harm': ['assist', 'help', 'support'],
            'unsafe': ['safe', 'secure', 'stable']
        }

        return alternatives.get(action, [])

    def get_structure_suggestions(self, intent: dict):
        """Get suggestions for structurally invalid commands"""
        suggestions = []

        if 'action' not in intent:
            suggestions.append('Specify an action (navigate, manipulate, detect, etc.)')

        if 'target' not in intent:
            suggestions.append('Specify a target for the action')

        return suggestions
```

The voice-to-action pipeline is essential for natural human-robot interaction, enabling robots to understand and execute spoken commands. The implementation includes speech recognition, natural language processing, intent extraction, action planning, and execution with safety validation.