---
sidebar_label: "Multimodal Integration (Speech, Vision, Motion)"
---

# Multimodal Integration (Speech, Vision, Motion)

## Introduction

Multimodal integration is the cornerstone of natural human-robot interaction in the Autonomous Humanoid system. This chapter explores the integration of speech processing, computer vision, and motion control to create seamless, intuitive interactions that feel natural to human users. The system must coordinate multiple sensory and actuation modalities to understand user intent, perceive the environment, and execute appropriate responses.

## Multimodal Architecture

### System Overview

The multimodal integration system coordinates multiple subsystems:

```
Speech Input → Natural Language Processing → Task Planning
     ↓              ↓                         ↓
Vision Input → Perception Fusion ←→ Motion Control ←→ Action Execution
     ↑              ↑                         ↑
Environment ←→ State Estimation ←→ Robot Control ←→ Physical World
```

### Key Integration Challenges

1. **Temporal Synchronization**: Aligning inputs and outputs across different modalities
2. **Semantic Integration**: Combining information from different sensory channels
3. **Attention Management**: Focusing on relevant sensory inputs at appropriate times
4. **Context Maintenance**: Preserving context across multiple interaction modalities
5. **Conflict Resolution**: Handling conflicting information from different modalities

## Multimodal Fusion Framework

### Data Fusion Architecture

```python
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import queue

class ModalityType(Enum):
    SPEECH = "speech"
    VISION = "vision"
    MOTION = "motion"
    TACTILE = "tactile"
    AUDIO = "audio"

@dataclass
class MultimodalInput:
    modality: ModalityType
    data: Any
    timestamp: float
    confidence: float
    source_id: str

@dataclass
class FusedEvent:
    event_type: str  # "command", "question", "response", "action"
    semantic_content: str
    relevant_inputs: List[MultimodalInput]
    context: Dict[str, Any]
    confidence: float
    timestamp: float

class MultimodalFusionEngine:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.fusion_rules = self._initialize_fusion_rules()
        self.context_manager = ContextManager()
        self.event_handlers = {}

        # Modality-specific processors
        self.speech_processor = SpeechProcessor()
        self.vision_processor = VisionProcessor()
        self.motion_processor = MotionProcessor()

        # Fusion thread
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()

    def _initialize_fusion_rules(self) -> Dict[str, Callable]:
        """Initialize rules for combining modalities"""
        return {
            'follow_me': self._fuse_follow_me,
            'grasp_object': self._fuse_grasp_object,
            'answer_question': self._fuse_answer_question,
            'greet_user': self._fuse_greet_user
        }

    def add_input(self, modality: ModalityType, data: Any, confidence: float = 1.0):
        """Add input from a specific modality"""
        input_event = MultimodalInput(
            modality=modality,
            data=data,
            timestamp=time.time(),
            confidence=confidence,
            source_id=f"{modality.value}_{int(time.time()*1000)}"
        )
        self.input_queue.put(input_event)

    def _fusion_loop(self):
        """Main fusion processing loop"""
        recent_inputs = []
        fusion_window = 2.0  # seconds to consider for fusion

        while True:
            try:
                # Collect inputs within fusion window
                while not self.input_queue.empty():
                    input_event = self.input_queue.get_nowait()
                    recent_inputs.append(input_event)

                # Remove old inputs
                current_time = time.time()
                recent_inputs = [
                    inp for inp in recent_inputs
                    if current_time - inp.timestamp <= fusion_window
                ]

                # Perform fusion if we have multiple modalities
                if len(recent_inputs) >= 2:
                    fused_event = self._perform_fusion(recent_inputs)
                    if fused_event:
                        self._handle_fused_event(fused_event)

                time.sleep(0.01)  # 100Hz processing

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Fusion loop error: {e}")

    def _perform_fusion(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Perform fusion of multiple modalities"""
        # Group inputs by temporal proximity
        grouped_inputs = self._group_temporally(inputs)

        for group in grouped_inputs:
            # Try to identify fusion pattern
            event = self._identify_fusion_pattern(group)
            if event:
                return event

        return None

    def _group_temporally(self, inputs: List[MultimodalInput]) -> List[List[MultimodalInput]]:
        """Group inputs that occurred close in time"""
        if not inputs:
            return []

        groups = []
        current_group = [inputs[0]]

        for inp in inputs[1:]:
            time_diff = abs(inp.timestamp - current_group[-1].timestamp)
            if time_diff < 0.5:  # 500ms window
                current_group.append(inp)
            else:
                if len(current_group) > 1:
                    groups.append(current_group)
                current_group = [inp]

        if len(current_group) > 1:
            groups.append(current_group)

        return groups

    def _identify_fusion_pattern(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Identify fusion pattern and create fused event"""
        # Create modality map
        modality_map = {}
        for inp in inputs:
            modality_map[inp.modality] = inp

        # Check for specific fusion patterns
        if ModalityType.SPEECH in modality_map and ModalityType.VISION in modality_map:
            speech_input = modality_map[ModalityType.SPEECH]
            vision_input = modality_map[ModalityType.VISION]

            # Process speech and vision together
            command = speech_input.data.get('command', '').lower()
            objects = vision_input.data.get('objects', [])

            if 'grasp' in command or 'pick up' in command:
                target_object = self._find_target_object(command, objects)
                if target_object:
                    return FusedEvent(
                        event_type='grasp_object',
                        semantic_content=f'Grasp {target_object["name"]}',
                        relevant_inputs=inputs,
                        context={
                            'target_object': target_object,
                            'command': command
                        },
                        confidence=0.8,
                        timestamp=time.time()
                    )

        return None

    def _find_target_object(self, command: str, objects: List[Dict]) -> Optional[Dict]:
        """Find object in vision data that matches speech command"""
        for obj in objects:
            if obj['name'].lower() in command or obj['class'].lower() in command:
                return obj
        return None

    def _handle_fused_event(self, event: FusedEvent):
        """Handle fused event by passing to appropriate handler"""
        handler_name = f"handle_{event.event_type.replace('-', '_')}"
        handler = getattr(self, handler_name, None)

        if handler:
            handler(event)
        else:
            print(f"No handler for event type: {event.event_type}")

    def handle_grasp_object(self, event: FusedEvent):
        """Handle grasp object event"""
        print(f"Handling grasp object: {event.context['target_object']['name']}")
        # Trigger manipulation system

    def _fuse_follow_me(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Fuse inputs for follow-me command"""
        # Implementation for follow-me fusion
        return None

    def _fuse_grasp_object(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Fuse inputs for grasp object command"""
        # Implementation for grasp object fusion
        return None

    def _fuse_answer_question(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Fuse inputs for answering questions"""
        # Implementation for question answering fusion
        return None

    def _fuse_greet_user(self, inputs: List[MultimodalInput]) -> Optional[FusedEvent]:
        """Fuse inputs for greeting user"""
        # Implementation for greeting fusion
        return None
```

## Speech and Vision Integration

### Joint Attention System

The system coordinates speech and vision to understand user focus:

```python
class JointAttentionSystem:
    def __init__(self):
        self.attention_history = []
        self.saliency_maps = {}
        self.gaze_prediction_model = self._load_gaze_model()
        self.speech_context = SpeechContext()

    def process_speech_vision_input(self, speech_data: Dict, vision_data: Dict) -> Dict:
        """Process combined speech and vision input to determine attention"""
        # Extract entities from speech
        speech_entities = self.speech_context.extract_entities(speech_data['text'])

        # Identify salient objects in vision
        salient_objects = self._find_salient_objects(vision_data['image'], vision_data['objects'])

        # Determine referent based on spatial and linguistic context
        referent = self._resolve_reference(speech_entities, salient_objects, vision_data)

        # Update attention history
        attention_event = {
            'timestamp': time.time(),
            'speech_entities': speech_entities,
            'salient_objects': salient_objects,
            'resolved_referent': referent,
            'confidence': 0.9
        }
        self.attention_history.append(attention_event)

        return {
            'referent': referent,
            'attention_map': self._create_attention_map(speech_entities, salient_objects),
            'context': self._update_context(referent)
        }

    def _find_salient_objects(self, image: Any, objects: List[Dict]) -> List[Dict]:
        """Find visually salient objects in the scene"""
        # Calculate saliency based on size, location, motion, color contrast
        salient_objects = []

        for obj in objects:
            saliency_score = self._calculate_saliency(obj, image)
            if saliency_score > 0.3:  # Threshold for salience
                obj['saliency'] = saliency_score
                salient_objects.append(obj)

        # Sort by saliency
        salient_objects.sort(key=lambda x: x['saliency'], reverse=True)
        return salient_objects

    def _calculate_saliency(self, obj: Dict, image: Any) -> float:
        """Calculate visual saliency of an object"""
        # Size saliency: larger objects are more salient
        size_saliency = min(1.0, obj['area'] / 10000)  # Normalize by image area

        # Position saliency: objects near center are more salient
        center_x, center_y = obj['center']
        img_h, img_w = image.shape[:2] if hasattr(image, 'shape') else (480, 640)
        center_distance = ((center_x - img_w/2)**2 + (center_y - img_h/2)**2)**0.5
        max_distance = ((img_w/2)**2 + (img_h/2)**2)**0.5
        position_saliency = 1.0 - (center_distance / max_distance)

        # Color contrast saliency: TODO - implement color contrast calculation
        color_saliency = 0.5  # Placeholder

        # Combine scores
        total_saliency = (0.4 * size_saliency + 0.4 * position_saliency + 0.2 * color_saliency)
        return total_saliency

    def _resolve_reference(self, speech_entities: List[str], salient_objects: List[Dict],
                         vision_data: Dict) -> Optional[Dict]:
        """Resolve linguistic reference to visual object"""
        if not speech_entities or not salient_objects:
            return None

        # Find best match between speech entities and visual objects
        best_match = None
        best_score = 0.0

        for entity in speech_entities:
            for obj in salient_objects:
                score = self._calculate_reference_score(entity, obj)
                if score > best_score:
                    best_score = score
                    best_match = obj

        return best_match if best_score > 0.5 else None

    def _calculate_reference_score(self, entity: str, obj: Dict) -> float:
        """Calculate score for matching entity to object"""
        # Name matching
        name_score = 0.0
        if entity.lower() in obj.get('name', '').lower() or entity.lower() in obj.get('class', '').lower():
            name_score = 1.0

        # Category matching
        category_score = 0.0
        entity_category = self._get_entity_category(entity)
        obj_category = obj.get('class', '')
        if entity_category and entity_category.lower() in obj_category.lower():
            category_score = 0.7

        # Contextual score based on saliency
        contextual_score = obj.get('saliency', 0.5)

        # Weighted combination
        total_score = (name_score * 0.5 + category_score * 0.3 + contextual_score * 0.2)
        return total_score

    def _get_entity_category(self, entity: str) -> str:
        """Get category for an entity"""
        # Simple mapping - in practice, use more sophisticated NLP
        category_map = {
            'bottle': 'drink',
            'cup': 'drink',
            'glass': 'drink',
            'book': 'read',
            'phone': 'device',
            'computer': 'device',
            'chair': 'furniture',
            'table': 'furniture'
        }
        return category_map.get(entity.lower(), entity)

    def _create_attention_map(self, speech_entities: List[str], salient_objects: List[Dict]) -> Dict:
        """Create attention map showing focus areas"""
        attention_map = {
            'speech_focus': speech_entities,
            'visual_focus': [obj['name'] for obj in salient_objects[:3]],  # Top 3 salient objects
            'resolved_reference': self.attention_history[-1]['resolved_referent']['name'] if self.attention_history else None
        }
        return attention_map

    def _update_context(self, referent: Optional[Dict]) -> Dict:
        """Update interaction context"""
        context = {
            'current_referent': referent,
            'attention_history': self.attention_history[-5:],  # Last 5 attention events
            'focus_object': referent['name'] if referent else None
        }
        return context

class SpeechContext:
    def __init__(self):
        # Load NLP models for entity extraction
        self.entity_keywords = {
            'object': ['the', 'that', 'this', 'bottle', 'cup', 'book', 'phone'],
            'action': ['grasp', 'pick', 'take', 'bring', 'give'],
            'location': ['there', 'here', 'on', 'in', 'under', 'next', 'near']
        }

    def extract_entities(self, text: str) -> List[str]:
        """Extract relevant entities from speech"""
        text_lower = text.lower()
        entities = []

        for category, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append(keyword)

        # Remove duplicates while preserving order
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities:
                unique_entities.append(entity)

        return unique_entities
```

## Context Management and State Tracking

### Multimodal Context System

Maintaining coherent context across modalities:

```python
from collections import deque
import json

class MultimodalContextManager:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.context_history = deque(maxlen=max_history)
        self.current_context = {
            'entities': {},
            'spatial_relations': {},
            'temporal_context': {},
            'user_intention': None,
            'task_state': None,
            'attention_focus': None
        }

    def update_context(self, modality_data: Dict, event_type: str) -> Dict:
        """Update context with new information from modalities"""
        timestamp = time.time()

        # Update entities
        if 'entities' in modality_data:
            for entity, info in modality_data['entities'].items():
                self.current_context['entities'][entity] = {
                    'info': info,
                    'last_seen': timestamp,
                    'modality': modality_data.get('modality', 'unknown')
                }

        # Update spatial relations
        if 'spatial_relations' in modality_data:
            self.current_context['spatial_relations'].update(modality_data['spatial_relations'])

        # Update temporal context
        self.current_context['temporal_context'][event_type] = timestamp

        # Update attention focus
        if 'attention_focus' in modality_data:
            self.current_context['attention_focus'] = modality_data['attention_focus']

        # Add to history
        context_snapshot = self.current_context.copy()
        context_snapshot['timestamp'] = timestamp
        context_snapshot['event_type'] = event_type
        self.context_history.append(context_snapshot)

        return self.current_context

    def get_relevant_context(self, query: str = None) -> Dict:
        """Get relevant context for current interaction"""
        if query:
            # Filter context based on query relevance
            return self._filter_context_by_query(query)
        else:
            return self.current_context

    def _filter_context_by_query(self, query: str) -> Dict:
        """Filter context based on query relevance"""
        filtered_context = {
            'relevant_entities': {},
            'spatial_context': {},
            'recent_events': []
        }

        # Find entities mentioned in query
        query_lower = query.lower()
        for entity, info in self.current_context['entities'].items():
            if entity.lower() in query_lower:
                filtered_context['relevant_entities'][entity] = info

        # Include spatial relations for relevant entities
        for rel, info in self.current_context['spatial_relations'].items():
            if any(entity in rel for entity in filtered_context['relevant_entities']):
                filtered_context['spatial_context'][rel] = info

        # Include recent events
        recent_threshold = time.time() - 30  # Last 30 seconds
        filtered_context['recent_events'] = [
            event for event in list(self.context_history)
            if event['timestamp'] > recent_threshold
        ]

        return filtered_context

    def resolve_coreferences(self, text: str) -> str:
        """Resolve pronouns and coreferences in text using context"""
        # Replace pronouns with resolved entities
        resolved_text = text

        # Common pronouns and their potential resolutions
        pronouns = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they']
        for pronoun in pronouns:
            if pronoun in resolved_text.lower():
                # Find most recently mentioned entity of appropriate type
                resolved_entity = self._resolve_pronoun(pronoun)
                if resolved_entity:
                    resolved_text = resolved_text.replace(pronoun, resolved_entity)

        return resolved_text

    def _resolve_pronoun(self, pronoun: str) -> str:
        """Resolve a pronoun to a specific entity"""
        # For 'it', find the most recently mentioned object
        if pronoun.lower() == 'it':
            for event in reversed(self.context_history):
                if 'resolved_reference' in event and event['resolved_reference']:
                    return event['resolved_reference']['name']

        # For 'this'/'that', find entity in attention focus
        elif pronoun.lower() in ['this', 'that']:
            if self.current_context['attention_focus']:
                return self.current_context['attention_focus']['name']

        return pronoun  # Return original if can't resolve

    def infer_user_intention(self) -> str:
        """Infer user intention from context"""
        # Analyze recent events and context to infer intention
        recent_events = list(self.context_history)[-5:]  # Last 5 events

        if not recent_events:
            return "unknown"

        # Look for patterns in events
        actions = [event.get('event_type', '') for event in recent_events]
        entities = []
        for event in recent_events:
            if 'resolved_reference' in event:
                entities.append(event['resolved_reference'])

        # Simple intention inference
        if 'grasp_object' in actions:
            return "object_interaction"
        elif 'follow_me' in actions:
            return "navigation"
        elif 'answer_question' in actions:
            return "information_request"
        else:
            return "exploration"
```

## Integration with Action Planning

### Coordinated Action Execution

Coordinating multimodal inputs into coherent actions:

```python
class CoordinatedActionPlanner:
    def __init__(self):
        self.fusion_engine = MultimodalFusionEngine()
        self.context_manager = MultimodalContextManager()
        self.action_library = self._initialize_action_library()

    def _initialize_action_library(self) -> Dict:
        """Initialize library of coordinated actions"""
        return {
            'grasp_object': {
                'preconditions': ['object_detected', 'arm_free'],
                'modalities': ['speech', 'vision'],
                'execution_plan': self._execute_grasp_object
            },
            'follow_user': {
                'preconditions': ['user_detected', 'navigation_clear'],
                'modalities': ['vision', 'audio'],
                'execution_plan': self._execute_follow_user
            },
            'answer_question': {
                'preconditions': ['question_parsed'],
                'modalities': ['speech'],
                'execution_plan': self._execute_answer_question
            },
            'greet_user': {
                'preconditions': ['user_approaching'],
                'modalities': ['vision', 'speech'],
                'execution_plan': self._execute_greet_user
            }
        }

    def process_multimodal_command(self, inputs: Dict) -> bool:
        """Process multimodal command and execute coordinated action"""
        try:
            # Update context with new inputs
            context = self.context_manager.update_context(inputs, 'command_received')

            # Perform multimodal fusion
            for modality, data in inputs.items():
                if modality in [ModalityType.SPEECH, ModalityType.VISION, ModalityType.MOTION]:
                    self.fusion_engine.add_input(ModalityType(modality), data)

            # Wait for fusion result (simplified - in practice would be async)
            time.sleep(0.1)

            # Determine appropriate action based on fused information
            action_type = self._determine_action_type(inputs, context)

            if action_type in self.action_library:
                action_def = self.action_library[action_type]

                # Check preconditions
                if self._check_preconditions(action_def['preconditions'], context):
                    # Execute coordinated action
                    success = action_def['execution_plan'](inputs, context)
                    return success

            return False

        except Exception as e:
            print(f"Error in multimodal command processing: {e}")
            return False

    def _determine_action_type(self, inputs: Dict, context: Dict) -> str:
        """Determine appropriate action type based on inputs and context"""
        # Analyze speech input
        if 'speech' in inputs:
            speech_text = inputs['speech'].get('text', '').lower()
            if any(word in speech_text for word in ['grasp', 'pick', 'take', 'get']):
                return 'grasp_object'
            elif any(word in speech_text for word in ['follow', 'come', 'with']):
                return 'follow_user'
            elif '?' in speech_text or any(word in speech_text for word in ['what', 'where', 'how', 'when']):
                return 'answer_question'

        # Analyze vision input
        if 'vision' in inputs:
            objects = inputs['vision'].get('objects', [])
            user_detected = any(obj.get('class') == 'person' for obj in objects)
            if user_detected and context.get('temporal_context', {}).get('greet_user', 0) > time.time() - 60:
                return 'greet_user'

        return 'unknown'

    def _check_preconditions(self, preconditions: List[str], context: Dict) -> bool:
        """Check if action preconditions are satisfied"""
        for precondition in preconditions:
            if not self._evaluate_precondition(precondition, context):
                return False
        return True

    def _evaluate_precondition(self, precondition: str, context: Dict) -> bool:
        """Evaluate a specific precondition"""
        if precondition == 'object_detected':
            return len(context['entities']) > 0
        elif precondition == 'arm_free':
            # Check if arm is not currently grasping
            return context.get('task_state') != 'grasping'
        elif precondition == 'user_detected':
            return any(obj.get('class') == 'person' for obj in context['entities'].values())
        elif precondition == 'navigation_clear':
            # Check if path is clear for navigation
            return True  # Simplified
        elif precondition == 'question_parsed':
            return 'question' in context.get('temporal_context', {})

        return True  # Default to true for unknown preconditions

    def _execute_grasp_object(self, inputs: Dict, context: Dict) -> bool:
        """Execute coordinated grasp object action"""
        print("Executing coordinated grasp object action")

        # 1. Look at the object
        if 'vision' in inputs:
            target_object = inputs['vision'].get('objects', [{}])[0]  # First object
            self._direct_attention_to_object(target_object)

        # 2. Confirm grasp intent
        if 'speech' in inputs:
            confirmation = self._confirm_action(inputs['speech']['text'])
            if not confirmation:
                return False

        # 3. Execute grasp
        success = self._perform_grasp(context)
        return success

    def _execute_follow_user(self, inputs: Dict, context: Dict) -> bool:
        """Execute coordinated follow user action"""
        print("Executing coordinated follow user action")

        # 1. Identify user to follow
        user_location = self._identify_user_location(inputs)
        if not user_location:
            return False

        # 2. Navigate to follow user
        success = self._navigate_and_follow(user_location)
        return success

    def _execute_answer_question(self, inputs: Dict, context: Dict) -> bool:
        """Execute coordinated answer question action"""
        print("Executing coordinated answer question action")

        # 1. Process question
        question = inputs['speech']['text']
        answer = self._process_question(question, context)

        # 2. Formulate response
        response = self._generate_response(answer)

        # 3. Deliver response
        success = self._speak_response(response)
        return success

    def _execute_greet_user(self, inputs: Dict, context: Dict) -> bool:
        """Execute coordinated greet user action"""
        print("Executing coordinated greet user action")

        # 1. Make eye contact
        self._make_eye_contact()

        # 2. Deliver greeting
        greeting = self._generate_greeting()

        # 3. Execute greeting
        success = self._speak_response(greeting)
        return success

    def _direct_attention_to_object(self, obj: Dict):
        """Direct robot attention to an object"""
        print(f"Directing attention to {obj.get('name', 'object')}")

    def _confirm_action(self, speech_text: str) -> bool:
        """Confirm action with user"""
        # In practice, would ask user to confirm
        return True

    def _perform_grasp(self, context: Dict) -> bool:
        """Perform the actual grasp"""
        print("Performing grasp")
        return True

    def _identify_user_location(self, inputs: Dict) -> Dict:
        """Identify user location from inputs"""
        if 'vision' in inputs:
            for obj in inputs['vision'].get('objects', []):
                if obj.get('class') == 'person':
                    return obj
        return {}

    def _navigate_and_follow(self, user_location: Dict) -> bool:
        """Navigate to follow the user"""
        print("Navigating to follow user")
        return True

    def _process_question(self, question: str, context: Dict) -> str:
        """Process a question and generate answer"""
        return f"I understand you're asking: {question}. I'm still learning to answer questions."

    def _generate_response(self, answer: str) -> str:
        """Generate spoken response"""
        return answer

    def _speak_response(self, response: str) -> bool:
        """Speak the response"""
        print(f"Speaking: {response}")
        return True

    def _make_eye_contact(self):
        """Make eye contact with user"""
        print("Making eye contact")