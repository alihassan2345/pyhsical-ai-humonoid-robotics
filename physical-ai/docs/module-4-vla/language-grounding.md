---
sidebar_position: 1
---

# Language Grounding for Robotics

## Connecting Words to Actions

Language grounding is the process of connecting natural language to physical actions and perceptions in the real world. This module covers how to implement language grounding for robotics applications using Isaac and ROS 2.

### Understanding Language Grounding

Language grounding in robotics involves:

- **Perception Grounding**: Connecting language descriptions to sensor data
- **Action Grounding**: Connecting language commands to motor actions
- **Spatial Grounding**: Connecting language references to spatial locations
- **Temporal Grounding**: Connecting language to temporal sequences

### Language Grounding Architecture

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import openai
from transformers import AutoTokenizer, AutoModel
import torch

class LanguageGroundingNode(Node):
    def __init__(self):
        super().__init__('language_grounding_node')

        # Initialize language model
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.language_model = AutoModel.from_pretrained('bert-base-uncased')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Subscribe to various data streams
        self.voice_sub = self.create_subscription(
            String,
            '/recognized_text',
            self.voice_callback,
            10
        )
        self.image_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/object_detections',
            self.detection_callback,
            10
        )

        # Publishers for grounded results
        self.command_pub = self.create_publisher(
            String,
            '/grounded_commands',
            10
        )
        self.target_pub = self.create_publisher(
            PoseStamped,
            '/grounded_target',
            10
        )

        # Store current context
        self.current_image = None
        self.current_detections = None
        self.context_history = []

        self.get_logger().info('Language grounding node initialized')

    def voice_callback(self, msg):
        """Process voice command and ground it to robot actions"""
        try:
            command = msg.data
            self.get_logger().info(f'Processing command: {command}')

            # Ground the command using language model
            grounded_action = self.ground_language_command(command)

            if grounded_action:
                # Publish the grounded command
                cmd_msg = String()
                cmd_msg.data = grounded_action
                self.command_pub.publish(cmd_msg)

                self.get_logger().info(f'Published grounded command: {grounded_action}')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')

    def image_callback(self, msg):
        """Process image for visual grounding"""
        try:
            # Convert ROS image to OpenCV format
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detection_callback(self, msg):
        """Process object detections for grounding"""
        try:
            self.current_detections = msg.detections

        except Exception as e:
            self.get_logger().error(f'Error processing detections: {e}')

    def ground_language_command(self, command):
        """Ground language command to robot actions"""
        try:
            # Tokenize the command
            tokens = self.tokenizer.tokenize(command)
            token_ids = self.tokenizer.encode(command, return_tensors='pt')

            # Get language embeddings
            with torch.no_grad():
                language_embeddings = self.language_model(token_ids)[0][:, 0, :]  # CLS token

            # Parse the command to identify key components
            parsed_command = self.parse_command(command)

            if parsed_command['action'] == 'navigate':
                target_location = self.find_location_in_context(parsed_command['target'])
                if target_location:
                    return f"navigate_to:{target_location}"
                else:
                    return f"navigate_to:{parsed_command['target']}"  # Use verbal target

            elif parsed_command['action'] == 'manipulate':
                target_object = self.find_object_in_context(parsed_command['object'])
                operation = parsed_command['operation']

                if target_object:
                    return f"manipulate:{operation}:{target_object['name']}:{target_object['pose']}"
                else:
                    return f"search_for:{parsed_command['object']}"

            elif parsed_command['action'] == 'detect':
                target_object = parsed_command['object']
                return f"detect:{target_object}"

            else:
                return f"unknown:{command}"

        except Exception as e:
            self.get_logger().error(f'Error in language grounding: {e}')
            return f"error:{str(e)}"

    def parse_command(self, command):
        """Parse command to extract action, target, and parameters"""
        command_lower = command.lower()

        # Define action patterns
        patterns = {
            'navigate': [
                r'navigate to (.+)',
                r'go to (.+)',
                r'move to (.+)',
                r'go to the (.+)'
            ],
            'manipulate': [
                r'pick up (.+)',
                r'grab (.+)',
                r'get (.+)',
                r'take (.+)',
                r'hold (.+)',
                r'put (.+) on (.+)',
                r'place (.+) on (.+)'
            ],
            'detect': [
                r'find (.+)',
                r'look for (.+)',
                r'spot (.+)',
                r'detect (.+)'
            ]
        }

        for action, regex_patterns in patterns.items():
            import re
            for pattern in regex_patterns:
                match = re.search(pattern, command_lower)
                if match:
                    groups = match.groups()

                    if action == 'navigate':
                        return {
                            'action': 'navigate',
                            'target': groups[0].strip()
                        }
                    elif action == 'manipulate':
                        if len(groups) == 1:
                            # Simple manipulation (e.g., pick up object)
                            return {
                                'action': 'manipulate',
                                'object': groups[0].strip(),
                                'operation': self.infer_operation(command)
                            }
                        elif len(groups) == 2:
                            # Complex manipulation (e.g., place object on target)
                            return {
                                'action': 'manipulate',
                                'object': groups[0].strip(),
                                'target': groups[1].strip(),
                                'operation': 'place'
                            }
                    elif action == 'detect':
                        return {
                            'action': 'detect',
                            'object': groups[0].strip()
                        }

        # If no pattern matches, return unknown action
        return {
            'action': 'unknown',
            'raw_command': command
        }

    def infer_operation(self, command):
        """Infer manipulation operation from command"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['pick', 'take', 'grab']):
            return 'grasp'
        elif any(word in command_lower for word in ['put', 'place', 'drop']):
            return 'place'
        elif any(word in command_lower for word in ['move', 'push', 'pull']):
            return 'move'
        else:
            return 'manipulate'

    def find_location_in_context(self, location_name):
        """Find location in current context (map, landmarks, etc.)"""
        # In a real implementation, this would look up the location in a map
        # or landmark database. For simulation, we'll use a simple lookup.

        location_map = {
            'kitchen': [3.0, 1.0, 0.0],
            'living room': [0.0, 0.0, 0.0],
            'bedroom': [-2.0, 2.0, 0.0],
            'office': [1.0, -2.0, 0.0],
            'dining room': [2.0, 0.0, 0.0]
        }

        location_key = location_name.lower().strip()

        if location_key in location_map:
            x, y, z = location_map[location_key]

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = z
            pose_msg.pose.orientation.w = 1.0

            return pose_msg
        else:
            # Location not found in map, return None
            return None

    def find_object_in_context(self, object_name):
        """Find object in current sensor context"""
        if not self.current_detections:
            return None

        # Look for the object in current detections
        for detection in self.current_detections:
            if object_name.lower() in detection.results[0].id.lower():
                # Create a pose based on detection (simplified)
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "camera_rgb_optical_frame"  # or base_link

                # Convert bounding box center to 3D position (simplified)
                # In reality, this would require depth information or PnP
                pose_msg.pose.position.x = detection.bbox.center.x
                pose_msg.pose.position.y = detection.bbox.center.y
                pose_msg.pose.position.z = 1.0  # Default depth

                return {
                    'name': detection.results[0].id,
                    'pose': pose_msg,
                    'confidence': detection.results[0].score
                }

        # Object not found in current detections
        return None

    def compute_visual_language_similarity(self, text, visual_features):
        """Compute similarity between text and visual features for grounding"""
        # Tokenize text
        text_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.language_model(**text_tokens)[0][:, 0, :]  # CLS token

        # Compute similarity (cosine similarity)
        similarity = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1),
            visual_features.unsqueeze(0),
            dim=2
        )

        return similarity


class MultiModalGroundingNode(LanguageGroundingNode):
    """Extended language grounding with multi-modal fusion"""

    def __init__(self):
        super().__init__()

        # Additional modalities
        self.spatial_reasoning = SpatialReasoningModule()
        self.temporal_grounding = TemporalGroundingModule()

    def ground_multimodal_command(self, command, visual_data, audio_data=None):
        """Ground command using multiple modalities"""
        # Parse command linguistically
        linguistic_structure = self.parse_command(command)

        # Ground to visual context
        visual_groundings = self.ground_to_visual_context(linguistic_structure, visual_data)

        # Ground to spatial context
        spatial_groundings = self.spatial_reasoning.ground_to_space(
            linguistic_structure, self.get_robot_pose()
        )

        # Ground to temporal context
        temporal_constraints = self.temporal_grounding.extract_temporal_constraints(command)

        # Fuse all modalities
        fused_grounding = self.fuse_modalities(
            linguistic_structure,
            visual_groundings,
            spatial_groundings,
            temporal_constraints
        )

        return fused_grounding

    def ground_to_visual_context(self, linguistic_structure, visual_data):
        """Ground linguistic elements to visual context"""
        visual_groundings = {}

        if linguistic_structure['action'] == 'detect':
            target_object = linguistic_structure['object']
            # Find object in visual data
            object_instances = self.find_object_instances(target_object, visual_data)
            visual_groundings['objects'] = object_instances

        elif linguistic_structure['action'] == 'navigate':
            target_location = linguistic_structure['target']
            # Find spatial landmarks that match location description
            landmarks = self.find_landmarks(target_location, visual_data)
            visual_groundings['landmarks'] = landmarks

        return visual_groundings

    def find_object_instances(self, object_name, visual_data):
        """Find instances of an object in visual data"""
        # This would use object detection, segmentation, etc.
        # to find all instances of an object in the visual scene
        pass

    def find_landmarks(self, location_description, visual_data):
        """Find landmarks that match location description"""
        # This would identify visual landmarks (e.g., doors, furniture)
        # that match the location description
        pass

    def fuse_modalities(self, linguistic, visual, spatial, temporal):
        """Fuse information from multiple modalities"""
        # Combine all grounded information into a coherent action plan
        fused_result = {
            'action': linguistic['action'],
            'targets': [],
            'constraints': {},
            'confidence': 0.0
        }

        # Add visual targets
        if 'objects' in visual:
            fused_result['targets'].extend(visual['objects'])

        # Add spatial constraints
        fused_result['constraints'].update(spatial)

        # Add temporal constraints
        fused_result['constraints'].update(temporal)

        # Compute overall confidence
        fused_result['confidence'] = self.compute_fusion_confidence(
            linguistic, visual, spatial, temporal
        )

        return fused_result

    def compute_fusion_confidence(self, linguistic, visual, spatial, temporal):
        """Compute confidence in the fused grounding result"""
        # Weight different modalities based on reliability
        weights = {
            'linguistic': 0.3,
            'visual': 0.4,
            'spatial': 0.2,
            'temporal': 0.1
        }

        # Compute confidence for each modality
        ling_conf = self.compute_linguistic_confidence(linguistic)
        vis_conf = self.compute_visual_confidence(visual)
        spat_conf = self.compute_spatial_confidence(spatial)
        temp_conf = self.compute_temporal_confidence(temporal)

        # Weighted average
        total_conf = (
            weights['linguistic'] * ling_conf +
            weights['visual'] * vis_conf +
            weights['spatial'] * spat_conf +
            weights['temporal'] * temp_conf
        )

        return total_conf


class SpatialReasoningModule:
    """Module for spatial reasoning and grounding"""

    def __init__(self):
        # Spatial knowledge base
        self.spatial_kb = SpatialKnowledgeBase()

        # Spatial relation extractor
        self.relation_extractor = SpatialRelationExtractor()

    def ground_to_space(self, linguistic_structure, robot_pose):
        """Ground linguistic structure to spatial context"""
        spatial_constraints = {}

        if linguistic_structure['action'] == 'navigate':
            # Extract spatial relations from location description
            spatial_relations = self.relation_extractor.extract_relations(
                linguistic_structure['target']
            )

            # Ground to spatial knowledge base
            target_pose = self.spatial_kb.resolve_location(
                linguistic_structure['target'],
                robot_pose,
                spatial_relations
            )

            spatial_constraints['target_pose'] = target_pose
            spatial_constraints['path_constraints'] = self.get_path_constraints(target_pose)

        return spatial_constraints

    def get_path_constraints(self, target_pose):
        """Get constraints for navigation path"""
        # This would include forbidden zones, preferred paths, etc.
        return []


class TemporalGroundingModule:
    """Module for temporal grounding"""

    def __init__(self):
        self.temporal_parser = TemporalExpressionParser()

    def extract_temporal_constraints(self, command):
        """Extract temporal constraints from command"""
        return self.temporal_parser.parse(command)


def main(args=None):
    rclpy.init(args=args)

    grounding_node = MultiModalGroundingNode()

    try:
        rclpy.spin(grounding_node)
    except KeyboardInterrupt:
        grounding_node.get_logger().info('Language grounding node stopped by user')
    finally:
        grounding_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Advanced Language Grounding Techniques

#### Neural Language Grounding

```python
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import clip

class NeuralLanguageGrounding(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512):
        super().__init__()

        # Visual encoder (CNN-based)
        self.visual_encoder = models.resnet50(pretrained=True)
        self.visual_encoder.fc = nn.Linear(self.visual_encoder.fc.in_features, hidden_dim)

        # Language encoder (BERT-based)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.language_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.lang_projection = nn.Linear(self.language_encoder.config.hidden_size, hidden_dim)

        # Multimodal fusion
        self.fusion = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)

        # Grounding predictor
        self.grounding_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Binary classification for grounding
        )

        # Object detection grounding
        self.object_grounding = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, texts):
        # Encode visual features
        visual_features = self.visual_encoder(images)  # [batch, hidden_dim]

        # Encode textual features
        encoded_texts = [self.tokenizer(text, return_tensors='pt', padding=True, truncation=True) for text in texts]
        text_embeddings = []

        for encoded_text in encoded_texts:
            with torch.no_grad():
                text_emb = self.language_encoder(**encoded_text)[0][:, 0, :]  # CLS token
                text_embeddings.append(text_emb)

        text_features = torch.stack(text_embeddings, dim=0)  # [batch, seq_len, hidden_size]
        text_features = self.lang_projection(text_features)  # [batch, seq_len, hidden_dim]

        # Multimodal fusion
        fused_features, attention_weights = self.fusion(
            visual_features.unsqueeze(1),  # [batch, 1, hidden_dim]
            text_features,  # [batch, seq_len, hidden_dim]
            text_features  # [batch, seq_len, hidden_dim]
        )

        # Flatten fused features
        fused_features = fused_features.squeeze(1)  # [batch, hidden_dim]

        # Combine visual and fused features
        combined_features = torch.cat([visual_features, fused_features], dim=1)  # [batch, 2*hidden_dim]

        # Predict grounding
        grounding_scores = self.grounding_predictor(combined_features)  # [batch, 1]

        # Object grounding predictions
        object_predictions = self.object_grounding(fused_features)  # [batch, vocab_size]

        return {
            'grounding_scores': torch.sigmoid(grounding_scores),
            'object_predictions': torch.softmax(object_predictions, dim=1),
            'attention_weights': attention_weights
        }
```

#### Vision-Language Models for Grounding

```python
import clip
import torch
import torch.nn.functional as F

class CLIPGrounding(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.device = device

    def ground_command_to_image(self, command, image):
        """Ground language command to image regions"""
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize command
        text_input = clip.tokenize([command]).to(self.device)

        # Get embeddings
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = torch.matmul(text_features, image_features.T)

        return similarity.item()

    def ground_command_to_objects(self, command, candidate_objects):
        """Ground command to specific objects"""
        # Tokenize command and object names
        texts = [command] + candidate_objects
        text_inputs = clip.tokenize(texts).to(self.device)

        # Get text features
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            command_features = text_features[0:1]  # First is the command
            object_features = text_features[1:]    # Rest are objects

            similarities = torch.matmul(command_features, object_features.T)

        # Get probabilities
        probs = F.softmax(similarities[0], dim=0)

        # Return ranked objects
        ranked_indices = torch.argsort(probs, descending=True)
        ranked_objects = [(candidate_objects[i], probs[i].item()) for i in ranked_indices]

        return ranked_objects
```

### Isaac Integration for Language Grounding

#### Isaac Perception for Grounding

```python
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core.robots import Robot
import numpy as np

class IsaacLanguageGrounding:
    def __init__(self, robot_prim_path):
        self.robot_prim_path = robot_prim_path
        self.setup_isaac_environment()

    def setup_isaac_environment(self):
        """Setup Isaac environment for language grounding"""
        # Initialize Isaac World
        self.world = World(stage_units_in_meters=1.0)

        # Add robot
        self.robot = self.world.scene.add(
            Robot(
                prim_path=self.robot_prim_path,
                name="grounding_robot",
                usd_path="path/to/robot.usd",
                position=np.array([0.0, 0.0, 0.1]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0])
            )
        )

        # Add perception sensors
        self.camera = self.world.scene.add(
            Camera(
                prim_path=f"{self.robot_prim_path}/camera",
                name="grounding_camera",
                position=np.array([0.1, 0.0, 0.1]),
                frequency=30,
                resolution=(640, 480)
            )
        )

        self.lidar = self.world.scene.add(
            LidarRtx(
                prim_path=f"{self.robot_prim_path}/lidar",
                name="grounding_lidar",
                translation=np.array([0.1, 0.0, 0.2]),
                config="Example_Rotary",
                depth_range=(0.1, 25.0),
                frequency=10
            )
        )

        # Add semantic segmentation capability
        self.setup_semantic_segmentation()

    def setup_semantic_segmentation(self):
        """Setup semantic segmentation for object grounding"""
        # Enable semantic segmentation on camera
        from omni.isaac.core.utils.semantics import add_semantics

        # Add semantics to camera
        add_semantics(self.camera.prim, "camera")

        # Create semantic labels for common objects
        self.semantic_labels = {
            "table": 1,
            "chair": 2,
            "cup": 3,
            "bottle": 4,
            "box": 5,
            "person": 6,
            "door": 7,
            "window": 8
        }

    def get_grounding_context(self):
        """Get current context for language grounding"""
        # Get robot pose
        robot_pos, robot_orn = self.robot.get_world_pose()

        # Get camera image and segmentation
        rgb_image = self.camera.get_rgb()
        seg_image = self.camera.get_semantic_segmentation()

        # Get LiDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        # Get object poses in scene
        scene_objects = self.get_scene_objects()

        return {
            'robot_pose': (robot_pos, robot_orn),
            'rgb_image': rgb_image,
            'segmentation': seg_image,
            'lidar_data': lidar_data,
            'objects': scene_objects,
            'timestamp': self.world.current_time_step_index
        }

    def get_scene_objects(self):
        """Get objects in current scene with poses"""
        objects = []

        # In a real implementation, this would iterate through all objects
        # in the scene and get their poses
        # For simulation, return mock objects
        for obj_name, label in self.semantic_labels.items():
            # Simulate object detection
            if np.random.random() > 0.3:  # 70% chance of detecting an object
                obj_pose = (
                    np.array([
                        np.random.uniform(-3, 3),
                        np.random.uniform(-3, 3),
                        np.random.uniform(0.1, 1.0)
                    ]),
                    np.array([0.0, 0.0, 0.0, 1.0])  # orientation
                )

                objects.append({
                    'name': obj_name,
                    'pose': obj_pose,
                    'label': label,
                    'confidence': np.random.uniform(0.6, 0.99)
                })

        return objects

    def ground_command(self, command):
        """Ground a command in the current Isaac environment"""
        # Get current context
        context = self.get_grounding_context()

        # Use language grounding to interpret command
        grounding_result = self.perform_language_grounding(command, context)

        return grounding_result

    def perform_language_grounding(self, command, context):
        """Perform actual language grounding using context"""
        # Parse the command
        parsed = self.parse_command(command)

        if parsed['action'] == 'find':
            # Find object in scene
            target_obj = parsed['object']
            found_objects = [obj for obj in context['objects']
                           if target_obj.lower() in obj['name'].lower()]

            if found_objects:
                # Return the most confident detection
                best_obj = max(found_objects, key=lambda x: x['confidence'])
                return {
                    'action': 'navigate_to_object',
                    'target_pose': best_obj['pose'][0],
                    'object_found': True,
                    'object_info': best_obj
                }
            else:
                return {
                    'action': 'search_area',
                    'target_area': self.estimate_search_area(target_obj),
                    'object_found': False
                }

        elif parsed['action'] == 'go_to':
            # Navigate to location
            location = parsed['location']
            target_pose = self.find_location_pose(location, context)

            if target_pose is not None:
                return {
                    'action': 'navigate',
                    'target_pose': target_pose,
                    'location_found': True
                }
            else:
                return {
                    'action': 'explore',
                    'search_pattern': 'spiral',
                    'location_found': False
                }

        else:
            return {
                'action': 'unknown',
                'command': command,
                'parsed': parsed
            }

    def parse_command(self, command):
        """Parse command to extract action and target"""
        command_lower = command.lower()

        # Simple parsing for demonstration
        if 'find' in command_lower or 'look for' in command_lower or 'spot' in command_lower:
            # Extract object to find
            import re
            match = re.search(r'(?:find|look for|spot)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                return {
                    'action': 'find',
                    'object': match.group(1).strip()
                }

        elif 'go to' in command_lower or 'navigate to' in command_lower:
            # Extract location
            import re
            match = re.search(r'(?:go to|navigate to)\s+(.+?)(?:\s|$)', command_lower)
            if match:
                return {
                    'action': 'go_to',
                    'location': match.group(1).strip()
                }

        # Default return
        return {
            'action': 'unknown',
            'raw_command': command
        }

    def find_location_pose(self, location_name, context):
        """Find pose for a named location"""
        # In simulation, use predefined locations
        location_map = {
            'kitchen': np.array([3.0, 1.0, 0.0]),
            'living room': np.array([0.0, 0.0, 0.0]),
            'bedroom': np.array([-2.0, 2.0, 0.0]),
            'office': np.array([1.0, -2.0, 0.0])
        }

        location_key = location_name.lower().strip()
        if location_key in location_map:
            return location_map[location_key]
        else:
            return None  # Location not found

    def estimate_search_area(self, target_object):
        """Estimate where to search for an object"""
        # For common objects, estimate likely locations
        object_locations = {
            'cup': [0.0, 0.0, 0.0],  # Near tables/kitchen
            'book': [0.0, 0.0, 0.0],  # On tables/shelves
            'phone': [0.0, 0.0, 0.0],  # Common areas
            'keys': [0.0, 0.0, 0.0]   # Entry areas
        }

        if target_object.lower() in object_locations:
            return object_locations[target_object.lower()]
        else:
            # Default to current area
            robot_pos, _ = self.robot.get_world_pose()
            return robot_pos
```

### Language Grounding Evaluation

#### Grounding Quality Metrics

```python
class GroundingEvaluator:
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'iou': []
        }

    def evaluate_grounding(self, predicted_grounding, ground_truth_grounding):
        """Evaluate the quality of language grounding"""
        # Calculate various metrics
        accuracy = self.calculate_accuracy(predicted_grounding, ground_truth_grounding)
        precision = self.calculate_precision(predicted_grounding, ground_truth_grounding)
        recall = self.calculate_recall(predicted_grounding, ground_truth_grounding)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        iou = self.calculate_iou(predicted_grounding, ground_truth_grounding)

        # Store metrics
        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1_score'].append(f1_score)
        self.metrics['iou'].append(iou)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou
        }

    def calculate_accuracy(self, pred, gt):
        """Calculate grounding accuracy"""
        # For action grounding: check if correct action was predicted
        if 'action' in pred and 'action' in gt:
            return 1.0 if pred['action'] == gt['action'] else 0.0

        # For object grounding: check if correct object was identified
        if 'object' in pred and 'object' in gt:
            return 1.0 if pred['object'] == gt['object'] else 0.0

        return 0.0

    def calculate_precision(self, pred, gt):
        """Calculate precision of grounding"""
        # Implementation depends on grounding type
        # This is a simplified version
        if 'target' in pred and 'target' in gt:
            return 1.0 if pred['target'] == gt['target'] else 0.0

        return 0.0

    def calculate_recall(self, pred, gt):
        """Calculate recall of grounding"""
        # For grounding, recall is similar to accuracy
        return self.calculate_accuracy(pred, gt)

    def calculate_iou(self, pred, gt):
        """Calculate IoU for spatial grounding"""
        if 'bbox' in pred and 'bbox' in gt:
            # Calculate intersection over union of bounding boxes
            pred_box = pred['bbox']
            gt_box = gt['bbox']

            # Calculate intersection
            inter_x1 = max(pred_box['x'], gt_box['x'])
            inter_y1 = max(pred_box['y'], gt_box['y'])
            inter_x2 = min(pred_box['x'] + pred_box['width'], gt_box['x'] + gt_box['width'])
            inter_y2 = min(pred_box['y'] + pred_box['height'], gt_box['y'] + gt_box['height'])

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

            # Calculate union
            pred_area = pred_box['width'] * pred_box['height']
            gt_area = gt_box['width'] * gt_box['height']
            union_area = pred_area + gt_area - inter_area

            return inter_area / union_area if union_area > 0 else 0.0

        return 0.0

    def get_average_metrics(self):
        """Get average metrics across all evaluations"""
        avg_metrics = {}
        for metric_name, values in self.metrics.items():
            if values:
                avg_metrics[f'avg_{metric_name}'] = sum(values) / len(values)
            else:
                avg_metrics[f'avg_{metric_name}'] = 0.0

        return avg_metrics
```

Language grounding is a critical capability for human-robot interaction, enabling robots to understand and execute natural language commands in their physical environment. The integration of Isaac tools provides the perception and simulation capabilities needed to develop robust grounding systems.