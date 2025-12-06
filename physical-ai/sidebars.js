// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros2/ros2-overview',
        'module-1-ros2/nodes-topics-services',
        'module-1-ros2/rclpy-bridge',
        'module-1-ros2/urdf-humanoids'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin/gazebo-basics',
        'module-2-digital-twin/physics-simulation',
        'module-2-digital-twin/sensor-simulation',
        'module-2-digital-twin/unity-visualization'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI–Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac/isaac-overview',
        'module-3-isaac/synthetic-data',
        'module-3-isaac/vslam-navigation',
        'module-3-isaac/nav2-humanoids'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision–Language–Action (VLA)',
      items: [
        'module-4-vla/vla-introduction',
        'module-4-vla/speech-to-text',
        'module-4-vla/llm-planning',
        'module-4-vla/ros-action-mapping'
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Humanoid Robot Systems',
      items: [
        'module-5-humanoids/kinematics-dynamics',
        'module-5-humanoids/bipedal-locomotion',
        'module-5-humanoids/manipulation-grasping',
        'module-5-humanoids/human-robot-interaction'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project: Autonomous Humanoid Robot',
      items: [
        'capstone/capstone-overview',
        'capstone/voice-command-integration',
        'capstone/task-planning-implementation',
        'capstone/navigation-manipulation-integration'
      ],
    }
  ],
};

export default sidebars;
