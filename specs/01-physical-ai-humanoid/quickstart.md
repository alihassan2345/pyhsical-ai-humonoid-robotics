# Quickstart Guide: Physical AI & Humanoid Robotics Curriculum

## Overview
This guide will help you set up the environment to work with the Physical AI & Humanoid Robotics curriculum. The curriculum covers Physical AI principles, ROS 2 systems, simulation environments, AI-robot integration, and humanoid robotics.

## Prerequisites
- Python 3.11+
- Git
- Docker (for containerized environments)
- Basic knowledge of Python and AI concepts

## Setup Steps

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install Docusaurus (for viewing documentation)
```bash
npm install
```

### 3. Install ROS 2 Humble Hawksbill
Follow the official installation guide for your OS:
- Ubuntu: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
- Windows: https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html
- Docker: Use the official ROS 2 Docker images

### 4. Set up Simulation Environment
Choose one or both:
- **Gazebo**: Install Ignition Gazebo Fortress or Garden
- **Unity**: Download Unity Hub and install Unity 2022.3 LTS or newer

### 5. Install NVIDIA Isaac Sim (Optional but Recommended)
- Download from NVIDIA Developer website
- Follow installation instructions for your platform
- Ensure compatibility with your ROS 2 installation

### 6. Verify Installation
Run these commands to verify your setup:
```bash
# Check Python version
python3 --version

# Check ROS 2 installation
source /opt/ros/humble/setup.bash  # or your ROS 2 setup script
ros2 --version

# Test ROS 2 with a simple example
ros2 run demo_nodes_cpp talker
```

### 7. Start the Documentation Server
```bash
npm start
```
This will start a local server at http://localhost:3000 where you can view the curriculum.

## First Module: Introduction to Physical AI
1. Navigate to the first module in the documentation
2. Complete the hands-on exercises
3. Verify your understanding with the assessment criteria

## Troubleshooting
- If ROS 2 commands are not found, ensure you've sourced the setup script
- If documentation doesn't build, check that Node.js and npm are properly installed
- For simulation issues, verify that your GPU drivers are up to date

## Next Steps
After completing the setup:
1. Proceed to Module 0: Introduction to Physical AI
2. Follow the progressive curriculum from basic concepts to advanced humanoid systems
3. Complete the capstone project integrating all learned concepts