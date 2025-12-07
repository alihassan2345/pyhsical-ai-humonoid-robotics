---
id: 6
title: Create-Hardware-Requirements-Spec
stage: spec
date: 2025-12-06
surface: agent
model: Claude Sonnet 4.5
feature: hardware-requirements
branch: 02-hardware-requirements
user: user
command: /sp.specify
labels: [specification, hardware, robotics, physical-ai]
links:
  spec: specs/02-hardware-requirements/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/02-hardware-requirements/spec.md
 - specs/02-hardware-requirements/checklists/requirements.md
tests: []
---
## Prompt

Project: Physical AI & Humanoid Robotics — Hardware Requirements

Objective:
Provide students with detailed hardware specifications and lab setup guidance to successfully run Physical AI modules, including simulation, perception, and AI-driven robotics.

Modules & Content:

1. Digital Twin Workstation (Required per Student)
   - GPU: NVIDIA RTX 4070 Ti (12GB VRAM) minimum; ideal 3090/4090 for smooth Sim-to-Real training
   - CPU: Intel Core i7 (13th Gen+) or AMD Ryzen 9
   - RAM: 64 GB DDR5 (minimum 32 GB)
   - OS: Ubuntu 22.04 LTS
   - Purpose: Run NVIDIA Isaac Sim, Gazebo, Unity, VLA models
   - Notes: ROS 2 native on Linux; dual-boot recommended if using Windows

2. Physical AI Edge Kit
   - Brain: NVIDIA Jetson Orin Nano (8GB) / Orin NX (16GB)
   - Eyes (Vision): Intel RealSense D435i or D455
   - Inner Ear (Balance): USB IMU (BNO055)
   - Voice Interface: USB Microphone/Speaker (e.g., ReSpeaker)
   - Purpose: Deploy ROS 2 nodes, simulate resource constraints, integrate VLA workflows

3. Robot Lab Options
   - Option A: Proxy (Budget)
     - Robot: Unitree Go2 Edu (~$1,800-$3,000)
     - Pros: Durable, affordable, ROS 2 support
     - Cons: Quadruped, not humanoid
   - Option B: Miniature Humanoid
     - Robot: Unitree G1 (~$16k) or Robotis OP3 (~$12k)
     - Budget Alternative: Hiwonder TonyPi Pro (~$600)
     - Notes: Cheap kits limited to kinematics; AI runs on Jetson
   - Option C: Premium Humanoid Lab
     - Robot: Unitree G1 Humanoid
     - Purpose: Full Capstone deployment, dynamic walking, ROS 2 SDK access

4. Architecture Summary
   - Sim Rig: PC with RTX 4080 + Ubuntu 22.04 → runs Isaac Sim, Gazebo, Unity, trains LLM/VLA
   - Edge Brain: Jetson Orin Nano → runs inference stack
   - Sensors: RealSense Camera + Lidar → feeds data to Jetson
   - Actuator: Unitree Go2/G1 → receives motor commands

5. Cloud-Native Option ("Ether" Lab")
   - Cloud Workstations: AWS g5.2xlarge (A10G, 24GB VRAM) or g6e.xlarge
   - Software: NVIDIA Isaac Sim via Omniverse Cloud
   - Cost Estimate: ~$205/quarter for 120 hours usage + storage
   - Local Bridge Hardware: Jetson Kit + physical robot ($3,700 total approx.)
   - Notes: Cloud-only introduces latency; training locally and flashing models recommended

6. Economy Jetson Student Kit (~$700)
   - Brain: Jetson Orin Nano Super Dev Kit (8GB)
   - Eyes: Intel RealSense D435i (IMU included)
   - Ears: ReSpeaker USB Mic Array v2.0
   - Wi-Fi: included
   - Power/Misc: SD Card (128GB), Jumper Wires
   - Purpose: Learning ROS 2, basic computer vision, Sim-to-Real deployment

7. Latency Trap
   - Problem: Cloud simulation is fast, but real robot control suffers latency
   - Solution: Train in Cloud → download model weights → flash to local Jetson kit

Success Criteria:
- Students can run simulation, perception, and VLA tasks without hardware cr

## Response snapshot

Successfully created a comprehensive specification for the Physical AI & Humanoid Robotics hardware requirements project. The specification includes:

1. Four detailed user stories with priorities:
   - Digital Twin Workstation Setup (P1) - Setting up high-performance workstations for simulation
   - Physical AI Edge Kit Deployment (P1) - Deploying ROS 2 nodes on edge computing platforms
   - Robot Lab Configuration (P2) - Providing access to physical robots for sim-to-real transfer
   - Cloud-Native Lab Option (P3) - Providing cloud-based alternatives for institutions with budget constraints

2. Functional requirements (FR-001 to FR-015) covering:
   - Hardware specifications for workstations and edge computing
   - Budget options and cost estimates
   - Setup instructions and compatibility guidance
   - Network and safety requirements
   - Troubleshooting and maintenance guidance

3. Success criteria (SC-001 to SC-010) with measurable outcomes:
   - 100% of students successfully setting up workstations
   - At least 10 FPS for basic perception tasks on Jetson platforms
   - 90% success rate for sim-to-real model transfer
   - Acceptable cloud performance (at least 30 FPS)
   - Under 4 hours setup time with documentation

4. Quality validation checklist confirming all requirements are met

The specification addresses all the hardware requirements outlined in the user's request, including detailed specifications for Digital Twin workstations, Physical AI Edge kits, robot platform options, cloud alternatives, and sim-to-real transfer workflows. The specification maintains focus on the educational objectives while providing practical guidance for implementation.

## Outcome

- ✅ Impact: Complete spec created for hardware requirements project
- 🧪 Tests: No automated tests for spec creation
- 📁 Files: specs/02-hardware-requirements/spec.md, checklists/requirements.md
- 🔁 Next prompts: Ready for planning phase
- 🧠 Reflection: Spec aligns with hardware requirements and educational goals

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Standard spec creation
- Next experiment (smallest change to try): Proceed to planning phase