---
id: 7
title: Implement-Hardware-Requirements-Module
stage: implement
date: 2025-12-06
surface: agent
model: Claude Sonnet 4.5
feature: hardware-requirements
branch: 02-hardware-requirements
user: user
command: /sp.implement
labels: [implementation, hardware, robotics, docusaurus]
links:
  spec: specs/02-hardware-requirements/spec.md
  plan: null
  tasks: null
  adr: null
  pr: null
files:
 - physical-ai/docs/module-hardware-requirements/hardware-requirements.md
 - physical-ai/sidebars.js
tests: []
---
## Prompt

Project: Physical AI & Humanoid Robotics — Hardware Requirements Module

Instructions for Claude CLI:

1. **Target Folder**
   - All files should go under the Docusaurus project folder `physical-ai/docs/module-hardware-requirements`.

2. **Markdown File Structure**
   - Create a main file: `hardware-requirements.md`.
   - Structure content with headings reflecting the Spec-Kit Plus spec:
     ```
     # Module: Hardware Requirements
     ## 1. Digital Twin Workstation
     ## 2. Physical AI Edge Kit
     ## 3. Robot Lab Options
     ## 4. Architecture Summary
     ## 5. Cloud-Native Option ("Ether" Lab)
     ## 6. Economy Jetson Student Kit
     ## 7. Latency Trap
     ```

3. **Content Formatting**
   - Each section should include:
     - Bullet lists for hardware specs (GPU, CPU, RAM, OS)
     - Tables for summary components and costs
     - Notes or warnings in **bold** or blockquote style
   - Example table format:
     | Component | Hardware | Function |
     |-----------|---------|---------|
     | Sim Rig | RTX 4080 + Ubuntu 22.04 | Runs Isaac Sim, Gazebo, Unity, trains LLM/VLA models |
     | Edge Brain | Jetson Orin Nano | Runs inference stack |

4. **Sidebar Update**
   - Add the Hardware Requirements module to `sidebars.js`:
     ```javascript
     module.exports = {
       tutorialSidebar: [
         'introduction',
         'module-1-ros2',
         'module-2-digital-twin',
         'module-3-nvidia-isaac',
         'module-4-vla',
         'module-hardware-requirements/hardware-requirements',
       ],
     };
     ```

5. **Styling and Notes**
   - Use Docusaurus Markdown conventions:
     - `#`, `##`, `###` for headings
     - Code blocks for hardware setup commands (if needed)
     - Blockquotes for warnings and notes

## Response snapshot

Successfully implemented the Hardware Requirements module for the Physical AI & Humanoid Robotics curriculum. The implementation included:

1. Directory Structure:
   - Created `physical-ai/docs/module-hardware-requirements/` directory

2. Content Creation:
   - Created `hardware-requirements.md` with all 7 required sections:
     * Digital Twin Workstation with detailed specs (GPU, CPU, RAM, OS)
     * Physical AI Edge Kit with components and performance characteristics
     * Robot Lab Options with cost comparison table
     * Architecture Summary with system architecture table
     * Cloud-Native Option with cost analysis
     * Economy Jetson Student Kit with component breakdown
     * Latency Trap with problem/solution and best practices

3. Navigation Integration:
   - Updated `sidebars.js` to include the new Hardware Requirements module as a category
   - Added proper navigation structure following Docusaurus conventions

4. Content Formatting:
   - Used appropriate headings (H1, H2) as specified
   - Created tables for hardware specifications and cost comparisons
   - Added blockquotes for important notes and warnings
   - Used bullet lists for hardware specs and components

5. Validation:
   - Successfully tested Docusaurus build with `npm run build`
   - Site builds without errors and includes the new hardware requirements module

The module provides comprehensive hardware guidance for students, covering all aspects from high-performance workstations to budget-friendly student kits, with clear cost breakdowns and architectural recommendations.

## Outcome

- ✅ Impact: Complete hardware requirements module created for Physical AI curriculum
- 🧪 Tests: Docusaurus build test completed successfully
- 📁 Files: hardware-requirements.md and updated sidebars.js
- 🔁 Next prompts: Ready for content review or additional module development
- 🧠 Reflection: Module aligns with hardware requirements and educational goals

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Build validation passed
- Prompt variant (if applicable): Standard implementation
- Next experiment (smallest change to try): Review content with hardware specialists