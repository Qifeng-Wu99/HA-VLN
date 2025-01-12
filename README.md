# HAVLN-CE

## Overall View
<div align="center">
  <img src="figs/task_define_final-1.png" alt="image" width="700"/>
</div>
Vision-and-Language Navigation (VLN) is crucial for enabling robots to assist humans in everyday environments. However, current VLN systems lack social awareness and rely on simplified instructions with static environments, limiting Sim2Real realizations. To narrow these gaps, we present Human-Aware Vision-and-Language Navigation (HA-VLN), expanding VLN to include both discrete (HA-VLN-DE) and continuous (HA-VLN-CE) environments with social behaviors. The HA-VLN Simulator enables real-time rendering of human activities and provides unified environments for navigation development. It introduces the Human Activity and Pose Simulation (HAPS) Dataset 2.0 with detailed 3D human motion models and the HA Room-to-Room (HA-R2R) Dataset with complex navigation instructions that include human activities. We propose an HA-VLN Vision-and-Language model (HA-VLN-VL) and a Cross-Model Attention model (HA-VLN-CMA) to address visual-language understanding and dynamic decision-making challenges. Comprehensive evaluations and analysis show that dynamic environments with human activities significantly challenge current systems, highlighting the need for specialized human-aware navigation systems for real-world deployment.

## HA-VLN-CE Simulator
<div align="center">
  <img src="figs/simulator_draft_v2-1.png" alt="image" width="700"/>
</div>

**HA-VLN-CE** simulator incorporates dynamic human activities into photorealistic Habitat environments. The annotation process includes: 1). integrating the HAPS 2.0 dataset with 172 activities and 486 detailed 3D motion models across 58,320 frames; 2). a two-stage annotation—Stage 1: coarse-to-fine using PSO algorithm and multi-view cameras, and Stage 2: human-in-the-loop for enhancing multi-human interactions and movements; 3). real-time rendering using a signaling mechanism; and 4). enabling agent-environment interactions.


Demo 1|Demo 2|Demo 3
--|--|--
<video src="vids/demo_1.mp4" width="400" />|<video src="vids/demo_2.mp4" width="400" />|<video src="vids/demo_3.mp4" width="400" /></video>


Demo 4|Demo 5|Demo 6
--|--|--
<video src="vids/demo_4.mp4" width="400" />|<video src="vids/demo_5.mp4" width="400" />|<video src="vids/demo_6.mp4" width="400" /></video>

<div align="center">
  <img src="figs/overview_example-1.png" alt="image2" width="700"/>
</div>