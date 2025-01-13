# Human-Aware Vision-and-Language Nagivation (HA-VLN)
![License](https://img.shields.io/badge/license-MIT-blue) 

* [Project Web Page](https://havln-project-website.vercel.app/)
* [Dataset](https://www.dropbox.com/home/HA-VLN)

<div align="center">
  <img src="demo/figs/task_define_final-1.png" alt="image" width="700"/>
</div>

Vision-and-Language Navigation (VLN) is crucial for enabling robots to assist humans in everyday environments. However, current VLN systems lack social awareness and rely on simplified instructions with static environments, limiting Sim2Real realizations. To narrow these gaps, we present Human-Aware Vision-and-Language Navigation (**HA-VLN**), expanding VLN to include both discrete (**HA-VLN-DE**) and continuous (**HA-VLN-CE**) environments with social behaviors. The HA-VLN Simulator enables real-time rendering of human activities and provides unified environments for navigation development. It introduces the Human Activity and Pose Simulation (**HAPS**) **Dataset 2.0** with detailed 3D human motion models and the HA Room-to-Room (**HA-R2R**) Dataset with complex navigation instructions that include human activities. We propose an HA-VLN Vision-and-Language model (**HA-VLN-VL**) and a Cross-Model Attention model (**HA-VLN-CMA**) to address visual-language understanding and dynamic decision-making challenges. Comprehensive evaluations and analysis show that dynamic environments with human activities significantly challenge current systems, highlighting the need for specialized human-aware navigation systems for real-world deployment.

## Table of Contents

- [HA-VLN-CE](#ha-vln-ce)
  - [Table of Contents](#-table-of-contents)
  - [🔧 Setup Environment](#-setup-environment)
  - [🐍 Create Conda Environment](#-create-conda-environment)
  - [📥 Download Dataset](#-download-dataset)
  - [🔄 Dataset Preprocessing](#-dataset-preprocessing)
  - [🏗️ Build Habitat](#-build-habitat)
  - [🚀 Run HAVLN Simulator](#-run-havln-simulator)
  - [🌆 Human-Scene Fusion](#-human-scene-fusion)
  - [🖥️ Real-time Human Rendering](#-real-time-human-rendering)
  - [📊 Training](#-training)
  - [📈 Visualization](#-visualization)

---

## 🔧 Setup Environment

 

---

## 🐍 Create Conda Environment

 

---

## 📥 Download Dataset

 

---

## 🔄 Dataset Preprocessing

 

---

## 🏗️ Build Habitat

 

---

## 🚀 Run HAVLN Simulator

 

---

## 🌆 Human-Scene Fusion

 

---

## 🖥️ Real-time Human Rendering

 

---

## 📊 Training

 

## 📈 Visualization

**We present several annotated instances of human subjects from the proposed HAPS 2.0 Dataset (Overall and single), showcasing a variety of well-aligned motions, movements, and interations.** 

<div align="center">
  <img src="demo/gifs/havln.gif" alt="image2" width="700"/>
</div>


**Overall View of Nine Annoated Scenarios from HA-VLN Simulator (90 scans in total)** 

<div align="center">
  <img src="demo/figs/overview_example-1.png" alt="image2" width="700"/>
</div>

**Single Humans with Movements (910 Humans in total)** 

Demo 1|Demo 2|Demo 3
--|--|--
<img src="demo/gifs/demo_1.gif" width="280">|<img src="demo/gifs/demo_2.gif" width="280">|<img src="demo/gifs/demo_3.gif" width="280">


Demo 4|Demo 5|Demo 6
--|--|--
<img src="demo/gifs/demo_4.gif" width="280">|<img src="demo/gifs/demo_5.gif" width="280">|<img src="demo/gifs/demo_6.gif" width="280">

**Navigation Visualization**

Navigation Demo 1|Navigation Demo 2
--|--
<img src="demo/gifs/nav1.gif" width="350">|<img src="demo/gifs/nav2.gif" width="350">
**Navigation Instruction**: Start by moving forward in the lounge area, where an individual is engaged in a phone conversation while pacing back and forth. Navigate carefully to avoid crossing their path. As you proceed, you will pass by a television mounted on the wall. Continue your movement, observing people relaxing and watching the TV, some seated comfortably on sofas. Further along, notice a group of friends raising their glasses in a toast, enjoying cocktails together. Maintain a steady course, ensuring you do not disrupt their gathering. Finally, reach the end of your path where a potted plant is situated next to a door. Stop at this location, positioning yourself near the plant and door without obstructing access.|**Navigation Instruction**: Exit the room and make a left turn. Proceed down the hallway where an individual is ironing clothes, carefully smoothing out wrinkles on garments. Continue walking and make another left turn. Enter the next room, which is a bedroom. Inside, someone is comfortably seated in bed, engrossed in reading a book. Move past the bed, ensuring not to disturb the reader. Turn left again to enter the bathroom. Once inside, position yourself near the sink and wait there, observing the surroundings without interfering with any activities.

## Contributing

We welcome contributions to this project! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

