# Human-Aware Vision-and-Language Nagivation (HA-VLN)
![License](https://img.shields.io/badge/license-MIT-blue) 

* 🚀 [Project Web Page](https://havln-project-website.vercel.app/)
* 📂 [Dataset](https://www.dropbox.com/scl/fo/6ofhh9vw5h21is38ahhgc/AOutW4EK3higqNOrX2hQ8rk?rlkey=gvvqy4lsusthzwt9974kkyn7s&st=7l5drspw&dl=0)

## 🧭 What does HA-VLN look like?

Navigation Demo 1|Navigation Demo 2
--|--
<img src="demo/gifs/nav1.gif" width="350">|<img src="demo/gifs/nav2.gif" width="350">
**Navigation Instruction**: Start by moving forward in the lounge area, **where an individual is engaged in a phone conversation while pacing back and forth**. Navigate carefully to avoid crossing their path. As you proceed, you will pass by a television mounted on the wall. Continue your movement, **observing people relaxing and watching the TV, some seated comfortably on sofas**. Further along, **notice a group of friends raising their glasses in a toast, enjoying cocktails together**. Maintain a steady course, ensuring you do not disrupt their gathering. Finally, reach the end of your path where a potted plant is situated next to a door. Stop at this location, positioning yourself near the plant and door without obstructing access.|**Navigation Instruction**: Exit the room and make a left turn. Proceed down the hallway **where an individual is ironing clothes, carefully smoothing out wrinkles on garments**. Continue walking and make another left turn. Enter the next room, which is a bedroom. Inside, **someone is comfortably seated in bed, engrossed in reading a book**. Move past the bed, ensuring not to disturb the reader. Turn left again to enter the bathroom. Once inside, position yourself near the sink and wait there, observing the surroundings without interfering with any activities.

## Abstract

<div align="center">
  <img src="demo/figs/task_define_final-1.png" alt="image" width="700"/>
</div>

We present Human-Aware Vision-and-Language Navigation (**HA-VLN**), expanding VLN to include both discrete (**HA-VLN-DE**) and continuous (**HA-VLN-CE**) environments with social behaviors (This repo focus on CE). The [HA-VLN Simulator](HASimulator) enables real-time rendering of human activities and provides unified APIs for navigation development. It introduces the Human Activity and Pose Simulation ([**HAPS 2.0 Dataset**](Data/HAPS2.0)) with detailed 3D human motion models and the HA Room-to-Room ([**HA-R2R**](Data/HA-R2R)) Dataset with complex navigation instructions that include human activities. We propose an HA-VLN Vision-and-Language model ([**HA-VLN-VL**](agent)) and a Cross-Model Attention model ([**HA-VLN-CMA**](agent)) to address visual-language understanding and dynamic decision-making challenges.

## Table of Contents

- [HA-VLN-CE](#ha-vln-ce)
  - [Table of Contents](#-table-of-contents)
  - [🚀 Quick Start](#-quick-start)
  - [📥 Download Dataset](#-download-dataset)
  - [🔄 Dataset Organization](#-dataset-organization)
  - [🌆 Human-Scene Fusion](#-human-scene-fusion)
  - [🖥️ Real-time Human Rendering](#-real-time-human-rendering)
  - [📊 Training](#-training)
  - [📈 Visualization](#-visualization)
  - [📂 Dataset Details](#-dataset-details)

---

## 🚀 Quick Start
```bash
git clone https://github.com/F1y1113/HAVLN-CE.git
cd HAVLN-CE
```
Set up a Conda environment for the simulator.
Please install habitat-lab (v0.1.7) and habitat-sim (v0.1.7) follow [ETPNav](https://github.com/MarSaKi/ETPNav/) (please note that we use python==3.7).
```bash
conda create -n havlnce python=3.7
conda activate havlnce

conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

git clone --branch v0.1.7 git@githubcom:facebookresearch/habitat-lab.git
cd habitat-lab
python setup.py develop --all # install habitat and habitat_baselines
cd ..
```

And follow [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/) to install GroundingDINO (please note that we use supervision==0.11.1).

```bash
cd HASimulator
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .

mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```
Finally, you should install necessary packages for agent.

```bash
pip install -r requirements.txt
```

---

## 📥 Download Dataset

To use the simulator, download the [Matterport3D Dataset](https://niessner.github.io/Matterport/) (access required).

```bash
python2 download_mp.py -o Data/scene_datasets --type matterport_mesh house_segmentations region_segmentations poisson_meshes
```

To download and extract HA-R2R and HAPS 2.0 datasets, simply run:

```bash
bash scripts/download_data.sh
```
Baseline models encode depth observations using a ResNet pre-trained on PointGoal navigation. Those weights can be downloaded from [here](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo). Extract the contents to [Data/ddppo-models](Data/ddppo-models)/{model}.pth.

---

## 🔄 Dataset Organization

- **Data**
  - **HA-R2R**
    - train
    - val_seen
    - val_unseen
  - **HAPS2.0**
    - balcony:A_child_excitedly_greeting_a_pet._0
    - balcony:A_couple_having_a_quiet,_intimate_conversation._0
    - ......
  - **ddppo-models**
  - **scene_datasets**

---

## 🌆 Human-Scene Fusion

We use nine cameras to annotate any anomalies, such as levitation or model clipping, in humans added to the scene. Check details in [scripts/human_scene_fusion.py](https://github.com/F1y1113/HAVLN-CE/blob/main/scripts/human_scene_fusion.py).
- **Inspired by:** Real-world **3D skeleton tracking** techniques.
- **Setup:**
  - **9 RGB cameras** surround each human model to refine **position & orientation**.
  - **Multi-view capture** to correct **clipping issues** with surrounding objects.
- **Camera Angles:**
  - **8 side cameras:** $\theta_{\text{lr}}^{i} = \frac{\pi i}{8}$, alternate **up/down tilt**.
  - **1 overhead camera:** $\theta_{\text{ud}}^{9} = \frac{\pi}{2}$.

To use it, you need to modify the data path as
 ```
data_path = "../Data/HAPS2.0"
output_path = "test/"
json_path = "../Data/human_motion.json"
 ```

---

## 🖥️ Real-time Human Rendering

Human Rendering is defined in the class **HAVLNCE** of [HASimulator/enviorments.py](https://github.com/F1y1113/HAVLN-CE/blob/main/HASimulator/environments.py).

Human Rendering uses child threads for timing and the main thread for adding, adding and recalculating the required navmesh in real time.


To enable human rendering, you should follow these setting in [vlnce task config](https://github.com/F1y1113/HAVLN-CE/blob/main/Agent/VLN-CE/habitat_extensions/config/vlnce_task.yaml).
```
SIMULATOR:
  ADD_HUMAN: True
  HUMAN_GLB_PATH: ../Data/HAPS2.0
  HUMAN_INFO_PATH: ../Data/human_motion.json
  RECOMPUTE_NAVMESH_PATH: ../Data/recompute_navmesh
```
---

## 📊 Training

 To train the agent of VLN-CE, you can use the script in orignal VLN-CE.
 ```bash
 cd agent
 python run_VLNCE.py \
  --exp-config path/to/experiment_config.yaml \
  --run-type {train | eval | inference}
 ```

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

## 📂 Dataset Details

🚀🚀🚀 [**Download Here**](https://www.dropbox.com/scl/fo/6ofhh9vw5h21is38ahhgc/AOutW4EK3higqNOrX2hQ8rk?rlkey=gvvqy4lsusthzwt9974kkyn7s&st=tqjr6by0&dl=0)

### HAPS Dataset 2.0

In real-world scenarios, human motion typically adapts and interacts with the surrounding region. The proposed **Human Activity and Pose Simulation (HAPS) Dataset 2.0** improves upon [**HAPS 1.0**](https://github.com/lpercc/HA3D_simulator/) by making the following enhancements:  
1. *Refining and diversifying human motions.*  
2. *Providing descriptions closely tied to region awareness.*   

HAPS 2.0 mitigates the limitations of existing human motion datasets by identifying **26 distinct regions** across **90 architectural scenes** and generating **486 human activity descriptions**, encompassing both **indoor and outdoor environments**. These descriptions, validated through **human surveys** and **quality control using ChatGPT-4**, include realistic actions and region annotations (e.g., *"workout gym exercise: An individual running on a treadmill"*).

The [**Motion Diffusion Model (MDM)**](https://guytevet.github.io/mdm-page/) converts these descriptions into **486 detailed 3D human motion models** $\mathbf{H}$[^1] using the **SMPL model**, each transformed into a **120-frame motion sequence** $\mathcal{H}$.  

Each **120-frame SMPL mesh sequence** $\mathcal{H} = \langle h_1, h_2, \ldots, h_{120} \rangle$ details **3D human motion and shape information** through the **SMPL model**.

---

[^1]: $\mathbf{H}\ = \mathbb{R}^{486 \times 120 \times(10+72+6890 \times 3)}$, representing **486 models**, each with **120 frames**, including **shape, pose, and mesh vertex parameters**.

### HA-R2R Dataset

Instruction Examples Table presents four instruction examples from the **Human-Aware Room-to-Room (HA-R2R) dataset**. These cases include various scenarios such as:
- **Multi-human interactions** (e.g., 1, 2, 3),
- **Agent-human interactions** (e.g., 1, 2, 3),
- **Agent encounters four or more humans** (e.g., 3),
- **No humans encountered** (e.g., 4).

These examples illustrate the diversity of **human-aligned navigation instructions** that challenge the agent in our task.

#### Instruction Examples Table

| **Instruction Example** |
|-------------------------|
| **1.** Exit the library and turn left. As you proceed straight ahead, you will enter the bedroom, **where you can observe a person actively searching for a lost item, perhaps checking under the bed or inside drawers**. Continue moving forward, **ensuring you do not disturb his search**. As you pass by, **you might see a family engaged in a casual conversation on the porch or terrace**, **be careful not to bump into them**. Maintain your course until you reach the closet. Stop just outside the closet and await further instructions. |
| **2.** Begin your path on the left side of the dining room, **where a group of friends is gathered around a table, enjoying dinner and exchanging stories with laughter**. As you move across this area, **be cautious not to disturb their gathering**. The dining room features a large table and chairs. Proceed through the doorway that leads out of the dining room. Upon entering the hallway, continue straight and then make a left turn. As you walk down this corridor, you might notice framed pictures along the walls. The sound of laughter and conversation from the dining room may still be audible as you move further away. Continue down the hallway until you reach the entrance of the office. Here, **you will observe a person engaged in taking photographs, likely focusing on capturing the view from a window or an interesting aspect of the room**. Stop at this point, ensuring you are positioned at the entrance without obstructing the photographer's activity. |
| **3.** Starting in the living room, **you can observe an individual practicing dance moves, possibly trying out new steps**. As you proceed straight ahead, **you will pass by couches where a couple is engaged in a quiet, intimate conversation, speaking softly to maintain their privacy**. Continue moving forward, ensuring you navigate around any furniture or obstacles in your path. As you transition into the hallway, **notice another couple enjoying a date night at the bar, perhaps sharing drinks and laughter**. **Maintain a steady course without disturbing them**, keeping to the right side of the hallway. Upon reaching the end of your path, you will find yourself back in the living room. Here, **a person is checking their appearance in a hallway mirror, possibly adjusting their attire or hair**. Stop by the right candle mounted on the wall, ensuring you are positioned without blocking any pathways. |
| **4.** Begin by leaving the room and turning to your right. Proceed down the hallway, be careful of any human activity or objects along the way. As you continue, look for the first doorway on your right. Enter through this doorway and advance towards the shelves. Once you reach the vicinity of the shelves, come to a halt and wait there. During this movement, avoid any obstacles or disruptions in the environment. |

 **purple-highlighted instructions** relate to **human movements**, and **blue-highlighted instructions** are associated with **agent-human interactions**.

---

#### HA-R2R Instruction Generation

To generate new instructions for the **HA-R2R dataset**, we employ **ChatGPT-4o** and **LLaMA-3-8B-Instruct** to **contextually enrich and expand scene information** based on the original instructions from the **R2R-CE dataset**.

**Few-Shot Prompting Approach**
Our approach utilizes a **few-shot template prompt**, consisting of:
- **A system prompt** 
- **A set of few-shot examples** 

The **system prompt** primes the LLMs with the **context and requirements** for generating **navigation instructions** in human-populated environments. It outlines the **desired characteristics**, such as:
- **Relevance** to the navigation task,
- **Integration of human activities and agent interactions**, and
- **Precision in describing environmental details**.

The **few-shot examples** serve as **guidelines** for how the instructions should be structured, demonstrating:
- **Incorporation of human activities**,
- **Use of relative position information**, and
- **Integration with original navigation instructions**.

For instance, one **example** includes:
> *“You will notice someone quietly making a phone call, so please remain quiet as you move.”*

---

**Iterative Refinement Process**
Initially, the models produced **irrelevant or subjective content** and lacked sufficient **detail about human activities**. To improve this:
1. We **reviewed outputs** and identified discrepancies.
2. We **refined the system prompt** to emphasize **neutral tone** and **avoid subjective descriptions**.
3. We **updated few-shot examples** to demonstrate **desired level of detail**.
4. Through multiple iterations, we ensured that **scene-enriched instructions** remained coherent and relevant to **HA-R2R**.

---

#### **LLM Prompt for HA-R2R Instruction Generation**
**System Prompt**
```json
{
  "system": "You are tasked with generating an objective and detailed description of a path and the related scene information based on the given input. Focus solely on observable entities and actions in the scene, avoiding descriptions of smell, atmosphere, or subjective feelings. Ensure clarity and precision in object and human activity descriptions.",
  "examples": "... //examples in Listing 4",
  "user": "Input:\n[instruction_text]: \"...\";\n[start]: \"...\";\n[during]: \"...\";\n[end]: \"...\";\n\nPlease provide an objective, step-by-step description of the path, focusing on the objects and human activities observed at each stage of the navigation (beginning, during, and end). Expand your description by including character actions, objective descriptions of objects, and details of the movement process. Your description should maintain a neutral tone and avoid embellishments. Please simulate the robot's movement along the path, while incorporating possible connections or interactions between the robot, objects, and human activities.\n\nOutput: ... //generation"
}

{
  "example_1": {
    "input": {
      "instruction_text": "Go through the doorway to the hall. Go down the hall past the landing. Take the first door on the right and stop in the doorway.",
      "start": "other room: A person using the room for a quiet phone call.",
      "during": "other room: A person using the room for a quiet phone call.",
      "end": "hallway: An individual cleaning or vacuuming."
    },
    "output": "You should start from the room. Before you walk through the doorway into the hallway, you will notice someone quietly making a phone call, so please remain quiet as you move. Then, proceed down the hallway past the landing. As you continue through the corridor, you will see a responsible cleaner using a vacuum to tidy up. Finally, your destination is at the end of the hallway, enter the first door on the right and stop in the doorway."
  }
}
```

### 🏃 Human Activities Annotation

#### 🌍 **Stage 1: Coarse Annotation**
- **Goal:** Assign human motions to specific **regions** and **objects** using a **coarse-to-fine approach**.
- **Process:**
  - Filter human motions **$\mathbf{H}$** based on region **$\mathbf{R}$** and object list **$\mathbf{O}$**.
  - Match motions **$h_i$** with objects **$j_i$** using **semantic similarity**.
  - Optimize human placements **$\mathbf{p}_{opt}^{h_i}$** using **Particle Swarm Optimization (PSO)**.  
- **Constraints:**
  - Search space limited by **region boundaries**.
  - Maintain **minimum safe distance** $\epsilon = 1m$ from other objects.
  - Ensures **naturalistic human placements** for training navigation agents.

#### 🎥 **Stage 2: Fine Annotation**
- **Inspired by:** Real-world **3D skeleton tracking** techniques.
- **Setup:**
  - **9 RGB cameras** surround each human model to refine **position & orientation**.
  - **Multi-view capture** to correct **clipping issues** with surrounding objects.
- **Camera Angles:**
  - **8 side cameras:** $\theta_{\text{lr}}^{i} = \frac{\pi i}{8}$, alternate **up/down tilt**.
  - **1 overhead camera:** $\theta_{\text{ud}}^{9} = \frac{\pi}{2}$.
- **Scale:** 529 human models annotated in **374 regions** across **90 scans**.

#### 👥 **Multi-Human Interaction & Motion Enrichment**
- **Goal:** Increase **scene diversity** and **human interactions**.
- **Process:**
  - Use **LLMs** to generate new multi-human interactions.
  - **Manual refinement (4 rounds)** ensures consistency.
  - Place new motions relative to objects & use **multi-camera annotation**.
- **Result:**  
  - **910 human models** across **428 regions**.
  - **Complex motions**: Walking downstairs, climbing stairs.
  - **Interaction stats:** 72 **two-human pairs**, 59 **three-human pairs**, 15 **four-human groups**.
- **Impact:** Enables precise **social modeling** for human-aware navigation.

## Contributing

We welcome contributions to this project! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how to contribute.

## License

This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

