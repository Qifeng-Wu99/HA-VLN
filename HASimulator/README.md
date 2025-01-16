# HAVLN-CE

## HA-VLN-CE simulator
<div align="center">
  <img src="../demo/figs/simulator_draft_v2-1.png" alt="image" width="700"/>
</div>

**HA-VLN-CE** simulator incorporates dynamic human activities into photorealistic Habitat environments. The annotation process includes: 1). integrating the [HAPS 2.0 dataset](https://www.dropbox.com/scl/fo/6ofhh9vw5h21is38ahhgc/AOutW4EK3higqNOrX2hQ8rk?rlkey=v88np78ugr49z3sqisnvo6a9i&st=xogu3trq&dl=0) with 172 activities and 486 detailed 3D motion models across 58,320 frames; 2). a two-stage annotation—Stage 1: coarse-to-fine using PSO algorithm and multi-view cameras, and Stage 2: human-in-the-loop for enhancing multi-human interactions and movements; 3). real-time rendering using a signaling mechanism; and 4). enabling agent-environment interactions.


Please follow the SIMULATOR part of the vlnce task config to use human rendering:
```
  ADD_HUMAN: True
  HUMAN_GLB_PATH: path/to/load/motion
  HUMAN_INFO_PATH: path/to/load/human/info
  RECOMPUTE_NAVMESH_PATH: path/to/save/load/navmesh
```

We provide some apis as following:

distance_to_human: Calculate the distance and angle between all people and the agent according to the current position information of the agent (and the direction of the agent).
collisions_detail: List whether there are collisions at each step
human_counting: Count the number of humans present on the current observation.

distance_to_human and collisions_detail can be turn on by adding "DISTANCE_TO_HUMAN" and "COLLISIONS_DETAIL" into MEASUREMENTS in vlnce task config.

```
  MEASUREMENTS: [
    DISTANCE_TO_GOAL,
    SUCCESS,
    SPL,
    NDTW,
    PATH_LENGTH,
    ORACLE_SUCCESS,
    STEPS_TAKEN,
    COLLISIONS,
    COLLISIONS_DETAIL,
    DISTANCE_TO_HUMAN
  ]
```

human_counting is based on [GroundDINO](https://github.com/IDEA-Research/GroundingDINO\).
After you install GroundDINO, you need to set correct path in detector.py.

```python
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = load_model("path/to/cofig", "path/to/checkpoint")
        self.box_threshold = 0.35
        self.text_threshold = 0.25
```
human_counting can be turn on via setting SIMULATOR part of the vlnce task config.
```
HUMAN_COUNTING: True
```



Also, the new metrics metioned in our paper are provied in metric.py.