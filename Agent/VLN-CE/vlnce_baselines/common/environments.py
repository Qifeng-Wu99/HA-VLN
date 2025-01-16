from typing import Any, Dict, Optional, Tuple, Union

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry

from habitat_extensions.discrete_planner import DiscretePathPlanner
from habitat_extensions.utils import generate_video, navigator_video_frame

import json
import os
import threading
import queue
import time
import magnum as mn
import random
from habitat.core.logging import logger
import habitat_sim


import sys 
sys.path.append('../..')
from HASimulator.environments import *

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

@baseline_registry.register_env(name="HAVLNCEDaggerEnv")
class HAVLNCEDaggerEnv(VLNCEDaggerEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)
        if config.TASK_CONFIG.SIMULATOR.ADD_HUMAN:
            # Initialize simulator resources
            self.havlnce_tool = HAVLNCE(config.TASK_CONFIG, self._env._sim)

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        if self._env._config.SIMULATOR.ADD_HUMAN:
            self.havlnce_tool.reset()

        observations = super().reset()

        return observations
    
    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        if self._env._config.SIMULATOR.ADD_HUMAN:
            self.havlnce_tool._handle_signals()
        observations = super().step(action, **kwargs)

        return observations


@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }


@baseline_registry.register_env(name="VLNCEWaypointEnv")
class VLNCEWaypointEnv(habitat.RLEnv):
    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self._rl_config = config.RL
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self) -> Tuple[float, float]:
        return (
            np.finfo(np.float).min,
            np.finfo(np.float).max,
        )

    def get_reward(self, observations: Observations) -> float:
        return self._env.get_metrics()[self._reward_measure_name]

    def _episode_success(self) -> bool:
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over or self._episode_success()

    def get_info(self, observations: Observations) -> Dict[str, Any]:
        return self.habitat_env.get_metrics()

    def get_num_episodes(self) -> int:
        return len(self.episodes)


@baseline_registry.register_env(name="VLNCEWaypointEnvDiscretized")
class VLNCEWaypointEnvDiscretized(VLNCEWaypointEnv):
    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []

        step_size = config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE
        self.discrete_planner = DiscretePathPlanner(
            forward_distance=step_size,
            turn_angle=np.deg2rad(config.TASK_CONFIG.SIMULATOR.TURN_ANGLE),
            goal_radius=round(step_size / 2, 2) + 0.01,  # 0.13m for 0.25m step
        )
        super().__init__(config, dataset)

    def get_reward(self, *args: Any, **kwargs: Any) -> float:
        return 0.0

    def reset(self) -> Observations:
        observations = self._env.reset()
        if self.video_option:
            agent_state = self._env.sim.get_agent_state()
            start_pos = agent_state.position
            start_heading = agent_state.rotation

            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, info, start_pos, start_heading
                )
            ]

        return observations

    def step(
        self, action: Union[int, str, Dict[str, Any]], *args, **kwargs
    ) -> Tuple[Observations, Any, bool, dict]:
        observations = None
        start_pos, start_heading = None, None

        if self.video_option:
            agent_state = self._env.sim.get_agent_state()
            start_pos = agent_state.position
            start_heading = agent_state.rotation

        if action != "STOP":
            plan = self.discrete_planner.plan(
                r=action["action_args"]["r"],
                theta=action["action_args"]["theta"],
            )
            if len(plan) == 0:
                agent_state = self._env.sim.get_agent_state()
                observations = self._env.sim.get_observations_at(
                    agent_state.position, agent_state.rotation
                )

            for discrete_action in plan:
                observations = self._env.step(discrete_action, *args, **kwargs)
                if self.video_option:
                    info = self.get_info(observations)
                    self.video_frames.append(
                        navigator_video_frame(
                            observations,
                            info,
                            start_pos,
                            start_heading,
                            action,
                        )
                    )

                if self._env.episode_over:
                    break
        else:
            observations = self._env.step(action, *args, **kwargs)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        start_pos,
                        start_heading,
                        action,
                    )
                )

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        if self.video_option and done:
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 5)},
                tb_writer=None,
                fps=8,
            )

        return observations, reward, done, info



# @baseline_registry.register_env(name="HAVLNCEDaggerEnv")
# class HAVLNCEDaggerEnv(habitat.RLEnv):
#     def __init__(self, config: Config, dataset: Optional[Dataset] = None):
#         super().__init__(config.TASK_CONFIG, dataset)
#         if config.TASK_CONFIG.SIMULATOR.ADD_HUMAN:
#             # Initialize simulator resources
#             self._initialize_simulator_resources()
#             self.__init_manager__()

#             # Initialize total signals sent and previous human object IDs
#             self.total_signals_sent = 0  # Total number of signals sent
#             self.previous_human_object_ids = []  # Store object IDs to remove later

#             self.recomputed_navmesh = False

#             # Initialize a lock for thread safety
#             self.human_model_lock = threading.Lock()

#             # Initialize the signal queue for thread communication, maxsize=120
#             self.signal_queue = queue.Queue(maxsize=120)

#             # Start the child thread to send signals
#             self._signal_thread = threading.Thread(target=self._signal_sender)
#             self._signal_thread.daemon = True
#             self._signal_thread.start()

#     def get_reward_range(self) -> Tuple[float, float]:
#         # We don't use a reward for DAgger, but the baseline_registry requires
#         # we inherit from habitat.RLEnv.
#         return (0.0, 0.0)

#     def get_reward(self, observations: Observations) -> float:
#         return 0.0

#     def get_done(self, observations: Observations) -> bool:
#         return self._env.episode_over

#     def get_info(self, observations: Observations) -> Dict[Any, Any]:
#         return self.habitat_env.get_metrics()
    
#     def reset(self) -> Observations:
#         r"""Resets the environments and returns the initial observations.

#         :return: initial observations from the environment.
#         """
#         if self._env._config.SIMULATOR.ADD_HUMAN:
#             # Clear signal queue and reset counters
#             self._reset_signal_queue_and_counters()

#             self.signal_queue.put('REFRESH_HUMAN_MODEL', timeout=1)
#             self.total_signals_sent += 1  # Increment total signals sent

#             # self._handle_signals()
#             self.add_new_human_model(frame_id=0)

#         observations = super().reset()

#         return observations
    
#     def _reset_signal_queue_and_counters(self):
#         """Reset the signal queue and related counters before starting a new episode."""
#         # Clear the signal queue
#         with self.signal_queue.mutex:
#             self.signal_queue.queue.clear()
#         # logger.info("Signal queue cleared.")

#         # Reset total signals sent counter
#         self.total_signals_sent = 0
#         # logger.info("Total signals sent counter reset to 0.")

#         # Reset frame counter if used
#         self.frame_counter = 0
#         # logger.info("Frame counter reset to 0.")

#         # Remove any existing human models
#         self.remove_previous_human_model()
#         # logger.info("Previous human models removed.")
#         self.frame_id = 0

#         self.recomputed_navmesh = False

#     def step(
#         self, action: Union[int, str, Dict[str, Any]], **kwargs
#     ) -> Observations:
#         if self._env._config.SIMULATOR.ADD_HUMAN:
#             self._handle_signals()
#         observations = super().step(action, **kwargs)

#         return observations

#     def _initialize_simulator_resources(self):
#         """Initialize resources required by the simulator."""
#         # Load human motion data and other resources
#         self.human_motion_json_path = "/datadrive/HAVLN-CE/scripts/motion_refine/adjust_phase4/human_motion_v9_sum_ver5.json"
#         # self.human_motion_json_path = "/datadrive/HAVLN-CE/scripts/motion_refine/adjust_phase3/human_motion_v8_sum_ver2_single.json"
#         # self.data_path = "/datadrive/vln_share/habitat-sim/data"
#         self.data_path = "/datadrive/HAVLN-CE/scripts/data_preprocess/human_motion/human_motion_glbs_v3"
#         self.nav_mesh_path = "/datadrive/HAVLN-CE/scripts/motion_refine/adjust_phase2/hq/saved_navmesh"

#         with open(self.human_motion_json_path, 'r') as f:
#             self.human_motion_data = json.load(f)

#         self.obj_templates_mgr = self._env._sim.get_object_template_manager()
#         logger.info("Simulator resources initialized.")

#     def _signal_sender(self):
#         """Child thread function to send signals to the main thread every 0.5 seconds."""
#         logger.info("Signal sender thread started.")
#         while True:
#             try:
#                 # Send a signal to refresh the human model if queue is not full
#                 self.signal_queue.put('REFRESH_HUMAN_MODEL', timeout=1)
#                 self.total_signals_sent += 1  # Increment total signals sent
#                 # logger.info(f"Signal sent to main thread to refresh human model. Total signals sent: {self.total_signals_sent}")
#             except queue.Full:
#                 # logger.warning("Signal queue is full. Cannot send more signals.")
#                 pass
#             # Wait for 0.5 seconds before sending the next signal
#             time.sleep(0.1)

#     def _handle_signals(self):
#         """Main thread method to handle signals from the child thread."""
#         try:
#             signals_processed = 0
#             while not self.signal_queue.empty():
#                 signal = self.signal_queue.get_nowait()
#                 signals_processed += 1
#                 # No need to process the signal content, we just count them
#             # if signals_processed > 0:
#             #     # Calculate frame ID based on total signals sent
#             #     frame_id = (self.total_signals_sent - 1) % 120
#             #     logger.info(f"Processing {signals_processed} signals. Using frame {frame_id}")
#             #     self.refresh_human_model(frame_id)
#             if signals_processed > 0:
#                 # Calculate frame ID based on total signals sent
#                 frame_id = (self.total_signals_sent - 1) % 120
                
#                 if self.total_signals_sent <= 120:
#                     # logger.info(f"Processing {signals_processed} signals. Using frame {frame_id}")
#                     # if not self.recomputed_navmesh:
#                     #     self.refresh_human_model(frame_id)
#                     #     self.frame_id = frame_id
#                     self.refresh_human_model(frame_id)
#                     self.frame_id = frame_id
#                 scan = self._env._sim._current_scene.split('/')[-2]
#                 self._env._sim.pathfinder.load_nav_mesh(os.path.join(self.nav_mesh_path, scan, scan + f'_{self.frame_id:03d}.navmesh'))
                    
#         except Exception as e:
#             logger.error(f"An error occurred while handling signals: {e}")

#     def refresh_human_model(self, frame_id):
#         """Method to refresh the human model (remove previous and add new one)."""
#         # Remove previous human model
#         self.remove_previous_human_model()
#         # Add new human model with the specified frame_id
#         self.add_new_human_model(frame_id)

#     def remove_previous_human_model(self):
#         """Remove the previous human model from the simulator."""
#         # logger.info("Removing previous human model.")
#         with self.human_model_lock:
#             if self.previous_human_object_ids:
#                 existing_ids = set(self._env._sim.get_existing_object_ids())
#                 for obj_id in self.previous_human_object_ids:
#                     if obj_id in existing_ids:
#                         self._env._sim.remove_object(obj_id)
#                         # logger.info(f"Removed object with ID {obj_id}")
#                     else:
#                         logger.warning(f"Object ID {obj_id} does not exist in simulator.")
#                         # raise
#                 self.previous_human_object_ids = []

#     def add_new_human_model(self, frame_id):
#         """Add a new human model to the simulator."""
#         # logger.info(f"Adding new human model at frame {frame_id}.")
#         scan = self._env._sim._current_scene.split('/')[-2]
#         human_motions = self.human_motion_data.get(scan, {})
#         # frame_id is passed as an argument
#         human_positions = {}

#         with self.human_model_lock:
#             self.previous_human_object_ids = []  # Reset the list for current objects

#             for viewpoint, viewpoint_data in human_motions.items():
#                 category = viewpoint_data["category"]
#                 glb_index = viewpoint_data["index"]
#                 # Get translation and rotation for the specified frame
#                 translation = viewpoint_data["translation"][frame_id]
#                 rotation_euler = viewpoint_data["rotation"][frame_id]

#                 category = category.replace(" ", "_").replace("/", "_")

#                 # template_id = self.obj_templates_mgr.load_configs('/datadrive/qih/habitat-sim/data/test_assets/objects/nested_box.object_config.json')[0]

#                 template_id = self.obj_templates_mgr.get_template_handle_by_ID(self.or_num +
#                                                 (self.category2idx[f'{category}_{glb_index}'])*120+frame_id)

#                 if translation is not None:
#                     translations = mn.Vector3(translation)
#                     rotation_quat = (
#                         mn.Quaternion.rotation(mn.Deg(rotation_euler[0]), mn.Vector3(1.0, 0.0, 0.0)) *
#                         mn.Quaternion.rotation(mn.Deg(rotation_euler[1]), mn.Vector3(0.0, 1.0, 0.0)) *
#                         mn.Quaternion.rotation(mn.Deg(rotation_euler[2]), mn.Vector3(0.0, 0.0, 1.0))
#                     )
#                     # object_id = self._env._sim.add_object(template_id)
#                     object_id = self._env._sim.add_object_by_handle(template_id)
#                     # logger.info(f"Added object with ID {object_id}")

#                     self._env._sim.set_translation(translations, object_id)
#                     self._env._sim.set_rotation(rotation_quat, object_id)
#                     rotation_quat = self._env._sim.get_rotation(object_id)
#                     human_positions[viewpoint] = (np.array(translation), rotation_euler)
#                     # print(rotation_euler, rotation_quat, rotation_quat.to_matrix())

#                     self._env._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, object_id)

#                     self._env._sim.step_physics(1.0 / 5.0)
#                     self.previous_human_object_ids.append(object_id)
            
#             self._env._sim._human_posisions = human_positions
            
#             # num_objects = len(self.previous_human_object_ids)
#             # logger.info(f"Added {num_objects} objects")

#             # navmesh_settings = habitat_sim.nav.NavMeshSettings()
#             # navmesh_settings.set_defaults()
#             # navmesh_settings.agent_radius = 0.1
#             # navmesh_settings.agent_height = 1.5
#             # self._env._sim.recompute_navmesh(self._env._sim.pathfinder, navmesh_settings, include_static_objects=True)
#             # self.recomputed_navmesh = True
    
#     def __init_manager__(self):
        
        
        # data_path = "/datadrive/vln_share/habitat-sim/data/"

        self.obj_templates_mgr = self._env._sim.get_object_template_manager()
        
        idx2category = {}
        self.category2idx = {}
        pre_num = self.obj_templates_mgr.get_num_templates()
        self.or_num = pre_num

        glb_lists = sorted(os.listdir(self.data_path))
        for i, category in enumerate(glb_lists):
            idx2category[i] = category
            self.category2idx[category] = i
            self.obj_templates_mgr.load_configs(str(f"{self.data_path}/{category}/"))
            add_num = self.obj_templates_mgr.get_num_templates() - pre_num
            assert add_num == 120
            pre_num = self.obj_templates_mgr.get_num_templates()