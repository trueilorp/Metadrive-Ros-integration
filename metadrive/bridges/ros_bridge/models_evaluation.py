from functools import partial
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, clear_output

import gymnasium as gym
from gymnasium import spaces

from metadrive.envs import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.image_obs import ImageObservation
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from sb3_contrib import TQC

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure

# ------------- READ CONFIG PARAMETERS --------------
# --- Load config file ---
with open("config.json", "r") as f:
	cfg = json.load(f)

# Extract training params
train_cfg = cfg["training"]
env_cfg = cfg["environment"]
veh_cfg = cfg["vehicle"]
model_name = cfg["model"]

# Assign training variables
total_timesteps = train_cfg["total_timesteps"]
n_steps = train_cfg["n_steps"]
batch_size = train_cfg["batch_size"]
img_shape = train_cfg["img_shape"]
buffer_size = train_cfg["buffer_size"]
learning_starts = train_cfg["learning_starts"]


# ------------- --------------- -------------- #
# ------------- --------------- -------------- #
# ------------- --------------- -------------- #

# ------------- CUSTOMIZE OBSERVATION SPACE -------------- 
class CustomizeObs(BaseObservation):
	"""
	Use camera, ego state info, navigation info and lidar as input
	"""
	IMAGE = "image"
	# STATE = "state"
	LIDAR = "state_and_lidar"

	def __init__(self, config):
		super(CustomizeObs, self).__init__(config)
		self.img_obs = ImageObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])
		# self.state_obs = StateObservation(config)
		self.lidar_obs = LidarStateObservation(config)

	@property
	def observation_space(self):
		return gym.spaces.Dict(
			{
				self.IMAGE: self.img_obs.observation_space,
				# self.STATE: self.state_obs.observation_space
				self.LIDAR: self.lidar_obs.observation_space
			}
		)

	def observe(self, vehicle: BaseVehicle):
		return {self.IMAGE: self.img_obs.observe(), 
				# self.STATE: self.state_obs.observe(vehicle), 
				self.LIDAR: self.lidar_obs.observe(vehicle)}

	def destroy(self):
		super(CustomizeObs, self).destroy()
		self.img_obs.destroy()
		# self.state_obs.destroy()
		self.lidar_obs.destroy()

# ------------- VEHICLE CONFIG --------------
vehicle_config = {
		"show_navi_mark": True,
		"enable_reverse": True,
		"show_lidar": True,
		"image_source":"rgb_camera",

		# "side_detector": {
		# 	"num_lasers": 32,
		# 	"distance": 50
		# },
		# "show_side_detector": True,

		# "lane_line_detector": {
		# 	"num_lasers": 32,
		# 	"distance": 50
		# },
		# "show_lane_line_detector": True,

		"lidar": {
			"num_lasers": veh_cfg["num_lasers"],
			"distance": veh_cfg["distance"],
			"num_others": veh_cfg["num_others"],
			"add_others_navi": veh_cfg["add_others_navi"],
			"gaussian_noise": 0.0,
			"dropout_prob": 0.0
		}
	}
	
# ------------- ENV CONFIG --------------
config = (dict(map="SC", # map="SCSTRS",
				use_render=True,
				image_observation=True,
				agent_observation=CustomizeObs, # custom observation

				manual_control=False,
				random_spawn_lane_index=False,
				num_scenarios=env_cfg["num_scenarios"],
				traffic_density=0.2,
				accident_prob=0,
				log_level=50,
				
				out_of_road_penalty=env_cfg["out_of_road_penalty"], # di default è 10
				crash_sidewalk_penalty=env_cfg["crash_sidewalk_penalty"], # di default è 0
				
				vehicle_config = vehicle_config,
				sensors={"rgb_camera": (RGBCamera, img_shape, img_shape)},
				stack_size=1 # tengo solo l'ultimo frame della camera, perchè memorizza più frame in stack
			))

# ------------- ENV CLASS FOR GYMNASIUM COMPATIBILITY --------------
class MetaDriveEnvGymnasium(MetaDriveEnv):
	def reset(self, *, seed=None, options=None):
		obs, info = super().reset(seed=seed)
		return obs, info
		
	def reward_function(self, vehicle_id: str):
		"""
		Override this func to get a new reward function
		:param vehicle_id: id of BaseVehicle
		:return: reward
		"""
		vehicle = self.agents[vehicle_id]
		step_info = dict()

		# Reward for moving forward in current lane
		if vehicle.lane in vehicle.navigation.current_ref_lanes:
			current_lane = vehicle.lane
			positive_road = 1
		else:
			current_lane = vehicle.navigation.current_ref_lanes[0]
			current_road = vehicle.navigation.current_road
			positive_road = 1 if not current_road.is_negative_road() else -1
		long_last, _ = current_lane.local_coordinates(vehicle.last_position)
		long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

		# reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
		if self.config["use_lateral_reward"]:
			lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
		else:
			lateral_factor = 1.0

		reward = 0.0
		reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
		reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

		step_info["step_reward"] = reward

		if self._is_arrive_destination(vehicle):
			reward = +self.config["success_reward"] # 10
		elif self._is_out_of_road(vehicle):
			reward = -self.config["out_of_road_penalty"] # -5
		elif vehicle.crash_vehicle:
			reward = -self.config["crash_vehicle_penalty"] # -5
		elif vehicle.crash_object:
			reward = -self.config["crash_object_penalty"] # -5
		elif vehicle.crash_sidewalk:
			reward = -self.config["crash_sidewalk_penalty"]
		step_info["route_completion"] = vehicle.navigation.route_completion

		return reward, step_info
		
# ----------- INIT ENV --------------
def create_env():
	env_base = MetaDriveEnvGymnasium(config) # creo l'ambiente --> Observation Space: Dict('image': Box(-0.0, 1.0, (900, 1200, 3, 3), float32), 'state': Box(-0.0, 1.0, (80,), float32))
	return env_base

# ----------- EVALUATION --------------
total_reward = 0
eval_env = DummyVecEnv([partial(create_env)])
eval_env.training = False

if model_name.upper() == "PPO":
	print("Loading PPO model...")
	eval_model = PPO.load(f"/home/trueilorp/metadrive/bridges/ros_bridge/models/ppo_metadrive_multimodal.zip", env=eval_env, device="cuda")
elif model_name.upper() == "DDPG":
	print("Loading DDPG model...")
	eval_model = DDPG.load(f"/home/trueilorp/metadrive/bridges/ros_bridge/models/ddpg_metadrive_multimodal.zip", env=eval_env, device="cuda")
elif model_name.upper() == "TQC":
	print("Loading TQC model...")
	eval_model = TQC.load(f"/home/trueilorp/metadrive/bridges/ros_bridge/models/tqc_metadrive_multimodal.zip", env=eval_env, device="cuda")
elif model_name.upper() == "SAC":
	print("Loading SAC model...")
	eval_model = SAC.load(f"/home/trueilorp/metadrive/bridges/ros_bridge/models/sac_metadrive_multimodal.zip", env=eval_env, device="cuda")
else:
	raise ValueError(f"Unsupported model type: {model_name}")
	
obs = eval_env.reset()

try:
	for i in range(total_timesteps):
		action, _states = eval_model.predict(obs, deterministic=True)
		print("\nPredicted Action:", action)
		obs, reward, done, info  = eval_env.step(action)
		# print("\n---------------\nObservation:", obs)
		total_reward += reward
		if done:
			print("episode_reward", total_reward)
			break # quando si schianta, esce dal loop
finally:
	eval_env.close()
	
	
'''
---------------
Observation: OrderedDict([('image', array([[[[[0.6117647 ],
          [0.60784316],
          [0.5764706 ]],

         [[0.60784316],
          [0.6       ],
          [0.5647059 ]],

         [[0.6039216 ],
          [0.5921569 ],
          [0.56078434]],

         ...,

         [[0.54901963],
          [0.48235294],
          [0.40392157]],

         [[0.5568628 ],
          [0.4862745 ],
          [0.40392157]],

         [[0.5686275 ],
          [0.49803922],
          [0.41960785]]],


        [[[0.60784316],
          [0.5882353 ],
          [0.5372549 ]],

         [[0.60784316],
          [0.58431375],
          [0.54901963]],

         [[0.62352943],
          [0.6039216 ],
          [0.57254905]],

         ...,

         [[0.5647059 ],
          [0.49019608],
          [0.4117647 ]],

         [[0.5647059 ],
          [0.49019608],
          [0.4117647 ]],

         [[0.5686275 ],
          [0.4862745 ],
          [0.40784314]]],


        [[[0.59607846],
          [0.54509807],
          [0.47058824]],

         [[0.5764706 ],
          [0.5294118 ],
          [0.45882353]],

         [[0.6039216 ],
          [0.5764706 ],
          [0.5254902 ]],

         ...,

         [[0.5686275 ],
          [0.49803922],
          [0.41960785]],

         [[0.57254905],
          [0.49019608],
          [0.40784314]],

         [[0.57254905],
          [0.49019608],
          [0.4117647 ]]],


        ...,


        [[[0.61960787],
          [0.58431375],
          [0.59607846]],

         [[0.61960787],
          [0.58431375],
          [0.59607846]],

         [[0.6039216 ],
          [0.57254905],
          [0.58431375]],

         ...,

         [[0.54901963],
          [0.5176471 ],
          [0.53333336]],

         [[0.54901963],
          [0.5176471 ],
          [0.53333336]],

         [[0.49019608],
          [0.45882353],
          [0.4745098 ]]],


        [[[0.6156863 ],
          [0.58431375],
          [0.59607846]],

         [[0.61960787],
          [0.5882353 ],
          [0.6       ]],

         [[0.61960787],
          [0.5921569 ],
          [0.6039216 ]],

         ...,

         [[0.6       ],
          [0.5647059 ],
          [0.5803922 ]],

         [[0.59607846],
          [0.5647059 ],
          [0.5764706 ]],

         [[0.5921569 ],
          [0.56078434],
          [0.5764706 ]]],


        [[[0.61960787],
          [0.5882353 ],
          [0.6       ]],

         [[0.62352943],
          [0.5921569 ],
          [0.6039216 ]],

         [[0.62352943],
          [0.5921569 ],
          [0.6039216 ]],

         ...,

         [[0.6       ],
          [0.5647059 ],
          [0.5803922 ]],

         [[0.6039216 ],
          [0.5686275 ],
          [0.58431375]],

         [[0.6       ],
          [0.5647059 ],
          [0.5803922 ]]]]], dtype=float32)), ('state_and_lidar', array([[0.23882736, 0.34450597, 0.3792187 , 0.70258933, 0.49959192,
        0.4755157 , 0.7831737 , 0.00439022, 0.2886428 , 0.75183237,
        0.06805038, 0.84269744, 1.        , 0.9277159 , 0.58002424,
        0.00644596, 0.        , 0.5       , 0.5       , 0.02745321,
        0.01737539, 0.02031884, 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 0.04505986, 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 0.5143057 , 1.        ,
        0.0550386 , 0.01724472, 0.01653957, 1.        ]], dtype=float32))])

Predicted Action: [[-0.20972602  0.5158218 ]]
'''