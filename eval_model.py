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

from nets.custom_cnn import CustomCNN
from obs.custom_obs import CustomizeObs
from env.custom_meta_env import MetaDriveEnvGymnasium

# ------------- READ CONFIG PARAMETERS --------------
# --- Load config file ---
with open("config.json", "r") as f:
	cfg = json.load(f)

# Extract training params
train_cfg = cfg["training"]
env_cfg = cfg["environment"]
veh_cfg = cfg["vehicle"]
model_name = cfg["model"]
model_zip_file = cfg["model_zip_file"]

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
	eval_model = PPO.load(os.path.join("models_trained", model_zip_file), env=eval_env, device="cuda")
elif model_name.upper() == "DDPG":
	print("Loading DDPG model...")
	eval_model = DDPG.load(os.path.join("models_trained", model_zip_file), env=eval_env, device="cuda")
elif model_name.upper() == "TQC":
	print("Loading TQC model...")
	eval_model = TQC.load(os.path.join("models_trained", model_zip_file), env=eval_env, device="cuda")
elif model_name.upper() == "SAC":
	print("Loading SAC model...")
	eval_model = SAC.load(os.path.join("models_trained", model_zip_file), env=eval_env, device="cuda")
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